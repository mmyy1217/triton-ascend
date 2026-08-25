//===- StageCostModels.cpp - Per-stage analytical models -----------------===//

#include "AscendModel/RouteModel/StageCostModels.h"
#include "AscendModel/RouteModel/StageDiscovery.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <system_error>

using namespace mlir;
using namespace mlir::ascend;

namespace {

static double iterations(const LogicalStage &stage) {
  return static_cast<double>(std::max<int64_t>(1, stage.iterationCount));
}

static double controlBody(const StageResourceCycles &resources) {
  return resources.control + resources.loopControl + resources.branchControl +
         resources.divergence + resources.synchronization;
}

static double serialBody(const StageResourceCycles &resources) {
  const double execution = resources.scalar + resources.load + resources.store +
                           resources.compute + resources.predicate +
                           resources.shuffle + resources.dot +
                           controlBody(resources) + resources.spill;
  // Issue is a shared front-end throughput bound, not an extra instruction
  // stream.  Adding it to execution double-counts every instruction.
  return std::max(execution, resources.issue);
}

static bool supportsAny(StageCostModelKind kind,
                        std::initializer_list<StageCostModelKind> kinds) {
  return llvm::is_contained(kinds, kind);
}

static bool permitsSimdOverlap(const LogicalStage &stage) {
  return stage.scheduleKind == StageScheduleKind::IndependentPipelined &&
         stage.features.permitsSimdRoofline();
}

static StageResourceCycles
materializeControlFlow(const LogicalStage &stage, StageMode mode,
                       StageResourceCycles resources,
                       const StageControlFlowRates &rates) {
  resources.loopControl +=
      static_cast<double>(stage.features.loopBackedgeCount) *
      rates.loopBackedgeCycles;
  resources.branchControl +=
      static_cast<double>(stage.features.conditionalBranchCount) *
      rates.conditionalBranchCycles;
  resources.synchronization +=
      static_cast<double>(stage.features.synchronizationCount) *
      rates.synchronizationCycles;
  if (mode == StageMode::SIMT) {
    resources.divergence +=
        static_cast<double>(stage.features.divergentBranchCount) *
        (1.0 - stage.features.activeLaneRatio) *
        rates.divergentBranchPenaltyCycles;
  }
  return resources;
}

static StageResourceCycles mapSIMDWorkload(const LogicalStage &stage,
                                           const StageModeProfile &profile) {
  StageResourceCycles resources;
  const StageWorkload &work = stage.workload;
  resources.setup = work.paysKernelSetup ? profile.setupCycles : 0.0;
  for (const auto &[name, elements] : work.operationElements) {
    auto rate = profile.operationRates.find(name);
    if (rate == profile.operationRates.end() || rate->second.throughput <= 0.0)
      continue;
    resources.compute +=
        std::ceil(elements / static_cast<double>(profile.vectorWidth)) /
        rate->second.throughput * rate->second.factor;
  }
  resources.scalar = work.scalarOperations / profile.scalarOperationsPerCycle;
  if (stage.features.hasIndirectMemory) {
    const double loadTransactions =
        std::max(work.loadWarpInstructions, work.loadBytes > 0.0 ? 1.0 : 0.0);
    const double storeTransactions =
        std::max(work.storeWarpInstructions, work.storeBytes > 0.0 ? 1.0 : 0.0);
    resources.load =
        loadTransactions / profile.indirectLoadTransactionsPerCycle;
    resources.store =
        storeTransactions / profile.indirectStoreTransactionsPerCycle;
    if (loadTransactions + storeTransactions > 0.0)
      resources.load += profile.indirectDependencyLatencyCycles;
  } else {
    resources.load = work.loadBytes / profile.loadBytesPerCycle;
    resources.store = work.storeBytes / profile.storeBytesPerCycle;
  }
  resources.predicate = std::ceil(work.predicateElements /
                                  static_cast<double>(profile.vectorWidth)) /
                        profile.predicateOperationsPerCycle;
  resources.shuffle = work.shuffleLaneSteps / profile.shuffleLanesPerCycle;
  if (work.dotFlops > 0.0) {
    resources.setup += profile.dotSetupCycles;
    resources.dot = work.dotFlops / profile.dotFlopsPerCycle;
  }
  resources.issue =
      std::ceil(work.issueElements / static_cast<double>(profile.issueWidth)) /
      profile.issueOperationsPerCycle;
  resources.spill =
      work.estimatedSpillTransactions / profile.spillTransactionsPerCycle;
  if (stage.features.hasLoopCarriedDataDependency)
    resources.criticalPath = resources.scalar + resources.compute +
                             resources.predicate + resources.shuffle +
                             resources.dot;
  else if (stage.features.hasReduction)
    resources.criticalPath =
        resources.compute + resources.predicate + resources.shuffle;
  return materializeControlFlow(stage, StageMode::SIMD, resources,
                                profile.controlFlow);
}

static StageResourceCycles mapSIMTWorkload(const LogicalStage &stage,
                                           const StageModeProfile &profile) {
  StageResourceCycles resources;
  const StageWorkload &work = stage.workload;
  resources.setup = work.paysKernelSetup ? profile.setupCycles : 0.0;
  for (const auto &[name, elements] : work.operationElements) {
    auto rate = profile.operationRates.find(name);
    if (rate == profile.operationRates.end() || rate->second.throughput <= 0.0)
      continue;
    resources.compute +=
        elements / rate->second.throughput * rate->second.factor;
  }
  resources.scalar = work.scalarOperations / profile.scalarOperationsPerCycle;
  if (stage.features.hasIndirectMemory) {
    const double loadTransactions =
        std::max(work.loadWarpInstructions, work.loadBytes > 0.0 ? 1.0 : 0.0);
    const double storeTransactions =
        std::max(work.storeWarpInstructions, work.storeBytes > 0.0 ? 1.0 : 0.0);
    resources.load =
        loadTransactions / profile.indirectLoadTransactionsPerCycle;
    resources.store =
        storeTransactions / profile.indirectStoreTransactionsPerCycle;
    if (loadTransactions + storeTransactions > 0.0)
      resources.load += profile.indirectDependencyLatencyCycles;
  } else {
    resources.load =
        work.loadWarpInstructions / profile.loadWarpInstructionsPerCycle;
    resources.store =
        work.storeWarpInstructions / profile.storeWarpInstructionsPerCycle;
  }
  resources.predicate =
      work.predicateElements / profile.predicateOperationsPerCycle;
  resources.shuffle = work.shuffleLaneSteps / profile.shuffleLanesPerCycle;
  if (work.dotFlops > 0.0) {
    resources.setup += profile.dotSetupCycles;
    resources.dot = work.dotFlops / profile.dotFlopsPerCycle;
  }
  resources.issue =
      std::ceil(work.issueElements / static_cast<double>(profile.issueWidth)) /
      profile.issueOperationsPerCycle;
  resources.spill =
      work.estimatedSpillTransactions / profile.spillTransactionsPerCycle;
  if (stage.features.hasLoopCarriedDataDependency)
    resources.criticalPath = resources.scalar + resources.compute +
                             resources.predicate + resources.shuffle +
                             resources.dot;
  else if (stage.features.hasReduction)
    resources.criticalPath =
        resources.compute + resources.predicate + resources.shuffle;
  return materializeControlFlow(stage, StageMode::SIMT, resources,
                                profile.controlFlow);
}

static double applySuperBlock(const LogicalStage &stage,
                              const StageResourceCycles &resources,
                              const StageImplementation &implementation,
                              const HardwareProfile &profile,
                              double stageCycles) {
  if (implementation.mode != StageMode::SIMT ||
      implementation.superblockFactor == 1)
    return stageCycles;
  const double factor = static_cast<double>(implementation.superblockFactor);
  const double effectiveFactor = std::min(
      factor, static_cast<double>(profile.superblockUsefulFactorLimit));
  const double latencySensitivePerIteration = resources.load + resources.store +
                                              resources.shuffle +
                                              resources.divergence;
  const double latencySensitive =
      iterations(stage) * latencySensitivePerIteration;
  // SuperBlock creates `factor` independent logical-program groups on one
  // physical core.  It can hide latency across those groups, but it cannot
  // divide dependent arithmetic, loop control, or synchronization.
  const double pressure =
      iterations(stage) * resources.spill * std::max(0.0, factor - 1.0);
  // A loop-carried Stage keeps its live-out state across iterations.  Factors
  // above the target's pressure-free point replicate that persistent state
  // across more logical warp groups.  This is a resource cost, not a blanket
  // per-factor penalty: straight-line and small-state Stages are unaffected.
  const double persistentStatePressure =
      stage.features.hasLoopCarriedDataDependency
          ? std::max(
                0.0,
                factor -
                    static_cast<double>(
                        profile.superblockPersistentStatePressureFreeFactor)) *
                static_cast<double>(stage.liveOutBytes) /
                profile.superblockPersistentStateBytesPerCycle
          : 0.0;
  const double issueFloor = resources.setup +
                            iterations(stage) * resources.issue +
                            resources.epilogue;
  // Persistent-state pressure is additional register/stack work and cannot
  // disappear behind the ordinary issue floor.
  return std::max(issueFloor, stageCycles - latencySensitive +
                                  latencySensitive / effectiveFactor +
                                  pressure) +
         persistentStatePressure;
}

class SIMDDispatchStageCostModel final : public SIMDStageCostModel {
public:
  llvm::StringRef getName() const override { return "simd_dispatch"; }
  bool supports(StageCostModelKind kind) const override {
    return supportsAny(kind, {StageCostModelKind::AutoBlockifyDispatch,
                              StageCostModelKind::AutoBlockifyLoop});
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    const double count =
        context.stage.costModelKind == StageCostModelKind::AutoBlockifyLoop
            ? iterations(context.stage)
            : 1.0;
    return r.setup + count * std::max(r.scalar + controlBody(r), r.issue) +
           r.epilogue;
  }
};

class SIMTDispatchStageCostModel final : public SIMTStageCostModel {
public:
  llvm::StringRef getName() const override { return "simt_dispatch"; }
  bool supports(StageCostModelKind kind) const override {
    return supportsAny(kind, {StageCostModelKind::AutoBlockifyDispatch,
                              StageCostModelKind::AutoBlockifyLoop});
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    const double count =
        context.stage.costModelKind == StageCostModelKind::AutoBlockifyLoop
            ? iterations(context.stage)
            : 1.0;
    return r.setup + count * std::max(r.scalar + controlBody(r), r.issue) +
           r.epilogue;
  }
};

class SIMDScalarStageCostModel final : public SIMDStageCostModel {
public:
  llvm::StringRef getName() const override { return "simd_scalar"; }
  bool supports(StageCostModelKind kind) const override {
    return supportsAny(
        kind,
        {StageCostModelKind::ScalarIssue, StageCostModelKind::ScalarControl,
         StageCostModelKind::ScalarMath, StageCostModelKind::IndexGeneration,
         StageCostModelKind::PredicateMask, StageCostModelKind::LoopPredicate});
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    return r.setup + iterations(context.stage) * serialBody(r) + r.epilogue;
  }
};

class SIMTScalarStageCostModel final : public SIMTStageCostModel {
public:
  llvm::StringRef getName() const override { return "simt_scalar"; }
  bool supports(StageCostModelKind kind) const override {
    return supportsAny(
        kind,
        {StageCostModelKind::ScalarIssue, StageCostModelKind::ScalarControl,
         StageCostModelKind::ScalarMath, StageCostModelKind::IndexGeneration,
         StageCostModelKind::PredicateMask, StageCostModelKind::LoopPredicate});
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    return r.setup + iterations(context.stage) * serialBody(r) + r.epilogue;
  }
};

class SIMDContinuousMemoryStageCostModel final : public SIMDStageCostModel {
public:
  llvm::StringRef getName() const override { return "simd_continuous_memory"; }
  bool supports(StageCostModelKind kind) const override {
    return supportsAny(kind, {StageCostModelKind::ContinuousTileMemory,
                              StageCostModelKind::ContinuousTileStore,
                              StageCostModelKind::ContinuousShortLoad,
                              StageCostModelKind::CachePolicyStore});
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    double body = serialBody(r);
    if (permitsSimdOverlap(context.stage))
      body = r.scalar + r.predicate + controlBody(r) + r.spill +
             std::max({r.load, r.store, r.issue});
    return r.setup + iterations(context.stage) * body + r.epilogue;
  }
};

class SIMTContinuousMemoryStageCostModel final : public SIMTStageCostModel {
public:
  llvm::StringRef getName() const override { return "simt_continuous_memory"; }
  bool supports(StageCostModelKind kind) const override {
    return supportsAny(kind, {StageCostModelKind::ContinuousTileMemory,
                              StageCostModelKind::ContinuousTileStore,
                              StageCostModelKind::ContinuousShortLoad,
                              StageCostModelKind::CachePolicyStore});
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    return r.setup + iterations(context.stage) * serialBody(r) + r.epilogue;
  }
};

class SIMDIndirectMemoryStageCostModel final : public SIMDStageCostModel {
public:
  llvm::StringRef getName() const override { return "simd_indirect_memory"; }
  bool supports(StageCostModelKind kind) const override {
    return supportsAny(kind, {StageCostModelKind::IndirectScalarMemory,
                              StageCostModelKind::IndirectGatherMemory});
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    return r.setup + iterations(context.stage) * serialBody(r) + r.epilogue;
  }
};

class SIMTIndirectMemoryStageCostModel final : public SIMTStageCostModel {
public:
  llvm::StringRef getName() const override { return "simt_indirect_memory"; }
  bool supports(StageCostModelKind kind) const override {
    return supportsAny(kind, {StageCostModelKind::IndirectScalarMemory,
                              StageCostModelKind::IndirectGatherMemory});
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    return r.setup + iterations(context.stage) * serialBody(r) + r.epilogue;
  }
};

class SIMDIndependentStageCostModel final : public SIMDStageCostModel {
public:
  llvm::StringRef getName() const override {
    return "simd_independent_pipeline";
  }
  bool supports(StageCostModelKind kind) const override {
    return kind == StageCostModelKind::IndependentPipelinedLoop;
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    double body = serialBody(r);
    if (permitsSimdOverlap(context.stage))
      body = std::max({r.load, r.store, r.compute + r.dot + r.shuffle,
                       r.scalar + r.predicate + controlBody(r), r.issue}) +
             r.spill;
    return r.setup + iterations(context.stage) * body + r.epilogue;
  }
};

class SIMTIndependentStageCostModel final : public SIMTStageCostModel {
public:
  llvm::StringRef getName() const override {
    return "simt_independent_pipeline";
  }
  bool supports(StageCostModelKind kind) const override {
    return kind == StageCostModelKind::IndependentPipelinedLoop;
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    return r.setup + iterations(context.stage) * serialBody(r) + r.epilogue;
  }
};

class SIMDRecurrenceStageCostModel final : public SIMDStageCostModel {
public:
  llvm::StringRef getName() const override { return "simd_recurrence"; }
  bool supports(StageCostModelKind kind) const override {
    return kind == StageCostModelKind::LoopCarriedRecurrence;
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    const double critical = r.criticalPath > 0.0
                                ? std::max(r.criticalPath + r.load + r.store +
                                               controlBody(r) + r.spill,
                                           r.issue)
                                : serialBody(r);
    return r.setup + iterations(context.stage) * critical + r.epilogue;
  }
};

class SIMTRecurrenceStageCostModel final : public SIMTStageCostModel {
public:
  llvm::StringRef getName() const override { return "simt_recurrence"; }
  bool supports(StageCostModelKind kind) const override {
    return kind == StageCostModelKind::LoopCarriedRecurrence;
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    const double criticalPerIteration =
        r.criticalPath > 0.0 ? std::max(r.criticalPath + r.load + r.store +
                                            controlBody(r) + r.spill,
                                        r.issue)
                             : serialBody(r);
    const int64_t independentGroups = std::max<int64_t>(
        1, context.stage.features.parallelRecurrenceGroupCount);
    const int64_t interleavedGroups = std::max<int64_t>(
        1, std::min(independentGroups, context.profile.logicalWarpGroupCount));
    // A carried dependency serializes iterations within one group, not
    // sibling recurrences with disjoint state.  SIMT can interleave those
    // groups, while the aggregate front-end issue stream remains a hard
    // throughput floor over all iterations.
    const double criticalIterations = std::ceil(
        iterations(context.stage) / static_cast<double>(interleavedGroups));
    const double criticalPath = criticalIterations * criticalPerIteration;
    const double issueFloor = iterations(context.stage) * r.issue;
    return r.setup + std::max(criticalPath, issueFloor) + r.epilogue;
  }
};

class SIMDReductionStageCostModel final : public SIMDStageCostModel {
public:
  llvm::StringRef getName() const override { return "simd_reduction"; }
  bool supports(StageCostModelKind kind) const override {
    return kind == StageCostModelKind::RowwiseReduction;
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    // Tree/shuffle depth is a dependency chain, while the shared issue rate
    // remains a lower bound over the complete instruction stream.
    const double execution =
        r.scalar + r.load + r.store + r.criticalPath + controlBody(r) + r.spill;
    return r.setup + iterations(context.stage) * std::max(execution, r.issue) +
           r.epilogue;
  }
};

class SIMTReductionStageCostModel final : public SIMTStageCostModel {
public:
  llvm::StringRef getName() const override { return "simt_reduction"; }
  bool supports(StageCostModelKind kind) const override {
    return kind == StageCostModelKind::RowwiseReduction;
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    const double execution =
        r.scalar + r.load + r.store + r.criticalPath + controlBody(r) + r.spill;
    return r.setup + iterations(context.stage) * std::max(execution, r.issue) +
           r.epilogue;
  }
};

class SIMDCubeStageCostModel final : public SIMDStageCostModel {
public:
  llvm::StringRef getName() const override { return "simd_cube"; }
  bool supports(StageCostModelKind kind) const override {
    return supportsAny(kind, {StageCostModelKind::CubeRoofline,
                              StageCostModelKind::TinyCubeRoofline});
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    double body = serialBody(r);
    if (permitsSimdOverlap(context.stage))
      body = r.scalar + r.predicate + controlBody(r) + r.shuffle + r.spill +
             std::max({r.load, r.compute + r.dot, r.store, r.issue});
    return r.setup + iterations(context.stage) * body + r.epilogue;
  }
};

class SIMTCubeStageCostModel final : public SIMTStageCostModel {
public:
  llvm::StringRef getName() const override { return "simt_dot"; }
  bool supports(StageCostModelKind kind) const override {
    return supportsAny(kind, {StageCostModelKind::CubeRoofline,
                              StageCostModelKind::TinyCubeRoofline});
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    return r.setup + iterations(context.stage) * serialBody(r) + r.epilogue;
  }
};

class SIMDConversionPackStageCostModel final : public SIMDStageCostModel {
public:
  llvm::StringRef getName() const override { return "simd_conversion_pack"; }
  bool supports(StageCostModelKind kind) const override {
    return kind == StageCostModelKind::ConversionPack;
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    double body = serialBody(r);
    if (permitsSimdOverlap(context.stage))
      body = r.predicate + controlBody(r) + r.spill +
             std::max({r.scalar + r.compute, r.load, r.store, r.issue});
    return r.setup + iterations(context.stage) * body + r.epilogue;
  }
};

class SIMTConversionPackStageCostModel final : public SIMTStageCostModel {
public:
  llvm::StringRef getName() const override { return "simt_conversion_pack"; }
  bool supports(StageCostModelKind kind) const override {
    return kind == StageCostModelKind::ConversionPack;
  }
  double estimate(const StageCostModelContext &context,
                  const StageImplementation &implementation,
                  const StageResourceCycles &resources) const override {
    const StageResourceCycles &r = resources;
    return r.setup + iterations(context.stage) * serialBody(r) + r.epilogue;
  }
};

static bool isDeclaredLegal(const LogicalStage &stage,
                            const StageImplementation &implementation) {
  if (!implementation.isValid())
    return false;
  if (implementation.mode == StageMode::SIMD)
    return stage.simdLegal && implementation.superblockFactor == 1;
  if (!stage.simtLegal)
    return false;
  return llvm::is_contained(stage.legalSimtFactors,
                            implementation.superblockFactor);
}

} // namespace

llvm::StringRef mlir::ascend::stringifyStageCostModel(StageCostModelKind kind) {
  switch (kind) {
  case StageCostModelKind::AutoBlockifyDispatch:
    return "auto_blockify_dispatch";
  case StageCostModelKind::AutoBlockifyLoop:
    return "auto_blockify_loop";
  case StageCostModelKind::ScalarIssue:
    return "scalar_issue";
  case StageCostModelKind::ScalarControl:
    return "scalar_control";
  case StageCostModelKind::ScalarMath:
    return "scalar_math";
  case StageCostModelKind::IndexGeneration:
    return "index_generation";
  case StageCostModelKind::PredicateMask:
    return "predicate_mask";
  case StageCostModelKind::LoopPredicate:
    return "loop_predicate";
  case StageCostModelKind::ContinuousTileMemory:
    return "continuous_tile_memory";
  case StageCostModelKind::ContinuousTileStore:
    return "continuous_tile_store";
  case StageCostModelKind::ContinuousShortLoad:
    return "continuous_short_load";
  case StageCostModelKind::CachePolicyStore:
    return "cache_policy_store";
  case StageCostModelKind::IndirectScalarMemory:
    return "indirect_scalar_memory";
  case StageCostModelKind::IndirectGatherMemory:
    return "indirect_gather_memory";
  case StageCostModelKind::IndependentPipelinedLoop:
    return "independent_pipelined_loop";
  case StageCostModelKind::LoopCarriedRecurrence:
    return "loop_carried_recurrence";
  case StageCostModelKind::RowwiseReduction:
    return "rowwise_reduction";
  case StageCostModelKind::CubeRoofline:
    return "cube_roofline";
  case StageCostModelKind::TinyCubeRoofline:
    return "tiny_cube_roofline";
  case StageCostModelKind::ConversionPack:
    return "conversion_pack";
  }
  llvm_unreachable("unknown StageCostModelKind");
}

std::optional<StageCostModelKind>
mlir::ascend::parseStageCostModel(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<StageCostModelKind>>(name)
      .Case("auto_blockify_dispatch", StageCostModelKind::AutoBlockifyDispatch)
      .Case("auto_blockify_loop", StageCostModelKind::AutoBlockifyLoop)
      .Case("scalar_issue", StageCostModelKind::ScalarIssue)
      .Case("scalar_control", StageCostModelKind::ScalarControl)
      .Case("scalar_math", StageCostModelKind::ScalarMath)
      .Case("index_generation", StageCostModelKind::IndexGeneration)
      .Case("predicate_mask", StageCostModelKind::PredicateMask)
      .Case("loop_predicate", StageCostModelKind::LoopPredicate)
      .Case("continuous_tile_memory", StageCostModelKind::ContinuousTileMemory)
      .Case("continuous_tile_store", StageCostModelKind::ContinuousTileStore)
      .Case("continuous_short_load", StageCostModelKind::ContinuousShortLoad)
      .Case("cache_policy_store", StageCostModelKind::CachePolicyStore)
      .Case("indirect_scalar_memory", StageCostModelKind::IndirectScalarMemory)
      .Case("indirect_gather_memory", StageCostModelKind::IndirectGatherMemory)
      .Case("independent_pipelined_loop",
            StageCostModelKind::IndependentPipelinedLoop)
      .Case("loop_carried_recurrence",
            StageCostModelKind::LoopCarriedRecurrence)
      .Case("rowwise_reduction", StageCostModelKind::RowwiseReduction)
      .Case("cube_roofline", StageCostModelKind::CubeRoofline)
      .Case("tiny_cube_roofline", StageCostModelKind::TinyCubeRoofline)
      .Case("conversion_pack", StageCostModelKind::ConversionPack)
      .Default(std::nullopt);
}

bool StageControlFlowRates::isFiniteAndNonNegative() const {
  const std::array<double, 4> values = {
      loopBackedgeCycles, conditionalBranchCycles, divergentBranchPenaltyCycles,
      synchronizationCycles};
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value) && value >= 0.0;
  });
}

bool StageModeProfile::isValid(StageMode mode) const {
  const std::array<double, 12> common = {setupCycles,
                                         predicateOperationsPerCycle,
                                         shuffleLanesPerCycle,
                                         dotSetupCycles,
                                         dotFlopsPerCycle,
                                         scalarOperationsPerCycle,
                                         issueOperationsPerCycle,
                                         spillTransactionsPerCycle,
                                         indirectLoadTransactionsPerCycle,
                                         indirectStoreTransactionsPerCycle,
                                         static_cast<double>(vectorWidth),
                                         static_cast<double>(issueWidth)};
  if (!std::all_of(
          common.begin(), common.end(),
          [](double value) { return std::isfinite(value) && value > 0.0; }) ||
      !std::isfinite(indirectDependencyLatencyCycles) ||
      indirectDependencyLatencyCycles < 0.0 ||
      !controlFlow.isFiniteAndNonNegative())
    return false;
  if (mode == StageMode::SIMD) {
    if (!(loadBytesPerCycle > 0.0 && storeBytesPerCycle > 0.0))
      return false;
  } else if (!(loadWarpInstructionsPerCycle > 0.0 &&
               storeWarpInstructionsPerCycle > 0.0)) {
    return false;
  }
  return llvm::all_of(operationRates, [](const auto &entry) {
    return std::isfinite(entry.second.throughput) &&
           entry.second.throughput > 0.0 &&
           std::isfinite(entry.second.factor) && entry.second.factor > 0.0;
  });
}

bool HardwareProfile::isValid() const {
  return !profileVersion.empty() && !target.empty() &&
         logicalWarpGroupCount > 0 && superblockUsefulFactorLimit > 0 &&
         superblockPersistentStatePressureFreeFactor > 0 &&
         superblockPersistentStatePressureFreeFactor <=
             superblockUsefulFactorLimit &&
         std::isfinite(superblockPersistentStateBytesPerCycle) &&
         superblockPersistentStateBytesPerCycle > 0.0 &&
         simd.isValid(StageMode::SIMD) && simt.isValid(StageMode::SIMT) &&
         transition.isValid();
}

ProfileProvider::ProfileProvider(HardwareProfile profile)
    : profile(std::move(profile)) {}

llvm::Expected<const HardwareProfile *>
ProfileProvider::getSnapshot(llvm::StringRef target,
                             llvm::StringRef profileVersion) const {
  if (!profile.isValid())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "hardware profile is invalid");
  if (!target.empty() && target != profile.target)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "hardware profile target '%s' does not match requested target '%s'",
        profile.target.c_str(), target.str().c_str());
  if (!profileVersion.empty() && profileVersion != profile.profileVersion)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "hardware profile version '%s' does not match requested version '%s'",
        profile.profileVersion.c_str(), profileVersion.str().c_str());
  return &profile;
}

StageCostModelRegistry::StageCostModelRegistry() {
  models.push_back(std::make_unique<SIMDDispatchStageCostModel>());
  models.push_back(std::make_unique<SIMTDispatchStageCostModel>());
  models.push_back(std::make_unique<SIMDScalarStageCostModel>());
  models.push_back(std::make_unique<SIMTScalarStageCostModel>());
  models.push_back(std::make_unique<SIMDContinuousMemoryStageCostModel>());
  models.push_back(std::make_unique<SIMTContinuousMemoryStageCostModel>());
  models.push_back(std::make_unique<SIMDIndirectMemoryStageCostModel>());
  models.push_back(std::make_unique<SIMTIndirectMemoryStageCostModel>());
  models.push_back(std::make_unique<SIMDIndependentStageCostModel>());
  models.push_back(std::make_unique<SIMTIndependentStageCostModel>());
  models.push_back(std::make_unique<SIMDRecurrenceStageCostModel>());
  models.push_back(std::make_unique<SIMTRecurrenceStageCostModel>());
  models.push_back(std::make_unique<SIMDReductionStageCostModel>());
  models.push_back(std::make_unique<SIMTReductionStageCostModel>());
  models.push_back(std::make_unique<SIMDCubeStageCostModel>());
  models.push_back(std::make_unique<SIMTCubeStageCostModel>());
  models.push_back(std::make_unique<SIMDConversionPackStageCostModel>());
  models.push_back(std::make_unique<SIMTConversionPackStageCostModel>());
}

const StageCostModelRegistry &StageCostModelRegistry::get() {
  static const StageCostModelRegistry registry;
  return registry;
}

llvm::Expected<const StageCostModel *>
StageCostModelRegistry::lookup(StageMode mode, StageCostModelKind kind) const {
  const StageCostModel *match = nullptr;
  for (const std::unique_ptr<StageCostModel> &model : models) {
    if (model->getMode() != mode || !model->supports(kind))
      continue;
    if (match)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "multiple StageCostModels registered for (%s, %s)",
          stringifyStageMode(mode).str().c_str(),
          stringifyStageCostModel(kind).str().c_str());
    match = model.get();
  }
  if (!match)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "no StageCostModel registered for (%s, %s)",
                                   stringifyStageMode(mode).str().c_str(),
                                   stringifyStageCostModel(kind).str().c_str());
  return match;
}

llvm::Error StageCostModelRegistry::verifyComplete() const {
  constexpr std::array<StageCostModelKind, 20> kinds = {
      StageCostModelKind::AutoBlockifyDispatch,
      StageCostModelKind::AutoBlockifyLoop,
      StageCostModelKind::ScalarIssue,
      StageCostModelKind::ScalarControl,
      StageCostModelKind::ScalarMath,
      StageCostModelKind::IndexGeneration,
      StageCostModelKind::PredicateMask,
      StageCostModelKind::LoopPredicate,
      StageCostModelKind::ContinuousTileMemory,
      StageCostModelKind::ContinuousTileStore,
      StageCostModelKind::ContinuousShortLoad,
      StageCostModelKind::CachePolicyStore,
      StageCostModelKind::IndirectScalarMemory,
      StageCostModelKind::IndirectGatherMemory,
      StageCostModelKind::IndependentPipelinedLoop,
      StageCostModelKind::LoopCarriedRecurrence,
      StageCostModelKind::RowwiseReduction,
      StageCostModelKind::CubeRoofline,
      StageCostModelKind::TinyCubeRoofline,
      StageCostModelKind::ConversionPack};
  for (StageMode mode : {StageMode::SIMD, StageMode::SIMT}) {
    for (StageCostModelKind kind : kinds) {
      auto model = lookup(mode, kind);
      if (!model)
        return model.takeError();
    }
  }
  return llvm::Error::success();
}

llvm::Expected<StageCostTable>
StageCostEvaluator::evaluate(const StagePartition &partition,
                             const HardwareProfile &profile) const {
  if (partition.domain.empty() || partition.phases.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StagePartition requires a domain and at least one Phase");
  if (!profile.isValid())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "HardwareProfile is invalid");
  if (llvm::Error error = registry.verifyComplete())
    return std::move(error);

  StageCostTable table;
  table.domain = partition.domain;
  table.boundarySource = partition.boundarySource;
  table.operationOwnershipComplete = partition.operationOwnershipComplete;
  table.modeledOperationCount = partition.modeledOperationCount;
  table.profileVersion = profile.profileVersion;
  llvm::StringSet<> stageIds;

  for (const LogicalPhase &phase : partition.phases) {
    if (phase.id.empty() || phase.stages.empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "every Phase requires an id and Stage");
    LogicalPhaseCost phaseCost;
    phaseCost.id = phase.id;
    phaseCost.description = phase.description;

    for (const LogicalStage &stage : phase.stages) {
      if (stage.id.empty() || !stageIds.insert(stage.id).second)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Stage ids must be non-empty and unique: '%s'", stage.id.c_str());
      if (stage.iterationCount <= 0 || !stage.features.isValid() ||
          !stage.workload.isFiniteAndNonNegative())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Stage '%s' has invalid iteration/features", stage.id.c_str());
      if (!stage.simdLegal && !stage.simtLegal)
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' has no legal StageMode",
                                       stage.id.c_str());
      if (stage.simtLegal && stage.legalSimtFactors.empty())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "SIMT Stage '%s' has no legal SuperBlock factor", stage.id.c_str());

      LogicalStageCost logicalCost;
      logicalCost.id = stage.id;
      logicalCost.description = stage.description;
      logicalCost.model = stringifyStageCostModel(stage.costModelKind).str();
      logicalCost.schedule = stage.scheduleKind;
      logicalCost.iterationCount = stage.iterationCount;
      logicalCost.features = stage.features;
      logicalCost.workload = stage.workload;
      logicalCost.ownedOperationCount =
          static_cast<int64_t>(stage.operations.size());
      logicalCost.liveInCount = static_cast<int64_t>(stage.liveIns.size());
      logicalCost.liveOutCount = static_cast<int64_t>(stage.liveOuts.size());
      logicalCost.liveInBytes = stage.liveInBytes;
      logicalCost.liveOutBytes = stage.liveOutBytes;
      logicalCost.localSimtScopeCount = stage.localSimtScopeCount;
      logicalCost.scopeInputTensorBytes = stage.scopeInputTensorBytes;
      logicalCost.scopeOutputTensorBytes = stage.scopeOutputTensorBytes;
      logicalCost.simtAnchorIndices = stage.simtAnchorIndices;
      logicalCost.localSimtMaterializable = stage.localSimtMaterializable;
      logicalCost.localSimtFactors = stage.localSimtFactors;
      logicalCost.operations = stage.operations;

      llvm::SmallVector<StageImplementation> implementations;
      if (stage.simdLegal)
        implementations.push_back({StageMode::SIMD, 1});
      if (stage.simtLegal)
        for (int64_t factor : stage.legalSimtFactors)
          implementations.push_back({StageMode::SIMT, factor});

      for (const StageImplementation &implementation : implementations) {
        if (!isDeclaredLegal(stage, implementation))
          return llvm::createStringError(std::errc::invalid_argument,
                                         "Stage '%s' has an illegal candidate",
                                         stage.id.c_str());
        auto model = registry.lookup(implementation.mode, stage.costModelKind);
        if (!model)
          return model.takeError();
        StageResourceCycles resources =
            implementation.mode == StageMode::SIMD
                ? mapSIMDWorkload(stage, profile.simd)
                : mapSIMTWorkload(stage, profile.simt);
        const StageCostModelContext context{stage, profile};

        StageImplementationCost cost;
        cost.implementation = implementation;
        cost.resources = resources;
        cost.modelName = (*model)->getName().str();
        cost.profileVersion = profile.profileVersion;
        cost.source =
            "post-transform TTIR StageWorkload + immutable HardwareProfile";
        cost.totalCycles = applySuperBlock(
            stage, resources, implementation, profile,
            (*model)->estimate(context, implementation, resources));
        if (!cost.isValid())
          return llvm::createStringError(std::errc::invalid_argument,
                                         "Stage '%s' produced an invalid cost",
                                         stage.id.c_str());
        logicalCost.implementations.push_back(std::move(cost));
      }

      phaseCost.stages.push_back(logicalCost);
      table.stages.push_back(std::move(logicalCost));
    }
    table.phases.push_back(std::move(phaseCost));
  }
  return table;
}

llvm::Expected<StageCostTable>
StageCostEvaluator::evaluate(const StageBoundaryGraph &graph,
                             const HardwareProfile &profile) const {
  if (graph.candidates.empty() || graph.boundaryCount() < 2)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageBoundaryGraph requires boundaries and candidate edges");
  StageCostTable table;
  table.domain = graph.domain;
  table.boundarySource = graph.boundarySource;
  table.operationOwnershipComplete = true;
  table.modeledOperationCount =
      static_cast<int64_t>(graph.dependenceGraph.units.size());
  table.profileVersion = profile.profileVersion;
  table.boundaryCount = static_cast<int64_t>(graph.boundaryCount());
  table.discoveryJSON =
      llvm::formatv("{0}", llvm::json::Value(graph.toJSON())).str();

  for (const CandidateStage &candidate : graph.candidates) {
    StagePartition partition;
    partition.domain = graph.domain;
    partition.boundarySource = graph.boundarySource;
    partition.operationOwnershipComplete = true;
    partition.modeledOperationCount =
        static_cast<int64_t>(candidate.stage.operations.size());
    LogicalPhase phase;
    phase.id = candidate.stage.id;
    phase.description = candidate.stage.description;
    phase.stages.push_back(candidate.stage);
    partition.phases.push_back(std::move(phase));
    auto evaluated = evaluate(partition, profile);
    if (!evaluated)
      return evaluated.takeError();
    LogicalStageCost cost = std::move(evaluated->stages.front());
    cost.beginBoundary = static_cast<int64_t>(candidate.beginBoundary);
    cost.endBoundary = static_cast<int64_t>(candidate.endBoundary);
    table.stages.push_back(std::move(cost));
  }
  return table;
}
