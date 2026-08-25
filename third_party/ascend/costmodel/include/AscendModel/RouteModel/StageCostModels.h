//===- StageCostModels.h - Per-stage analytical models --------*- C++ -*-===//
//
// StagePartitioner, StageCostEvaluator, and KernelRouteSolver are separate
// components.  This file defines the immutable data passed between them and
// the mode-specific StageCostModel tree used by StageCostEvaluator.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ROUTEMODEL_STAGECOSTMODELS_H
#define ASCENDMODEL_ROUTEMODEL_STAGECOSTMODELS_H

#include "AscendModel/RouteModel/StageRouteCostModel.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlir::ascend {

struct StageBoundaryGraph;

enum class StageCostModelKind {
  AutoBlockifyDispatch,
  AutoBlockifyLoop,
  ScalarIssue,
  ScalarControl,
  ScalarMath,
  IndexGeneration,
  PredicateMask,
  LoopPredicate,
  ContinuousTileMemory,
  ContinuousTileStore,
  ContinuousShortLoad,
  CachePolicyStore,
  IndirectScalarMemory,
  IndirectGatherMemory,
  IndependentPipelinedLoop,
  LoopCarriedRecurrence,
  RowwiseReduction,
  CubeRoofline,
  TinyCubeRoofline,
  ConversionPack,
};

llvm::StringRef stringifyStageCostModel(StageCostModelKind kind);
std::optional<StageCostModelKind> parseStageCostModel(llvm::StringRef name);

struct StageControlFlowRates {
  double loopBackedgeCycles = 0.0;
  double conditionalBranchCycles = 0.0;
  double divergentBranchPenaltyCycles = 0.0;
  double synchronizationCycles = 0.0;

  bool isFiniteAndNonNegative() const;
};

struct LogicalStage {
  std::string id;
  std::string description;
  StageCostModelKind costModelKind = StageCostModelKind::ScalarIssue;
  StageScheduleKind scheduleKind = StageScheduleKind::StraightLine;
  int64_t iterationCount = 1;
  StageModelFeatures features;
  StageWorkload workload;
  /// Exact TTIR ownership when StagePartition was built from an operation
  /// graph.  Feature-summary fallback partitions deliberately leave this
  /// empty and must not be treated as materialization evidence.
  std::vector<Operation *> operations;
  /// SSA values crossing the Stage boundary.  These are derived from the
  /// same exact operation ownership as `operations`; they are the contract
  /// consumed by legality checks and the scope materializer.
  std::vector<Value> liveIns;
  std::vector<Value> liveOuts;
  int64_t liveInBytes = 0;
  int64_t liveOutBytes = 0;
  /// Exact tensor traffic at the local scope boundary.  Unlike Stage
  /// live-in/live-out, these fields mirror the SSA values captured by and
  /// returned from the materialized scope.scope regions.
  int64_t localSimtScopeCount = 0;
  int64_t scopeInputTensorBytes = 0;
  int64_t scopeOutputTensorBytes = 0;
  /// Indices into the immutable SimtAnchorPlan.  A mixed route may
  /// materialize only anchors owned by Stages that the solver selected as
  /// SIMT; consuming every materializable anchor would violate the route.
  std::vector<unsigned> simtAnchorIndices;
  bool simdLegal = false;
  bool simtLegal = false;
  /// True when this Stage has exact operation ownership/live-in/live-out and
  /// can therefore become a local SIMT scope inside a mixed kernel.
  bool localSimtMaterializable = false;
  std::vector<int64_t> legalSimtFactors;
  std::vector<int64_t> localSimtFactors;
};

struct LogicalPhase {
  std::string id;
  std::string description;
  std::vector<LogicalStage> stages;
};

struct StagePartition {
  std::string domain;
  /// "operation_graph" for exact post-transform TTIR ownership, otherwise
  /// "feature_summary_fallback" for the temporary aggregate implementation.
  std::string boundarySource = "feature_summary_fallback";
  bool operationOwnershipComplete = false;
  int64_t modeledOperationCount = 0;
  std::vector<LogicalPhase> phases;
};

struct StageOperationRate {
  double throughput = 0.0;
  double factor = 1.0;
};

struct StageModeProfile {
  double setupCycles = 0.0;
  int64_t vectorWidth = 1;
  int64_t issueWidth = 1;
  llvm::StringMap<StageOperationRate> operationRates;
  double loadBytesPerCycle = 0.0;
  double storeBytesPerCycle = 0.0;
  double loadWarpInstructionsPerCycle = 0.0;
  double storeWarpInstructionsPerCycle = 0.0;
  double predicateOperationsPerCycle = 0.0;
  double shuffleLanesPerCycle = 0.0;
  double dotSetupCycles = 0.0;
  double dotFlopsPerCycle = 0.0;
  double scalarOperationsPerCycle = 0.0;
  double issueOperationsPerCycle = 0.0;
  double spillTransactionsPerCycle = 0.0;
  /// Loaded-index memory cannot use the continuous MTE/LSU throughput model.
  /// These rates operate on logical warp/transaction counts and include one
  /// uncovered dependency latency per Stage iteration.
  double indirectLoadTransactionsPerCycle = 0.0;
  double indirectStoreTransactionsPerCycle = 0.0;
  double indirectDependencyLatencyCycles = 0.0;
  StageControlFlowRates controlFlow;

  bool isValid(StageMode mode) const;
};

struct HardwareProfile {
  std::string profileVersion;
  std::string target;
  /// Logical warp groups available to one SIMT program.  This is a compile
  /// option, not a hardware constant, and bounds cross-group interleaving in
  /// recurrence Stage models.
  int64_t logicalWarpGroupCount = 1;
  /// SuperBlock hides latency, but large factors also replicate long-lived
  /// recurrence state.  These target-profile values model the resulting
  /// register/stack pressure without naming a workload.
  /// Largest factor that still gives proportional latency-hiding benefit.
  int64_t superblockUsefulFactorLimit = 1;
  /// Largest factor that may replicate loop-carried live state without an
  /// explicit persistent-state pressure charge.  This is intentionally
  /// independent from the latency-hiding limit: straight-line kernels may
  /// benefit through F4 while recurrence state becomes expensive above F2.
  int64_t superblockPersistentStatePressureFreeFactor = 1;
  double superblockPersistentStateBytesPerCycle = 1.0;
  StageModeProfile simd;
  StageModeProfile simt;
  StageTransitionCost transition;

  bool isValid() const;
};

class ProfileProvider {
public:
  explicit ProfileProvider(HardwareProfile profile);

  llvm::Expected<const HardwareProfile *>
  getSnapshot(llvm::StringRef target, llvm::StringRef profileVersion) const;

private:
  HardwareProfile profile;
};

struct StageCostModelContext {
  const LogicalStage &stage;
  const HardwareProfile &profile;
};

class StageCostModel {
public:
  virtual ~StageCostModel() = default;
  virtual StageMode getMode() const = 0;
  virtual llvm::StringRef getName() const = 0;
  virtual bool supports(StageCostModelKind kind) const = 0;
  virtual double estimate(const StageCostModelContext &context,
                          const StageImplementation &implementation,
                          const StageResourceCycles &resources) const = 0;
};

class SIMDStageCostModel : public StageCostModel {
public:
  StageMode getMode() const final { return StageMode::SIMD; }
};

class SIMTStageCostModel : public StageCostModel {
public:
  StageMode getMode() const final { return StageMode::SIMT; }
};

class StageCostModelRegistry {
public:
  static const StageCostModelRegistry &get();

  llvm::Expected<const StageCostModel *> lookup(StageMode mode,
                                                StageCostModelKind kind) const;
  llvm::Error verifyComplete() const;

private:
  StageCostModelRegistry();
  std::vector<std::unique_ptr<StageCostModel>> models;
};

class StageCostEvaluator {
public:
  explicit StageCostEvaluator(
      const StageCostModelRegistry &registry = StageCostModelRegistry::get())
      : registry(registry) {}

  llvm::Expected<StageCostTable> evaluate(const StagePartition &partition,
                                          const HardwareProfile &profile) const;

  llvm::Expected<StageCostTable> evaluate(const StageBoundaryGraph &graph,
                                          const HardwareProfile &profile) const;

private:
  const StageCostModelRegistry &registry;
};

} // namespace mlir::ascend

#endif // ASCENDMODEL_ROUTEMODEL_STAGECOSTMODELS_H
