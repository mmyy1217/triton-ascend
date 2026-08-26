//===- StageRouteCostModel.h - Logical-stage route model -------*- C++ -*-===//
//
// A kernel is represented as serial algorithm stages.  Every Stage is
// implemented entirely by SIMD or entirely by SIMT.  A mixed kernel is a
// route containing both modes; there is deliberately no mixed Stage.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_STAGEMODEL_STAGEROUTECOSTMODEL_H
#define ASCENDMODEL_STAGEMODEL_STAGEROUTECOSTMODEL_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlir::ascend {

enum class StageMode { SIMD, SIMT };
enum class StageKernelRouteKind { AllSIMD, AllSIMT, Mixed };
enum class StageScheduleKind {
  StraightLine,
  IndependentPipelined,
  LoopCarriedSerial,
  PartiallyDependent,
};

llvm::StringRef stringifyStageMode(StageMode mode);
llvm::StringRef stringifyStageKernelRoute(StageKernelRouteKind kind);
llvm::StringRef stringifyStageSchedule(StageScheduleKind kind);

struct StageImplementation {
  StageMode mode = StageMode::SIMD;
  /// SIMD always uses factor=1.  SIMT may use F1/F2/F4 when legal.
  int64_t superblockFactor = 1;

  bool isValid() const;
  llvm::json::Object toJSON() const;
};

/// Structural facts owned by one logical Stage.  Pointer induction is kept
/// separate from a true loop-carried data dependency because later address
/// lowering can remove it without serializing the Stage payload.
struct StageModelFeatures {
  bool hasLoop = false;
  bool hasLoopCarriedDataDependency = false;
  bool hasPointerInduction = false;
  bool hasContiguousMemory = false;
  bool hasIndirectMemory = false;
  bool hasReduction = false;
  bool hasDot = false;
  bool hasConversionPack = false;
  int64_t conditionalBranchCount = 0;
  int64_t divergentBranchCount = 0;
  int64_t loopBackedgeCount = 0;
  int64_t synchronizationCount = 0;
  /// Number of mutually independent loop-carried recurrence groups owned by
  /// this Stage.  Each group is serial internally, but SIMT may interleave
  /// different groups on independent warp groups.  Non-recurrence Stages and
  /// a single recurrence use one group.
  int64_t parallelRecurrenceGroupCount = 1;
  double activeLaneRatio = 1.0;
  std::string source;

  bool isValid() const;
  bool permitsSimdRoofline() const;
  llvm::json::Object toJSON() const;
};

/// Mode-independent work owned exactly once by one Stage.  Values are
/// logical elements/bytes, not mode-specific instructions or cycles.
struct StageWorkload {
  llvm::StringMap<double> operationElements;
  double scalarOperations = 0.0;
  double loadBytes = 0.0;
  double storeBytes = 0.0;
  double loadWarpInstructions = 0.0;
  double storeWarpInstructions = 0.0;
  double predicateElements = 0.0;
  double shuffleLaneSteps = 0.0;
  double dotFlops = 0.0;
  double issueElements = 0.0;
  double estimatedSpillTransactions = 0.0;
  bool paysKernelSetup = false;

  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
};

/// Resource costs for one iteration after raw Stage workload has been mapped
/// through the selected immutable hardware profile.  setup/epilogue are paid
/// once; all other fields are per iteration.
struct StageResourceCycles {
  double setup = 0.0;
  double scalar = 0.0;
  double load = 0.0;
  double store = 0.0;
  double compute = 0.0;
  double predicate = 0.0;
  double shuffle = 0.0;
  double dot = 0.0;
  double control = 0.0;
  double loopControl = 0.0;
  double branchControl = 0.0;
  double divergence = 0.0;
  double synchronization = 0.0;
  double spill = 0.0;
  double issue = 0.0;
  double criticalPath = 0.0;
  double epilogue = 0.0;

  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
};

struct StageImplementationCost {
  StageImplementation implementation;
  double totalCycles = 0.0;
  StageResourceCycles resources;
  std::string modelName;
  std::string profileVersion;
  std::string source;

  bool isValid() const;
  llvm::json::Object toJSON() const;
};

struct LogicalStageCost {
  std::string id;
  std::string description;
  std::string model;
  StageScheduleKind schedule = StageScheduleKind::StraightLine;
  int64_t iterationCount = 1;
  StageModelFeatures features;
  StageWorkload workload;
  int64_t ownedOperationCount = 0;
  int64_t liveInCount = 0;
  int64_t liveOutCount = 0;
  /// Static tensor footprint crossing the Stage boundary.  Counts alone are
  /// insufficient for a mixed route: returning tensor<8xf16> and
  /// tensor<8x1024xf16> are both one SSA value but have very different
  /// register/stack hand-off costs.
  int64_t liveInBytes = 0;
  int64_t liveOutBytes = 0;
  /// Number of primitive scope regions produced by the current immutable
  /// anchor plan when this Stage is selected as local SIMT.
  int64_t localSimtScopeCount = 0;
  int64_t scopeInputTensorBytes = 0;
  int64_t scopeOutputTensorBytes = 0;
  std::vector<unsigned> simtAnchorIndices;
  bool localSimtMaterializable = false;
  std::vector<int64_t> localSimtFactors;
  std::vector<StageImplementationCost> implementations;
  int64_t beginBoundary = -1;
  int64_t endBoundary = -1;
  std::vector<Operation *> operations;

  llvm::json::Object toJSON() const;
};

struct StageCostTable {
  std::string boundarySource;
  bool operationOwnershipComplete = false;
  int64_t modeledOperationCount = 0;
  std::string profileVersion;
  std::vector<LogicalStageCost> stages;
  int64_t boundaryCount = 0;
  std::string discoveryJSON;

  llvm::json::Object toJSON() const;
};

struct StageTransitionCost {
  double simdToSimtCycles = 0.0;
  double simtToSimdCycles = 0.0;
  /// Local scope values cross the SIMD/SIMT register-file boundary through
  /// UB.  All rates are aggregate single-vector-core rates.
  double simdUbLoadBytesPerCycle = 1.0;
  double simdUbStoreBytesPerCycle = 1.0;
  double simtUbLoadBytesPerCycle = 1.0;
  double simtUbStoreBytesPerCycle = 1.0;
  int64_t simtWarpSize = 1;
  /// Low-confidence upper-bound proxy charged once per local SIMT Scope Run
  /// only by the conservative shadow route.
  double scopeSetupProxyCycles = 0.0;
  std::string scopeSetupProxyConfidence = "none";
  std::string scopeSetupProxySource;
  std::string source;

  bool isValid() const;
  double get(StageMode from, StageMode to) const;
  llvm::json::Object toJSON() const;
};

struct ScopeRunCost {
  int64_t beginBoundary = -1;
  int64_t endBoundary = -1;
  int64_t superblockFactor = 1;
  std::vector<size_t> candidateStageIndices;
  int64_t liveInCount = 0;
  int64_t liveOutCount = 0;
  int64_t liveInTensorBytes = 0;
  int64_t liveOutTensorBytes = 0;
  double nominalTransitionCycles = 0.0;
  double setupProxyCycles = 0.0;
  double chargedTransitionCycles = 0.0;
  bool materializable = false;
  std::string rejectionReason;

  llvm::json::Object toJSON() const;
};

struct StageRoutePlan {
  StageKernelRouteKind candidate = StageKernelRouteKind::AllSIMD;
  bool legal = false;
  std::vector<StageImplementation> implementations;
  std::vector<size_t> stageIndices;
  std::vector<double> entryTransitionCycles;
  std::vector<double> logicalStageCycles;
  std::vector<ScopeRunCost> scopeRuns;
  int64_t routeSuperblockFactor = 1;
  double totalCycles = 0.0;
  std::string source;

  llvm::json::Object toJSON() const;
};

struct StageCostModelSummary {
  bool applied = false;
  std::string boundarySource;
  bool operationOwnershipComplete = false;
  int64_t modeledOperationCount = 0;
  std::string profileVersion;
  std::vector<LogicalStageCost> stages;
  int64_t boundaryCount = 0;
  std::string discoveryJSON;
  StageTransitionCost transition;
  StageRoutePlan allSimd;
  StageRoutePlan allSimt;
  StageRoutePlan mixed;
  StageRoutePlan conservativeMixed;

  llvm::json::Object toJSON() const;
};

llvm::Expected<StageCostModelSummary>
solveStageRoutes(const StageCostTable &costTable,
                 const StageTransitionCost &transition);

} // namespace mlir::ascend

#endif // ASCENDMODEL_STAGEMODEL_STAGEROUTECOSTMODEL_H
