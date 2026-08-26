//===- SimdSimtCostModel.h - Ascend SIMD/SIMT candidate model -*- C++ -*-===//
//
// This file exposes the target-profile-backed candidate cost model used to
// compare all-SIMD, all-SIMT, and mixed SIMD/SIMT execution for generic TTIR.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_STAGEMODEL_SIMDSIMTCOSTMODEL_H
#define ASCENDMODEL_STAGEMODEL_SIMDSIMTCOSTMODEL_H

#include "AscendModel/StageModel/SimtAnchorAnalysis.h"
#include "AscendModel/StageModel/StageRouteCostModel.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace ascend {

enum class SimdSimtCandidateKind {
  AllSIMD,
  AllSIMTOnly,
  MixedSIMDSIMT,
};

llvm::StringRef stringifySimdSimtCandidate(SimdSimtCandidateKind candidate);

/// Resource summary for the exact non-overlapping TTIR operations that the
/// current materializer can wrap in local SIMT scopes.  It intentionally
/// contains no Operation pointers, so reports remain stable and serializable.
struct SimtAnchorFeatureSummary {
  /// All mechanism anchors recognized before target/lowering checks.
  int64_t recognizedCount = 0;
  /// Materializable anchors used by the mixed candidate.  `count` is retained
  /// as the serialized compatibility name for this value.
  int64_t count = 0;
  int64_t coveredOperationCount = 0;
  int64_t loadOps = 0;
  int64_t storeOps = 0;
  int64_t reduceOps = 0;
  int64_t scanOps = 0;
  int64_t gatherOps = 0;
  int64_t dotOps = 0;
  int64_t atomicOps = 0;
  int64_t histogramOps = 0;

  int64_t maxTensorNumel = 1;
  int64_t maxElementBits = 0;
  int64_t maskRankSum = 0;
  int64_t uniqueMaskValues = 0;
  int64_t uniqueMaskRankSum = 0;
  int64_t predicateElements = 0;
  /// Loop-weighted mask lanes on predicate-producing/consuming IR edges.
  int64_t predicateLaneEvaluations = 0;
  int64_t pointerTensorOps = 0;
  int64_t loadedIndexDependentMemoryOps = 0;
  int64_t laneDependentPointerOps = 0;
  int64_t maxReduceAxisExtent = 1;
  int64_t weightedReduceAxisElements = 0;
  /// Loop-weighted input lanes times log2(reduction/scan axis extent).
  int64_t shuffleLaneSteps = 0;
  int64_t staticLoopCount = 0;
  int64_t staticLoopTripCountSum = 0;
  int64_t modeledDynamicLoopCount = 0;
  int64_t modeledDynamicLoopTripCountSum = 0;
  int64_t conditionalBranchCount = 0;
  int64_t divergentBranchCount = 0;
  double activeLaneRatio = 1.0;

  bool hasControlFlow = false;

  llvm::StringMap<int64_t> weightedOps;
  llvm::StringMap<int64_t> opElements;
  double loadBytes = 0.0;
  double storeBytes = 0.0;
  int64_t loadWarpInstructions = 0;
  int64_t storeWarpInstructions = 0;
  int64_t dotFlops = 0;

  int64_t capturedTensorCount = 0;
  int64_t escapingTensorCount = 0;
  double capturedTensorBytes = 0.0;
  double escapingTensorBytes = 0.0;

  std::vector<std::string> mechanismKinds;
  std::vector<TensorAtomicFacts> tensorAtomics;
  std::vector<HistogramFacts> histograms;
  std::vector<PlainCumsumFacts> plainCumsums;
  std::vector<TriangularSolveFacts> triangularSolves;
  CandidateLowerability kernelLowerability;

  llvm::json::Object toJSON() const;
};

/// Static, workload-name-independent properties extracted directly from a
/// generic TTIR ModuleOp.  Weighted fields include statically known scf.for
/// trip counts.
struct SimdSimtFeatureSummary {
  int64_t loadOps = 0;
  int64_t storeOps = 0;
  int64_t reduceOps = 0;
  int64_t scanOps = 0;
  int64_t gatherOps = 0;
  int64_t dotOps = 0;
  int64_t atomicOps = 0;
  int64_t histogramOps = 0;
  int64_t broadcastOps = 0;
  int64_t expandDimsOps = 0;
  int64_t splatOps = 0;
  int64_t addPtrOps = 0;

  int64_t arithOps = 0;
  int64_t mathOps = 0;
  int64_t addOps = 0;
  int64_t subOps = 0;
  int64_t mulOps = 0;
  int64_t divOps = 0;
  int64_t maxOps = 0;
  int64_t absOps = 0;
  int64_t expOps = 0;
  int64_t logOps = 0;
  int64_t cmpOps = 0;
  int64_t selectOps = 0;
  int64_t castOps = 0;
  int64_t clampOps = 0;
  int64_t scalarOps = 0;

  int64_t maxTensorRank = 0;
  int64_t maxTensorNumel = 1;
  int64_t maxElementBits = 0;
  int64_t maskTensorOps = 0;
  int64_t maskRankSum = 0;
  int64_t uniqueMaskValues = 0;
  int64_t uniqueMaskRankSum = 0;
  int64_t predicateElements = 0;
  /// Loop-weighted mask lanes on predicate-producing/consuming IR edges.
  int64_t predicateLaneEvaluations = 0;
  int64_t maskBroadcastOps = 0;
  int64_t pointerTensorOps = 0;
  int64_t pointerUnstructuredDims = 0;
  /// Real SSA provenance count.  Unlike laneDependentPointerOps, this field
  /// requires the address backward slice to reach a loaded/gathered index.
  int64_t loadedIndexDependentMemoryOps = 0;
  /// Legacy rank-based proxy retained for report compatibility.
  int64_t laneDependentPointerOps = 0;
  int64_t rowLocalReduceOps = 0;
  int64_t maxReduceAxisExtent = 1;
  int64_t weightedReduceAxisElements = 0;
  /// Loop-weighted input lanes times log2(reduction/scan axis extent).
  int64_t shuffleLaneSteps = 0;
  int64_t scalarLoadOps = 0;
  int64_t scalarStoreOps = 0;
  int64_t vectorPtrSplatOps = 0;
  int64_t vectorReduceToScalarOps = 0;
  bool rank1IndirectVectorReduce = false;

  llvm::StringMap<int64_t> weightedOps;
  llvm::StringMap<int64_t> opElements;
  double loadBytes = 0.0;
  double storeBytes = 0.0;
  int64_t loadWarpInstructions = 0;
  int64_t storeWarpInstructions = 0;
  int64_t dotFlops = 0;
  int64_t dotOutputElements = 0;
  std::vector<std::array<int64_t, 3>> dotMNK;
  int64_t staticLoopCount = 0;
  int64_t staticLoopTripCountSum = 0;
  int64_t staticLoopTripCountMax = 1;
  int64_t modeledDynamicLoopCount = 0;
  int64_t modeledDynamicLoopTripCountSum = 0;
  /// scf.for iter_args whose values feed data computation in a later
  /// iteration. These dependencies serialize a Stage roofline.
  int64_t loopCarriedDataDependencyCount = 0;
  /// Loop-carried pointer/address induction tracked separately because
  /// downstream address lowering can remove it.
  int64_t pointerInductionDependencyCount = 0;
  int64_t conditionalBranchCount = 0;
  int64_t divergentBranchCount = 0;
  double activeLaneRatio = 1.0;

  /// Scheduling/layout facts read from the transformed TTIR consumed by this
  /// model.  These make it explicit that layout merging and AutoBlockify V1
  /// ran before feature extraction rather than being guessed from source TTIR.
  bool ttirLayoutMergeApplied = false;
  int64_t coalesceFactor = 1;
  int64_t coalesceAxis = -1;
  bool autoBlockifyV1Applied = false;
  int64_t autoBlockifyV1LoopCount = 0;
  int64_t autoBlockifyV1ScheduleOpCount = 0;
  bool autoBlockifyV1HasDynamicTripCount = false;

  bool hasDot = false;
  bool hasGather = false;
  bool hasAtomic = false;
  bool hasHistogram = false;
  bool hasScan = false;
  bool hasExplicitScope = false;
  bool hasControlFlow = false;
  bool hasDynamicShape = false;
  bool hasUnknownTripCount = false;

  /// Pattern observations are diagnostic inputs to the model.  They do not
  /// create a mandatory online route: all three candidates stay selectable.
  std::vector<std::string> observedMixedKinds;

  SimtAnchorFeatureSummary simtAnchors;

  llvm::json::Object toJSON() const;
};

/// Applicability is strictly a hardware/lowering statement.  It says whether
/// transformed TTIR contains a recognized SIMT mechanism and whether the
/// current target can materialize at least one corresponding anchor.
struct SimtApplicabilityResult {
  bool mechanismDetected = false;
  bool targetSupported = false;
  bool materializable = false;
  int64_t recognizedAnchorCount = 0;
  int64_t materializableAnchorCount = 0;
  std::vector<std::string> mechanisms;
  std::vector<std::string> reasons;

  llvm::json::Object toJSON() const;
};

struct SimdSimtCandidateScores {
  double allSimd = 0.0;
  double allSimtOnly = 0.0;
  double mixedSimdSimt = 0.0;

  double get(SimdSimtCandidateKind candidate) const;
  llvm::json::Object toJSON() const;
};

struct SimdSimtCostModelOptions {
  /// Empty selects TRITON_ASCEND_SIMD_SIMT_PROFILE, then the source-tree
  /// profile compiled into AscendModelStageModel.
  std::string profilePath;
  std::string actualTarget;
  unsigned numWarps = 32;
  bool includeFeaturesInJSON = true;
  /// True only when the target backend can materialize the shared TTIR anchor
  /// plan.  Candidate costs remain reportable when false, but mixed is not
  /// eligible for selection.
  bool compileOn91095 = false;
  /// True when backend integration can wrap a mixed local scope with the
  /// AutoBlockify V1 logical-program schedule for F2/F4.
  bool scopeSuperblockMaterializable = false;
  /// True when backend integration can apply AutoBlockify V1 to a pure-SIMT
  /// kernel.  This is deliberately independent of local-scope batching.
  bool wholeKernelSuperblockMaterializable = false;
  /// Transformed logical program count after layout coalescing.  Zero means
  /// the frontend cannot provide a stable launch-size fact for this compile.
  int64_t logicalProgramCountHint = 0;
  std::string routeTransformCapabilityJSON = "{}";
};

struct SimdSimtCostReport {
  int64_t schemaVersion = 15;
  std::string model = "ascend_stage_model_v1_cpp";
  std::string profileVersion;
  std::string profileTarget;
  std::string actualTarget;
  std::string profileContentSha256;
  std::string selectionProfileContentSha256;
  std::string microbenchmarkProfileVersion;
  std::string microbenchmarkProfileTarget;
  std::string microbenchmarkProfileContentSha256;
  std::string scoreUnit;
  std::string scoreScope = "per_program_ranking_proxy";

  SimdSimtCandidateScores candidateCosts;
  SimdSimtCandidateScores candidateRatiosToBest;
  bool allSimdCandidateLegal = true;
  bool allSimtOnlyCandidateLegal = true;
  bool mixedCandidateLegal = false;
  SimdSimtCandidateKind decision = SimdSimtCandidateKind::AllSIMD;
  double bestScore = 0.0;

  bool targetCompatible = true;
  std::vector<std::string> unsupported;
  SimtApplicabilityResult applicability;

  SimdSimtFeatureSummary features;
  StageCostModelSummary stageModel;
  bool includeFeaturesInJSON = true;

  llvm::json::Object toJSON() const;
  void printJSON(llvm::raw_ostream &os, bool pretty = true) const;
};

/// Return the configured profile path.  The value is resolved in this
/// order: TRITON_ASCEND_SIMD_SIMT_PROFILE, compiled source-tree path.
std::string getDefaultSimdSimtProfilePath();

/// Analyze generic TTIR without depending on Triton C++ op classes.
llvm::Expected<SimdSimtFeatureSummary>
analyzeSimdSimtFeatures(mlir::ModuleOp module, bool compileOn91095 = true);

/// Analyze using a caller-owned immutable plan.  Selector uses this overload
/// so the exact operations charged by the mixed score are the operations later
/// marked for materialization.
llvm::Expected<SimdSimtFeatureSummary>
analyzeSimdSimtFeatures(mlir::ModuleOp module,
                        const SimtAnchorPlan &anchorPlan);

/// Analyze a ModuleOp and score all three candidates in one call.
llvm::Expected<SimdSimtCostReport>
analyzeSimdSimtCandidates(mlir::ModuleOp module,
                          const SimdSimtCostModelOptions &options = {});

llvm::Expected<SimdSimtCostReport>
analyzeSimdSimtCandidates(mlir::ModuleOp module,
                          const SimtAnchorPlan &anchorPlan,
                          const SimdSimtCostModelOptions &options = {});

} // namespace ascend
} // namespace mlir

#endif // ASCENDMODEL_STAGEMODEL_SIMDSIMTCOSTMODEL_H
