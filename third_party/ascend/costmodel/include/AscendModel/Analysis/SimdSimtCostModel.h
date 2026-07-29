//===- SimdSimtCostModel.h - Ascend SIMD/SIMT candidate model -*- C++ -*-===//
//
// This file exposes the target-profile-backed candidate cost model used to
// compare all-SIMD, all-SIMT, and mixed SIMD/SIMT execution for generic TTIR.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ANALYSIS_SIMDSIMTCOSTMODEL_H
#define ASCENDMODEL_ANALYSIS_SIMDSIMTCOSTMODEL_H

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"

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
  int64_t maxElementBits = 32;
  int64_t maskTensorOps = 0;
  int64_t maskRankSum = 0;
  int64_t maskBroadcastOps = 0;
  int64_t pointerTensorOps = 0;
  int64_t pointerUnstructuredDims = 0;
  int64_t laneDependentPointerOps = 0;
  int64_t rowLocalReduceOps = 0;
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

  bool hasDot = false;
  bool hasGather = false;
  bool hasAtomic = false;
  bool hasHistogram = false;
  bool hasScan = false;
  bool hasExplicitScope = false;
  bool hasControlFlow = false;

  /// Pattern observations are diagnostic inputs to the model.  They do not
  /// create a mandatory online route: all three candidates stay selectable.
  std::vector<std::string> observedMixedKinds;

  llvm::json::Object toJSON() const;
};

struct SimdSimtCandidateScores {
  double allSimd = 0.0;
  double allSimtOnly = 0.0;
  double mixedSimdSimt = 0.0;

  double get(SimdSimtCandidateKind candidate) const;
  llvm::json::Object toJSON() const;
};

/// Detailed values retained so the JSON report can explain every major term
/// in the versioned analytical and structural formulas.
struct SimdSimtCostBreakdown {
  llvm::StringMap<double> simdOpSystemCycles;
  llvm::StringMap<double> simtOpSystemCycles;
  llvm::StringMap<double> structuralComponents;

  double simdComputeCycles = 0.0;
  double simtComputeCycles = 0.0;
  double simdDotCycles = 0.0;
  double simtDotCycles = 0.0;
  double simdLoadCycles = 0.0;
  double simdStoreCycles = 0.0;
  double simdMemoryCycles = 0.0;
  double simtLoadCycles = 0.0;
  double simtStoreCycles = 0.0;
  double simtMemoryCycles = 0.0;
  double simtShuffleInstructions = 0.0;
  double simtShuffleCycles = 0.0;
  double simtPredicateInstructions = 0.0;
  double simtPredicateCycles = 0.0;
  double simdSetupCycles = 0.0;
  double simtSetupCycles = 0.0;
  double simdIssuePayloadCycles = 0.0;
  double simtIssuePayloadCycles = 0.0;
  double simdAnalyticalCycles = 0.0;
  double simtAnalyticalCycles = 0.0;
  double programIssueScale = 1.0;

  double irregularDensity = 0.0;
  double tinyDotUnderfill = 0.0;
  double structuralPenaltyRatio = 0.0;
  double structuralFloorCycles = 0.0;

  double mixedSimdFraction = 0.0;
  double mixedSetupCycles = 0.0;
  double standaloneSimtSetupCycles = 0.0;
  double transitionDeltaCycles = 0.0;
  double tinyDotMixedResidualRatio = 0.0;
  int64_t measuredNumWarps = 0;
  std::string mixedCostSource;

  llvm::json::Object toJSON(const SimdSimtFeatureSummary &features) const;
};

struct SimdSimtCostModelOptions {
  /// Empty selects TRITON_ASCEND_SIMD_SIMT_PROFILE, then the source-tree
  /// profile compiled into AscendModelAnalysis.
  std::string profilePath;
  std::string actualTarget;
  unsigned numWarps = 32;
  /// Minimum relative advantage over the established all-SIMD baseline.
  /// The absolute floor is fixed at 64 selection-score cycles.
  double marginRatio = 0.10;
  bool includeFeaturesInJSON = true;
  /// Keep scoring outside the calibrated feature domain for offline/report
  /// diagnostics.  Production auto selection disables this so that coverage
  /// rejection happens before candidate-cost evaluation.
  bool scoreOutsideCalibrationCoverage = true;
};

struct SimdSimtCostReport {
  int64_t schemaVersion = 6;
  std::string model = "ascend_candidate_cost_v2_cpp";
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

  /// False means coverage rejected the kernel before any candidate cost was
  /// evaluated.  In that state the numeric score members are placeholders and
  /// are serialized as null rather than as meaningful zeros.
  bool candidateCostsEvaluated = false;
  SimdSimtCandidateScores candidateCosts;
  SimdSimtCandidateScores candidateRatiosToBest;
  SimdSimtCandidateKind decision = SimdSimtCandidateKind::AllSIMD;
  SimdSimtCandidateKind runnerUp = SimdSimtCandidateKind::AllSIMTOnly;
  double bestScore = 0.0;
  double runnerUpScore = 0.0;
  /// Alias of decisionAdvantage kept as the compact gate-facing gain field.
  double gainScore = 0.0;
  /// For non-SIMD decisions this is all_simd - best.  For an all-SIMD
  /// decision this is runner_up - best, so the baseline can win its own gate.
  double decisionAdvantage = 0.0;
  double requiredGainScore = 0.0;
  double marginRatio = 0.10;

  bool targetCompatible = true;
  bool selectionScoreValid = false;
  bool absoluteCostValid = false;
  std::string rankingConfidence = "none";
  std::string minimumConfidenceForDecision = "medium";
  std::string absoluteConfidence = "none";
  bool gatePassed = false;
  std::vector<std::string> gateReasons;
  std::vector<std::string> unsupported;
  bool calibrationCovered = false;
  std::string calibrationDomain = "out_of_calibration_domain";

  SimdSimtFeatureSummary features;
  SimdSimtCostBreakdown breakdown;
  bool includeFeaturesInJSON = true;

  llvm::json::Object toJSON() const;
  void printJSON(llvm::raw_ostream &os, bool pretty = true) const;
};

/// Return the configured profile path.  The value is resolved in this
/// order: TRITON_ASCEND_SIMD_SIMT_PROFILE, compiled source-tree path.
std::string getDefaultSimdSimtProfilePath();

/// Analyze generic TTIR without depending on Triton C++ op classes.
llvm::Expected<SimdSimtFeatureSummary>
analyzeSimdSimtFeatures(mlir::ModuleOp module);

/// Run the versioned profile formula on an already materialized feature
/// summary.  By default this preserves full out-of-coverage scoring for
/// offline diagnostics; production callers can disable it in the options.
/// This overload is useful for golden feature tests and offline tools.
llvm::Expected<SimdSimtCostReport>
estimateSimdSimtCandidates(const SimdSimtFeatureSummary &features,
                           const SimdSimtCostModelOptions &options = {});

/// Analyze a ModuleOp and, when admitted for scoring, score all three
/// candidates in one call.
llvm::Expected<SimdSimtCostReport>
analyzeSimdSimtCandidates(mlir::ModuleOp module,
                          const SimdSimtCostModelOptions &options = {});

} // namespace ascend
} // namespace mlir

#endif // ASCENDMODEL_ANALYSIS_SIMDSIMTCOSTMODEL_H
