#include "AscendModel/Analysis/SimdSimtCostModel.h"

#include <gtest/gtest.h>

using mlir::ascend::SimdSimtCandidateKind;
using mlir::ascend::SimdSimtCostModelOptions;
using mlir::ascend::SimdSimtFeatureSummary;
using mlir::ascend::estimateSimdSimtCandidates;

namespace {

SimdSimtCostModelOptions options(unsigned numWarps) {
  SimdSimtCostModelOptions result;
  result.profilePath = TRITON_ASCEND_SIMD_SIMT_TEST_PROFILE_PATH;
  result.actualTarget = "Ascend950PR_9579";
  result.numWarps = numWarps;
  return result;
}

SimdSimtFeatureSummary gatherDotFeatures() {
  SimdSimtFeatureSummary f;
  f.loadOps = 3;
  f.storeOps = 1;
  f.dotOps = 1;
  f.broadcastOps = 5;
  f.expandDimsOps = 4;
  f.splatOps = 9;
  f.addPtrOps = 7;
  f.arithOps = 9;
  f.addOps = 2;
  f.mulOps = 5;
  f.scalarOps = 7;
  f.maxTensorRank = 2;
  f.maxTensorNumel = 256;
  f.maxElementBits = 32;
  f.pointerTensorOps = 11;
  f.pointerUnstructuredDims = 18;
  f.laneDependentPointerOps = 9;
  f.vectorPtrSplatOps = 4;
  f.loadBytes = 1088;
  f.storeBytes = 1024;
  f.loadWarpInstructions = 17;
  f.storeWarpInstructions = 8;
  f.dotFlops = 8192;
  f.dotOutputElements = 256;
  f.dotMNK.push_back({16, 16, 16});
  f.hasDot = true;
  f.observedMixedKinds.push_back("conditional_indirect_memory");

  f.weightedOps["add"] = 2;
  f.weightedOps["mul"] = 5;
  f.weightedOps["load"] = 3;
  f.weightedOps["store"] = 1;
  f.opElements["add"] = 32;
  f.opElements["mul"] = 50;
  f.opElements["load"] = 528;
  f.opElements["store"] = 256;
  return f;
}

SimdSimtFeatureSummary fbgemmFeatures() {
  SimdSimtFeatureSummary f;
  f.loadOps = 7;
  f.storeOps = 2;
  f.reduceOps = 1;
  f.broadcastOps = 6;
  f.expandDimsOps = 6;
  f.splatOps = 10;
  f.addPtrOps = 12;
  f.arithOps = 29;
  f.mathOps = 2;
  f.addOps = 1;
  f.mulOps = 7;
  f.divOps = 2;
  f.maxOps = 2;
  f.absOps = 2;
  f.cmpOps = 1;
  f.castOps = 2;
  f.clampOps = 2;
  f.scalarOps = 19;
  f.maxTensorRank = 2;
  f.maxTensorNumel = 64;
  f.maxElementBits = 64;
  f.maskTensorOps = 12;
  f.maskRankSum = 25;
  f.maskBroadcastOps = 4;
  f.pointerTensorOps = 19;
  f.pointerUnstructuredDims = 20;
  f.laneDependentPointerOps = 10;
  f.rowLocalReduceOps = 1;
  f.scalarLoadOps = 2;
  f.vectorPtrSplatOps = 6;
  f.loadBytes = 4152;
  f.storeBytes = 2064;
  f.loadWarpInstructions = 37;
  f.storeWarpInstructions = 17;
  f.staticLoopCount = 2;
  f.staticLoopTripCountSum = 16;
  f.staticLoopTripCountMax = 8;
  f.hasControlFlow = true;
  f.observedMixedKinds.push_back("conditional_indirect_memory");

  f.weightedOps["abs"] = 9;
  f.weightedOps["add"] = 1;
  f.weightedOps["cast"] = 2;
  f.weightedOps["clamp"] = 9;
  f.weightedOps["cmp"] = 1;
  f.weightedOps["div"] = 2;
  f.weightedOps["load"] = 21;
  f.weightedOps["max"] = 16;
  f.weightedOps["mul"] = 14;
  f.weightedOps["reduce"] = 8;
  f.weightedOps["store"] = 9;
  f.opElements["abs"] = 516;
  f.opElements["add"] = 4;
  f.opElements["cast"] = 8;
  f.opElements["clamp"] = 516;
  f.opElements["cmp"] = 4;
  f.opElements["div"] = 8;
  f.opElements["load"] = 1038;
  f.opElements["max"] = 40;
  f.opElements["mul"] = 533;
  f.opElements["reduce"] = 512;
  f.opElements["store"] = 516;
  return f;
}

SimdSimtFeatureSummary outOfCoverageFeatures() {
  SimdSimtFeatureSummary f = gatherDotFeatures();
  // One FLOP above the profile's tiny-dot coverage ceiling.  Since dotFlops is
  // non-zero, neither reduction-only coverage domain can admit this feature.
  f.dotFlops = 16385;
  return f;
}

SimdSimtFeatureSummary rank1IndirectVectorReductionFeatures() {
  SimdSimtFeatureSummary f;
  f.reduceOps = 1;
  f.maxTensorRank = 1;
  f.maxTensorNumel = 256;
  f.maxElementBits = 32;
  f.rank1IndirectVectorReduce = true;
  f.weightedOps["reduce"] = 8;
  return f;
}

} // namespace

TEST(SimdSimtCostModelTest, GatherDotGoldenScoresAndModelAdmission) {
  auto modelOptions = options(32);
  modelOptions.scoreOutsideCalibrationCoverage = false;
  auto report =
      estimateSimdSimtCandidates(gatherDotFeatures(), modelOptions);
  if (!report)
    FAIL() << llvm::toString(report.takeError());

  EXPECT_TRUE(report->candidateCostsEvaluated);
  EXPECT_NEAR(report->candidateCosts.allSimd, 1606.31661024008, 1.0e-6);
  EXPECT_NEAR(report->candidateCosts.allSimtOnly, 1396.79705238268, 1.0e-6);
  EXPECT_NEAR(report->candidateCosts.mixedSimdSimt, 1417.74900816842,
              1.0e-6);
  EXPECT_EQ(report->decision, SimdSimtCandidateKind::AllSIMTOnly);
  EXPECT_TRUE(report->selectionScoreValid);
  EXPECT_EQ(report->calibrationDomain, "tiny_irregular_dot");
  EXPECT_EQ(report->rankingConfidence, "low");
  EXPECT_TRUE(report->gatePassed);
  EXPECT_TRUE(report->gateReasons.empty());
  EXPECT_EQ(report->schemaVersion, 6);
  EXPECT_EQ(report->profileVersion,
            "david-v100-simd-simt-20260728-v5");
  EXPECT_FALSE(report->selectionProfileContentSha256.empty());
  EXPECT_EQ(report->microbenchmarkProfileVersion,
            "david-v100-shared-microbench-20260728-v1");
  EXPECT_EQ(report->microbenchmarkProfileTarget,
            "Ascend950PR/dav-c310");
  EXPECT_FALSE(report->microbenchmarkProfileContentSha256.empty());
  EXPECT_NE(report->profileContentSha256,
            report->selectionProfileContentSha256);
}

TEST(SimdSimtCostModelTest, FbgemmGoldenScoresAndModelAdmission) {
  auto modelOptions = options(4);
  modelOptions.scoreOutsideCalibrationCoverage = false;
  auto report =
      estimateSimdSimtCandidates(fbgemmFeatures(), modelOptions);
  if (!report)
    FAIL() << llvm::toString(report.takeError());

  EXPECT_TRUE(report->candidateCostsEvaluated);
  EXPECT_NEAR(report->candidateCosts.allSimd, 27870.893025909671, 1.0e-6);
  EXPECT_NEAR(report->candidateCosts.allSimtOnly, 13405.573578357647,
              1.0e-6);
  EXPECT_NEAR(report->candidateCosts.mixedSimdSimt, 20126.085626591921,
              1.0e-6);
  EXPECT_EQ(report->decision, SimdSimtCandidateKind::AllSIMTOnly);
  EXPECT_TRUE(report->selectionScoreValid);
  EXPECT_EQ(report->calibrationDomain, "masked_rowwise_reduction");
  EXPECT_EQ(report->rankingConfidence, "low");
  EXPECT_TRUE(report->gatePassed);
  EXPECT_TRUE(report->gateReasons.empty());
}

TEST(SimdSimtCostModelTest, Rank1IndirectVectorReductionIsCovered) {
  auto modelOptions = options(4);
  modelOptions.scoreOutsideCalibrationCoverage = false;
  auto report = estimateSimdSimtCandidates(
      rank1IndirectVectorReductionFeatures(), modelOptions);
  if (!report)
    FAIL() << llvm::toString(report.takeError());

  EXPECT_TRUE(report->calibrationCovered);
  EXPECT_EQ(report->calibrationDomain,
            "rank1_indirect_vector_reduction");
  EXPECT_TRUE(report->selectionScoreValid);
  EXPECT_TRUE(report->candidateCostsEvaluated);
}

TEST(SimdSimtCostModelTest,
     OutOfCoverageAutoSkipsButDiagnosticsStillScore) {
  auto autoOptions = options(32);
  autoOptions.scoreOutsideCalibrationCoverage = false;
  auto skipped =
      estimateSimdSimtCandidates(outOfCoverageFeatures(), autoOptions);
  if (!skipped)
    FAIL() << llvm::toString(skipped.takeError());

  EXPECT_FALSE(skipped->calibrationCovered);
  EXPECT_EQ(skipped->calibrationDomain,
            "out_of_calibration_domain");
  EXPECT_FALSE(skipped->selectionScoreValid);
  EXPECT_FALSE(skipped->candidateCostsEvaluated);
  EXPECT_FALSE(skipped->gatePassed);
  ASSERT_EQ(skipped->gateReasons.size(), 1u);
  EXPECT_EQ(skipped->gateReasons.front(), "selection_score_invalid");
  EXPECT_DOUBLE_EQ(skipped->candidateCosts.allSimd, 0.0);
  EXPECT_DOUBLE_EQ(skipped->candidateCosts.allSimtOnly, 0.0);
  EXPECT_DOUBLE_EQ(skipped->candidateCosts.mixedSimdSimt, 0.0);

  auto diagnosticOptions = options(32);
  diagnosticOptions.scoreOutsideCalibrationCoverage = true;
  auto diagnostic = estimateSimdSimtCandidates(
      outOfCoverageFeatures(), diagnosticOptions);
  if (!diagnostic)
    FAIL() << llvm::toString(diagnostic.takeError());

  EXPECT_FALSE(diagnostic->calibrationCovered);
  EXPECT_FALSE(diagnostic->selectionScoreValid);
  EXPECT_TRUE(diagnostic->candidateCostsEvaluated);
  EXPECT_GT(diagnostic->candidateCosts.allSimd, 0.0);
  EXPECT_GT(diagnostic->candidateCosts.allSimtOnly, 0.0);
  EXPECT_GT(diagnostic->candidateCosts.mixedSimdSimt, 0.0);
  EXPECT_FALSE(diagnostic->gatePassed);
}
