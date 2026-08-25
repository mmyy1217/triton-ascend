#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/RouteModel/StageCostModels.h"
#include "AscendModel/RouteModel/StageDiscovery.h"
#include "AscendModel/RouteModel/StagePartitioner.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

using mlir::ascend::estimateSimdSimtCandidates;
using mlir::ascend::HardwareProfile;
using mlir::ascend::LogicalPhase;
using mlir::ascend::LogicalStage;
using mlir::ascend::SimdSimtCandidateKind;
using mlir::ascend::SimdSimtCostModelOptions;
using mlir::ascend::SimdSimtFeatureSummary;
using mlir::ascend::solveStageRoutes;
using mlir::ascend::StageCostEvaluator;
using mlir::ascend::StageCostModelKind;
using mlir::ascend::StageCostModelRegistry;
using mlir::ascend::StageCostTable;
using mlir::ascend::StageDiscovery;
using mlir::ascend::StageDiscoveryOptions;
using mlir::ascend::StageFeatureAnalysis;
using mlir::ascend::StageMode;
using mlir::ascend::StageModeLegalityAnalysis;
using mlir::ascend::StagePartition;
using mlir::ascend::StagePartitioner;
using mlir::ascend::StagePartitionerOptions;
using mlir::ascend::StageScheduleKind;
using mlir::ascend::StageTransitionCost;
using mlir::ascend::StageWorkload;
using mlir::ascend::StageWorkloadAnalysis;
using mlir::ascend::TriangularSolveFacts;

namespace {

SimdSimtCostModelOptions options(unsigned numWarps) {
  SimdSimtCostModelOptions result;
  result.profilePath = TRITON_ASCEND_SIMD_SIMT_TEST_PROFILE_PATH;
  result.actualTarget = "Ascend950PR_9579";
  result.numWarps = numWarps;
  result.compileOn91095 = true;
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
  f.loadedIndexDependentMemoryOps = 2;
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
  f.simtAnchors.count = 2;
  f.simtAnchors.recognizedCount = 2;
  f.simtAnchors.loadedIndexDependentMemoryOps = 2;
  f.simtAnchors.mechanismKinds.push_back("loaded_index_dependent_memory");

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
  f.loadedIndexDependentMemoryOps = 3;
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
  f.simtAnchors.count = 3;
  f.simtAnchors.recognizedCount = 3;
  f.simtAnchors.loadedIndexDependentMemoryOps = 3;
  f.simtAnchors.mechanismKinds.push_back("loaded_index_dependent_memory");

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

SimdSimtFeatureSummary solveTrilBt16Features() {
  SimdSimtFeatureSummary f;
  f.reduceOps = 1;
  f.maxTensorRank = 2;
  f.maxTensorNumel = 256;
  f.maxElementBits = 32;
  f.maskRankSum = 20;
  f.pointerTensorOps = 5;
  f.laneDependentPointerOps = 2;
  f.staticLoopCount = 1;
  f.staticLoopTripCountSum = 1;
  f.staticLoopTripCountMax = 1;
  f.hasControlFlow = true;
  f.weightedOps["reduce"] = 1;
  return f;
}

SimdSimtFeatureSummary triangularBt16StageFeatures() {
  SimdSimtFeatureSummary f = solveTrilBt16Features();
  f.loadOps = 1;
  f.storeOps = 1;
  f.loadBytes = 16 * 16 * 4;
  f.storeBytes = 16 * 16 * 4;
  f.loadWarpInstructions = 8;
  f.storeWarpInstructions = 8;
  f.staticLoopCount = 1;
  f.staticLoopTripCountSum = 14;
  f.staticLoopTripCountMax = 14;
  f.predicateElements = 16 * 16 * 14;
  f.predicateLaneEvaluations = 16 * 16 * 14;
  f.shuffleLaneSteps = 16 * 16 * 4 * 14;
  f.simtAnchors.count = 1;
  f.simtAnchors.recognizedCount = 1;
  f.simtAnchors.reduceOps = 1;
  f.simtAnchors.predicateElements = f.predicateElements;
  f.simtAnchors.predicateLaneEvaluations = f.predicateLaneEvaluations;
  f.simtAnchors.shuffleLaneSteps = f.shuffleLaneSteps;
  f.simtAnchors.mechanismKinds.push_back("triangular_solve_loop");
  TriangularSolveFacts triangular;
  triangular.blockRows = 16;
  triangular.blockColumns = 16;
  triangular.accumulatorType = "f32";
  triangular.recurrenceStartRow = 2;
  triangular.recurrenceLoopCount = 14;
  f.simtAnchors.triangularSolves.push_back(triangular);
  return f;
}

SimdSimtFeatureSummary triangularUnknownLoopFeatures() {
  SimdSimtFeatureSummary f = solveTrilBt16Features();
  // BT64 has four sibling 16x16 recurrences. Their TTIR upper bounds remain
  // dynamic, but the full-tile structural estimate is 14 trips per loop.
  f.staticLoopCount = 4;
  f.staticLoopTripCountSum = 56;
  f.staticLoopTripCountMax = 14;
  f.modeledDynamicLoopCount = 4;
  f.modeledDynamicLoopTripCountSum = 56;
  f.hasUnknownTripCount = true;
  f.maskRankSum = 62;
  f.loadOps = 4;
  f.storeOps = 1;
  f.loadBytes = 4 * 16 * 16 * 4;
  f.storeBytes = 64 * 64 * 4;
  f.loadWarpInstructions = 32;
  f.storeWarpInstructions = 128;
  f.weightedOps["reduce"] = 56;
  f.shuffleLaneSteps = 57344;
  f.predicateLaneEvaluations = 35840;
  // The BT64 merge tail contains 16 dense 16x16x16 dots.  They are outside
  // the SIMT anchor and remain eligible for SIMD/Cube lowering.
  f.dotOps = 16;
  f.dotFlops = 16 * 2 * 16 * 16 * 16;
  f.dotOutputElements = 16 * 16 * 16;
  f.hasDot = true;
  f.simtAnchors.count = 1;
  f.simtAnchors.recognizedCount = 1;
  f.simtAnchors.reduceOps = 4;
  f.simtAnchors.maxTensorNumel = 256;
  f.simtAnchors.maskRankSum = 62;
  f.simtAnchors.staticLoopCount = 4;
  f.simtAnchors.staticLoopTripCountSum = 56;
  f.simtAnchors.modeledDynamicLoopCount = 4;
  f.simtAnchors.modeledDynamicLoopTripCountSum = 56;
  f.simtAnchors.weightedOps["reduce"] = 56;
  f.simtAnchors.shuffleLaneSteps = 57344;
  f.simtAnchors.predicateLaneEvaluations = 35840;
  f.simtAnchors.mechanismKinds.push_back("triangular_solve_loop");
  f.simtAnchors.kernelLowerability.allSimtOnly =
      mlir::ascend::CandidateLoweringStatus::BackendConditional;
  TriangularSolveFacts triangular;
  triangular.blockRows = 16;
  triangular.blockColumns = 16;
  triangular.accumulatorType = "f32";
  triangular.recurrenceStartRow = 2;
  triangular.recurrenceLoopCount = 4;
  triangular.denseDotTailOps = 16;
  triangular.requiresCubeTailPartition = true;
  f.simtAnchors.triangularSolves.push_back(triangular);
  return f;
}

} // namespace

namespace {

HardwareProfile hardwareProfile(StageTransitionCost transition = {}) {
  HardwareProfile profile;
  profile.profileVersion = "unit-test-profile-v1";
  profile.target = "Ascend950PR_9579";
  profile.superblockUsefulFactorLimit = 4;
  profile.superblockPersistentStatePressureFreeFactor = 2;
  profile.superblockPersistentStateBytesPerCycle = 8.0;
  auto fill = [](auto &mode) {
    mode.setupCycles = 10.0;
    mode.vectorWidth = 64;
    mode.issueWidth = 64;
    mode.operationRates["f32.add"] = {1.0, 1.0};
    mode.operationRates["f32.mul"] = {1.0, 1.0};
    mode.operationRates["f32.max"] = {1.0, 1.0};
    mode.operationRates["convert.cast"] = {1.0, 1.0};
    mode.loadBytesPerCycle = 32.0;
    mode.storeBytesPerCycle = 16.0;
    mode.loadWarpInstructionsPerCycle = 1.0;
    mode.storeWarpInstructionsPerCycle = 1.0;
    mode.predicateOperationsPerCycle = 1.0;
    mode.shuffleLanesPerCycle = 32.0;
    mode.dotSetupCycles = 8.0;
    mode.dotFlopsPerCycle = 64.0;
    mode.scalarOperationsPerCycle = 1.0;
    mode.issueOperationsPerCycle = 4.0;
    mode.spillTransactionsPerCycle = 1.0;
    mode.indirectLoadTransactionsPerCycle = 0.5;
    mode.indirectStoreTransactionsPerCycle = 0.5;
    mode.indirectDependencyLatencyCycles = 20.0;
    mode.controlFlow = {2.0, 3.0, 10.0, 7.0};
  };
  fill(profile.simd);
  fill(profile.simt);
  profile.simt.vectorWidth = 1;
  profile.simt.issueWidth = 32;
  profile.transition = std::move(transition);
  return profile;
}

LogicalStage
logicalStage(llvm::StringRef id, StageCostModelKind kind,
             StageScheduleKind schedule = StageScheduleKind::StraightLine,
             int64_t iterations = 1) {
  LogicalStage stage;
  stage.id = id.str();
  stage.description = id.str();
  stage.costModelKind = kind;
  stage.scheduleKind = schedule;
  stage.iterationCount = iterations;
  stage.simdLegal = true;
  stage.simtLegal = true;
  stage.legalSimtFactors = {1};
  stage.features.source = "unit_test_stage_analysis";
  stage.workload.paysKernelSetup = true;
  stage.workload.operationElements["f32.add"] = 64.0;
  stage.workload.issueElements = 4.0;
  return stage;
}

llvm::Expected<StageCostTable>
evaluateOneStage(LogicalStage stage,
                 HardwareProfile profile = hardwareProfile()) {
  LogicalPhase phase;
  phase.id = "phase";
  phase.description = "phase";
  phase.stages.push_back(std::move(stage));
  StagePartition partition;
  partition.domain = "unit_test";
  partition.phases.push_back(std::move(phase));
  return StageCostEvaluator().evaluate(partition, profile);
}

} // namespace

TEST(SimdSimtCostModelTest, GatherDotUsesStageCostEvaluator) {
  auto report = estimateSimdSimtCandidates(gatherDotFeatures(), options(32));
  if (!report)
    FAIL() << llvm::toString(report.takeError());

  EXPECT_TRUE(report->stageModel.applied);
  EXPECT_EQ(report->stageModel.domain, "indirect_underfilled_dot");
  EXPECT_GT(report->candidateCosts.allSimd, 0.0);
  EXPECT_GT(report->candidateCosts.allSimtOnly, 0.0);
  EXPECT_GT(report->candidateCosts.mixedSimdSimt, 0.0);
  for (const auto &stage : report->stageModel.stages)
    for (const auto &cost : stage.implementations) {
      EXPECT_TRUE(cost.implementation.mode == StageMode::SIMD ||
                  cost.implementation.mode == StageMode::SIMT);
      EXPECT_GT(cost.totalCycles, 0.0) << stage.id;
      EXPECT_FALSE(cost.modelName.empty());
      EXPECT_EQ(cost.profileVersion, report->profileVersion);
    }
}

TEST(SimdSimtCostModelTest, FbgemmWithoutAutoBlockifyExposesOnlySuperBlockF1) {
  auto report = estimateSimdSimtCandidates(fbgemmFeatures(), options(4));
  if (!report)
    FAIL() << llvm::toString(report.takeError());

  EXPECT_TRUE(report->stageModel.applied);
  EXPECT_EQ(report->stageModel.domain, "loaded_index_rowwise_reduction");
  EXPECT_TRUE(report->stageModel.allSimd.legal);
  EXPECT_TRUE(report->stageModel.allSimt.legal);
  EXPECT_TRUE(report->stageModel.mixed.legal);
  EXPECT_EQ(report->breakdown.mixedCostSource,
            "stage_cost_evaluator_route_sum");
  auto reduction =
      llvm::find_if(report->stageModel.stages, [](const auto &stage) {
        return stage.model == "rowwise_reduction";
      });
  ASSERT_NE(reduction, report->stageModel.stages.end());
  ASSERT_EQ(reduction->implementations.size(), 2u);
  EXPECT_EQ(reduction->implementations[0].modelName, "simd_reduction");
  EXPECT_EQ(reduction->implementations[1].modelName, "simt_reduction");
  EXPECT_EQ(reduction->implementations[1].implementation.superblockFactor, 1);
  EXPECT_GT(reduction->implementations[0].resources.criticalPath, 0.0);
}

TEST(SimdSimtCostModelTest, TriangularSolveUsesStageCostEvaluator) {
  auto report =
      estimateSimdSimtCandidates(triangularUnknownLoopFeatures(), options(32));
  if (!report)
    FAIL() << llvm::toString(report.takeError());

  EXPECT_TRUE(report->stageModel.applied);
  EXPECT_EQ(report->stageModel.domain, "triangular_recurrence");
  ASSERT_EQ(report->stageModel.phases.size(), 4u);
  ASSERT_EQ(report->stageModel.stages.size(), 5u);
  ASSERT_EQ(report->stageModel.mixed.implementations.size(), 5u);
  bool hasSimd = false;
  bool hasSimt = false;
  for (const auto &implementation : report->stageModel.mixed.implementations) {
    hasSimd |= implementation.mode == StageMode::SIMD;
    hasSimt |= implementation.mode == StageMode::SIMT;
  }
  EXPECT_TRUE(hasSimd);
  EXPECT_TRUE(hasSimt);
  EXPECT_EQ(report->stageModel.mixed.routeSuperblockFactor, 1);
  EXPECT_EQ(report->breakdown.mixedCostSource,
            "stage_cost_evaluator_route_sum");
}

TEST(SimdSimtCostModelTest, Bt16RecurrenceEvaluatesF2AndF4Pressure) {
  auto features = triangularBt16StageFeatures();
  features.autoBlockifyV1Applied = true;
  auto report = estimateSimdSimtCandidates(features, options(4));
  if (!report)
    FAIL() << llvm::toString(report.takeError());
  ASSERT_TRUE(report->stageModel.applied);
  EXPECT_EQ(report->stageModel.domain, "triangular_recurrence");
  auto recurrence =
      llvm::find_if(report->stageModel.stages, [](const auto &stage) {
        return stage.model == "loop_carried_recurrence";
      });
  ASSERT_NE(recurrence, report->stageModel.stages.end());
  ASSERT_EQ(recurrence->iterationCount, 14);
  auto findFactor = [&](int64_t factor) {
    return llvm::find_if(
        recurrence->implementations, [&](const auto &implementation) {
          return implementation.implementation.mode == StageMode::SIMT &&
                 implementation.implementation.superblockFactor == factor;
        });
  };
  const auto f2 = findFactor(2);
  const auto f4 = findFactor(4);
  ASSERT_NE(f2, recurrence->implementations.end());
  ASSERT_NE(f4, recurrence->implementations.end());
  // Feature-summary fallback has no exact SSA live-out byte count, so it can
  // model F4 latency hiding but cannot invent recurrence-state pressure.  The
  // operation-graph test below supplies liveOutBytes and verifies F4 > F2.
  EXPECT_LE(f4->totalCycles, f2->totalCycles);
}

TEST(SimdSimtCostModelTest, UnknownPatternUsesAggregateFallback) {
  auto features = triangularUnknownLoopFeatures();
  features.simtAnchors.mechanismKinds.clear();
  features.simtAnchors.triangularSolves.clear();
  features.simtAnchors.count = 0;
  auto report = estimateSimdSimtCandidates(features, options(32));
  if (!report)
    FAIL() << llvm::toString(report.takeError());
  EXPECT_FALSE(report->stageModel.applied);
  EXPECT_GT(report->candidateCosts.allSimd, 0.0);
}

TEST(SimdSimtCostModelTest, RegistryCoversEveryModeAndKind) {
  if (llvm::Error error = StageCostModelRegistry::get().verifyComplete())
    FAIL() << llvm::toString(std::move(error));
}

TEST(SimdSimtCostModelTest, StageHasOnlySimdOrSimtImplementations) {
  LogicalStage stage = logicalStage("scalar", StageCostModelKind::ScalarIssue);
  auto table = evaluateOneStage(std::move(stage));
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  ASSERT_EQ(table->stages.front().implementations.size(), 2u);
  EXPECT_EQ(table->stages.front().implementations[0].implementation.mode,
            StageMode::SIMD);
  EXPECT_EQ(table->stages.front().implementations[1].implementation.mode,
            StageMode::SIMT);
}

TEST(SimdSimtCostModelTest,
     ScopeSuperBlockLegalityOpensOnlyWithBackendMaterializer) {
  auto makePartition = [] {
    StagePartition partition;
    partition.domain = "unit_test";
    mlir::ascend::LogicalPhase phase;
    phase.id = "phase";
    LogicalStage stage =
        logicalStage("payload", StageCostModelKind::ScalarIssue);
    stage.localSimtMaterializable = true;
    stage.localSimtFactors = {1};
    phase.stages.push_back(std::move(stage));
    partition.phases.push_back(std::move(phase));
    return partition;
  };

  StagePartition f1Only = makePartition();
  if (llvm::Error error = StageModeLegalityAnalysis().analyze(f1Only, 4, false))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(f1Only.phases[0].stages[0].localSimtFactors,
            (std::vector<int64_t>{1}));

  StagePartition scopeSuperblock = makePartition();
  if (llvm::Error error =
          StageModeLegalityAnalysis().analyze(scopeSuperblock, 4, true))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(scopeSuperblock.phases[0].stages[0].localSimtFactors,
            (std::vector<int64_t>{1, 2, 4}));
}

TEST(SimdSimtCostModelTest, KernelMixedRouteComesFromAdjacentStageModes) {
  StageCostTable table;
  table.domain = "unit_test";
  table.profileVersion = "unit-test-profile-v1";
  auto addStage = [&](llvm::StringRef id, double simd, double simt) {
    mlir::ascend::LogicalStageCost stage;
    stage.id = id.str();
    stage.description = id.str();
    stage.localSimtMaterializable = true;
    stage.localSimtFactors = {1};
    auto cost = [&](StageMode mode, double cycles) {
      mlir::ascend::StageImplementationCost result;
      result.implementation = {mode, 1};
      result.totalCycles = cycles;
      result.modelName = "unit_test";
      result.profileVersion = table.profileVersion;
      result.source = "unit_test";
      return result;
    };
    stage.implementations = {cost(StageMode::SIMD, simd),
                             cost(StageMode::SIMT, simt)};
    table.stages.push_back(stage);
  };
  addStage("head", 10.0, 20.0);
  addStage("payload", 100.0, 50.0);
  addStage("store", 30.0, 45.0);
  mlir::ascend::LogicalPhaseCost phase;
  phase.id = "kernel";
  phase.stages = table.stages;
  table.phases.push_back(std::move(phase));

  StageTransitionCost transition;
  transition.simdToSimtCycles = 5.0;
  transition.simtToSimdCycles = 7.0;
  auto result = solveStageRoutes(table, transition);
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  EXPECT_DOUBLE_EQ(result->allSimd.totalCycles, 140.0);
  EXPECT_DOUBLE_EQ(result->allSimt.totalCycles, 115.0);
  EXPECT_DOUBLE_EQ(result->mixed.totalCycles, 102.0);
  ASSERT_EQ(result->mixed.implementations.size(), 3u);
  EXPECT_EQ(result->mixed.implementations[0].mode, StageMode::SIMD);
  EXPECT_EQ(result->mixed.implementations[1].mode, StageMode::SIMT);
  EXPECT_EQ(result->mixed.implementations[2].mode, StageMode::SIMD);
}

TEST(SimdSimtCostModelTest, BoundaryGraphRouteChoosesPartitionAndModeTogether) {
  StageCostTable table;
  table.domain = "boundary_graph_test";
  table.boundarySource = "stage_boundary_graph";
  table.profileVersion = "unit_test";
  table.boundaryCount = 3;
  auto makeCost = [&](StageMode mode, double cycles) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, 1};
    cost.totalCycles = cycles;
    cost.modelName = "unit_test";
    cost.profileVersion = table.profileVersion;
    cost.source = "unit_test";
    return cost;
  };
  auto makeStage = [&](llvm::StringRef id, int64_t begin, int64_t end,
                       double simd, double simt) {
    mlir::ascend::LogicalStageCost stage;
    stage.id = id.str();
    stage.description = stage.id;
    stage.beginBoundary = begin;
    stage.endBoundary = end;
    stage.localSimtMaterializable = end - begin < 2;
    stage.localSimtFactors = stage.localSimtMaterializable
                                 ? std::vector<int64_t>{1}
                                 : std::vector<int64_t>{};
    stage.implementations = {makeCost(StageMode::SIMD, simd),
                             makeCost(StageMode::SIMT, simt)};
    return stage;
  };
  table.stages = {
      makeStage("whole", 0, 2, 100.0, 100.0),
      makeStage("scalar_prefix", 0, 1, 50.0, 5.0),
      makeStage("copy_loop", 1, 2, 5.0, 50.0),
  };

  auto result = solveStageRoutes(table, StageTransitionCost{});
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  ASSERT_TRUE(result->mixed.legal);
  EXPECT_EQ(result->mixed.stageIndices, std::vector<size_t>({1, 2}));
  ASSERT_EQ(result->mixed.implementations.size(), 2u);
  EXPECT_EQ(result->mixed.implementations[0].mode, StageMode::SIMT);
  EXPECT_EQ(result->mixed.implementations[1].mode, StageMode::SIMD);
  EXPECT_DOUBLE_EQ(result->mixed.totalCycles, 10.0);
}

TEST(SimdSimtCostModelTest, MixedScopePaysExactBidirectionalUbHandoffCost) {
  StageCostTable table;
  table.domain = "scope_handoff";
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, double cycles) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, 1};
    cost.totalCycles = cycles;
    cost.modelName = "unit_test";
    cost.profileVersion = table.profileVersion;
    cost.source = "unit_test";
    return cost;
  };
  mlir::ascend::LogicalStageCost head;
  head.id = "head";
  head.description = head.id;
  head.implementations = {makeCost(StageMode::SIMD, 10.0),
                          makeCost(StageMode::SIMT, 20.0)};
  mlir::ascend::LogicalStageCost payload;
  payload.id = "large_result_payload";
  payload.description = payload.id;
  payload.localSimtMaterializable = true;
  payload.localSimtFactors = {1};
  payload.localSimtScopeCount = 2;
  payload.scopeInputTensorBytes = 4096;
  payload.scopeOutputTensorBytes = 16384;
  payload.implementations = {makeCost(StageMode::SIMD, 100.0),
                             makeCost(StageMode::SIMT, 10.0)};
  mlir::ascend::LogicalStageCost tail = head;
  tail.id = "tail";
  tail.description = tail.id;
  table.stages = {head, payload, tail};
  mlir::ascend::LogicalPhaseCost phase;
  phase.id = "phase";
  phase.stages = table.stages;
  table.phases.push_back(std::move(phase));

  StageTransitionCost transition;
  transition.simdUbLoadBytesPerCycle = 512.0;
  transition.simdUbStoreBytesPerCycle = 256.0;
  transition.simtUbLoadBytesPerThreadPerCycle = 4.0;
  transition.simtUbStoreBytesPerThreadPerCycle = 4.0;
  transition.simtWarpSize = 32;
  auto routes = solveStageRoutes(table, transition);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  // Input: 4096/256 + 4096/(4*32) = 48 cycles.
  // Output: 16384/(4*32) + 16384/512 = 160 cycles.
  // Head/payload/tail: 10 + (10 + 208) + 10 = 238 cycles.
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 238.0);
  EXPECT_GT(routes->mixed.totalCycles, routes->allSimd.totalCycles);
}

TEST(SimdSimtCostModelTest, IndependentLoopUsesSimdRooflineAndSerialSimtCost) {
  LogicalStage stage =
      logicalStage("independent", StageCostModelKind::IndependentPipelinedLoop,
                   StageScheduleKind::IndependentPipelined, 4);
  stage.features.hasLoop = true;
  stage.features.hasPointerInduction = true;

  stage.workload.loadBytes = 640.0;
  stage.workload.storeBytes = 160.0;
  stage.workload.loadWarpInstructions = 20.0;
  stage.workload.storeWarpInstructions = 10.0;
  stage.workload.dotFlops = 512.0;
  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_TRUE(stage.features.permitsSimdRoofline());
  EXPECT_LT(table->stages[0].implementations[0].totalCycles,
            table->stages[0].implementations[1].totalCycles);
}

TEST(SimdSimtCostModelTest, TrueLoopCarriedDependencyDisablesSimdRoofline) {
  LogicalStage stage =
      logicalStage("dependent", StageCostModelKind::IndependentPipelinedLoop,
                   StageScheduleKind::IndependentPipelined, 4);
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;

  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.workload.loadBytes = 640.0;
  stage.workload.storeBytes = 160.0;
  stage.workload.dotFlops = 512.0;

  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_FALSE(stage.features.permitsSimdRoofline());
  EXPECT_GT(table->stages[0].implementations[0].totalCycles, 0.0);
}

TEST(SimdSimtCostModelTest, ControlFlowUsesCountsRatesAndLaneActivity) {
  LogicalStage stage =
      logicalStage("control", StageCostModelKind::ScalarControl,
                   StageScheduleKind::StraightLine, 2);
  stage.simdLegal = false;
  stage.features.hasLoop = true;
  stage.features.conditionalBranchCount = 3;
  stage.features.divergentBranchCount = 2;
  stage.features.loopBackedgeCount = 1;
  stage.features.synchronizationCount = 1;
  stage.features.activeLaneRatio = 0.5;

  stage.workload.scalarOperations = 1.0;

  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages[0].implementations[0];
  EXPECT_DOUBLE_EQ(cost.resources.loopControl, 2.0);
  EXPECT_DOUBLE_EQ(cost.resources.branchControl, 9.0);
  EXPECT_DOUBLE_EQ(cost.resources.divergence, 10.0);
  EXPECT_DOUBLE_EQ(cost.resources.synchronization, 7.0);
  EXPECT_DOUBLE_EQ(cost.totalCycles, 196.0);
}

TEST(SimdSimtCostModelTest, RecurrenceAccumulatesCriticalPathAndTraffic) {
  LogicalStage stage =
      logicalStage("recurrence", StageCostModelKind::LoopCarriedRecurrence,
                   StageScheduleKind::LoopCarriedSerial, 4);
  stage.simdLegal = false;
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;

  stage.workload.loadWarpInstructions = 10.0;
  stage.workload.storeWarpInstructions = 5.0;
  stage.workload.estimatedSpillTransactions = 7.0;

  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_GT(table->stages[0].implementations[0].totalCycles, 100.0);
  EXPECT_GT(table->stages[0].implementations[0].resources.criticalPath, 0.0);
}

TEST(SimdSimtCostModelTest,
     SimtRecurrenceInterleavesIndependentGroupsButKeepsIssueFloor) {
  LogicalStage serial = logicalStage("serial_recurrence",
                                     StageCostModelKind::LoopCarriedRecurrence,
                                     StageScheduleKind::LoopCarriedSerial, 16);
  serial.simdLegal = false;
  serial.features.hasLoop = true;
  serial.features.hasLoopCarriedDataDependency = true;
  serial.workload.shuffleLaneSteps = 128.0;
  serial.workload.issueElements = 64.0;

  LogicalStage grouped = serial;
  grouped.id = "grouped_recurrence";
  grouped.features.parallelRecurrenceGroupCount = 4;
  HardwareProfile profile = hardwareProfile();
  profile.logicalWarpGroupCount = 4;

  auto serialTable = evaluateOneStage(std::move(serial), profile);
  auto groupedTable = evaluateOneStage(std::move(grouped), profile);
  if (!serialTable)
    FAIL() << llvm::toString(serialTable.takeError());
  if (!groupedTable)
    FAIL() << llvm::toString(groupedTable.takeError());
  const double serialCycles =
      serialTable->stages[0].implementations[0].totalCycles;
  const double groupedCycles =
      groupedTable->stages[0].implementations[0].totalCycles;
  EXPECT_LT(groupedCycles, serialCycles);
  const auto &resources = groupedTable->stages[0].implementations[0].resources;
  EXPECT_GE(groupedCycles,
            resources.setup + 16.0 * resources.issue + resources.epilogue);
}

TEST(SimdSimtCostModelTest,
     SuperBlockF4ChargesReplicatedPersistentRecurrenceState) {
  LogicalStage stage = logicalStage("stateful_recurrence",
                                    StageCostModelKind::LoopCarriedRecurrence,
                                    StageScheduleKind::LoopCarriedSerial, 16);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;
  stage.features.parallelRecurrenceGroupCount = 4;
  stage.liveOutBytes = 4096;
  stage.workload.loadWarpInstructions = 16.0;
  stage.workload.shuffleLaneSteps = 128.0;

  HardwareProfile profile = hardwareProfile();
  profile.logicalWarpGroupCount = 4;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  EXPECT_LT(costs[1].totalCycles, costs[0].totalCycles);
  EXPECT_GT(costs[2].totalCycles, costs[1].totalCycles);
}

TEST(SimdSimtCostModelTest, IndirectMemoryUsesDependencyProfile) {
  LogicalStage stage =
      logicalStage("indirect", StageCostModelKind::IndirectGatherMemory,
                   StageScheduleKind::PartiallyDependent);
  stage.features.hasIndirectMemory = true;
  stage.workload.loadBytes = 1024.0;
  stage.workload.loadWarpInstructions = 8.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.indirectLoadTransactionsPerCycle = 0.25;
  profile.simd.indirectDependencyLatencyCycles = 80.0;
  profile.simt.indirectLoadTransactionsPerCycle = 1.0;
  profile.simt.indirectDependencyLatencyCycles = 20.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 2u);
  EXPECT_DOUBLE_EQ(costs[0].resources.load, 112.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.load, 28.0);
  EXPECT_LT(costs[1].totalCycles, costs[0].totalCycles);
}

TEST(SimdSimtCostModelTest, MixedRouteRejectsUnmaterializableSimtStage) {
  StageCostTable table;
  table.domain = "unit_test";
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, double cycles) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, 1};
    cost.totalCycles = cycles;
    cost.modelName = "unit_test";
    cost.profileVersion = table.profileVersion;
    cost.source = "unit_test";
    return cost;
  };
  mlir::ascend::LogicalStageCost head;
  head.id = "head";
  head.description = head.id;
  head.localSimtMaterializable = false;
  head.implementations = {makeCost(StageMode::SIMD, 1.0),
                          makeCost(StageMode::SIMT, 100.0)};
  mlir::ascend::LogicalStageCost payload;
  payload.id = "unmaterializable_payload";
  payload.description = payload.id;
  payload.localSimtMaterializable = false;
  payload.implementations = {makeCost(StageMode::SIMD, 100.0),
                             makeCost(StageMode::SIMT, 1.0)};
  table.stages = {head, payload};
  mlir::ascend::LogicalPhaseCost phase;
  phase.id = "phase";
  phase.stages = table.stages;
  table.phases.push_back(std::move(phase));
  auto routes = solveStageRoutes(table, StageTransitionCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_TRUE(routes->allSimt.legal);
  EXPECT_FALSE(routes->mixed.legal);
}

TEST(SimdSimtCostModelTest, MixedRouteChargesEveryMaterializedScope) {
  StageCostTable table;
  table.domain = "scope_count";
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, double cycles) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, 1};
    cost.totalCycles = cycles;
    cost.modelName = "unit_test";
    cost.profileVersion = table.profileVersion;
    cost.source = "unit_test";
    return cost;
  };
  mlir::ascend::LogicalStageCost head;
  head.id = "head";
  head.description = head.id;
  head.implementations = {makeCost(StageMode::SIMD, 1.0),
                          makeCost(StageMode::SIMT, 100.0)};
  mlir::ascend::LogicalStageCost gather;
  gather.id = "two_anchor_gather";
  gather.description = gather.id;
  gather.localSimtMaterializable = true;
  gather.localSimtFactors = {1};
  gather.localSimtScopeCount = 2;
  gather.implementations = {makeCost(StageMode::SIMD, 100.0),
                            makeCost(StageMode::SIMT, 1.0)};
  mlir::ascend::LogicalStageCost tail = head;
  tail.id = "tail";
  tail.description = tail.id;
  table.stages = {head, gather, tail};
  mlir::ascend::LogicalPhaseCost phase;
  phase.id = "phase";
  phase.stages = table.stages;
  table.phases.push_back(std::move(phase));

  StageTransitionCost transition;
  transition.simdToSimtCycles = 10.0;
  transition.simtToSimdCycles = 10.0;
  auto routes = solveStageRoutes(table, transition);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  // 1 SIMD head + (10 enter + 1 payload + 20 extra scope pair) +
  // (10 leave + 1 SIMD tail).
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 43.0);
}

TEST(SimdSimtCostModelTest, SuperBlockLatencyHidingStopsAtUsefulFactorLimit) {
  LogicalStage stage =
      logicalStage("simt_payload", StageCostModelKind::ScalarIssue);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.workload.loadWarpInstructions = 40.0;
  HardwareProfile cappedProfile = hardwareProfile();
  cappedProfile.superblockUsefulFactorLimit = 2;
  cappedProfile.superblockPersistentStatePressureFreeFactor = 2;
  auto table = evaluateOneStage(stage, cappedProfile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  auto routes = solveStageRoutes(*table, cappedProfile.transition);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_TRUE(routes->allSimt.legal);
  EXPECT_EQ(routes->allSimt.routeSuperblockFactor, 2);
  EXPECT_LT(routes->allSimt.totalCycles,
            table->stages[0].implementations[0].totalCycles);
}

TEST(SimdSimtCostModelTest, PureSimtRouteUsesOneUniformSuperBlockFactor) {
  StageCostTable table;
  table.domain = "uniform_superblock";
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](int64_t factor, double cycles) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {StageMode::SIMT, factor};
    cost.totalCycles = cycles;
    cost.modelName = "unit_test";
    cost.profileVersion = table.profileVersion;
    cost.source = "unit_test";
    return cost;
  };
  mlir::ascend::LogicalStageCost first;
  first.id = "first";
  first.description = first.id;
  first.implementations = {makeCost(1, 5.0), makeCost(2, 1.0),
                           makeCost(4, 3.0)};
  mlir::ascend::LogicalStageCost second;
  second.id = "second";
  second.description = second.id;
  second.implementations = {makeCost(1, 5.0), makeCost(2, 4.0),
                            makeCost(4, 1.0)};
  table.stages = {first, second};
  mlir::ascend::LogicalPhaseCost phase;
  phase.id = "phase";
  phase.stages = table.stages;
  table.phases.push_back(std::move(phase));

  auto routes = solveStageRoutes(table, StageTransitionCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimt.legal);
  EXPECT_EQ(routes->allSimt.routeSuperblockFactor, 4);
  ASSERT_EQ(routes->allSimt.implementations.size(), 2u);
  EXPECT_EQ(routes->allSimt.implementations[0].superblockFactor, 4);
  EXPECT_EQ(routes->allSimt.implementations[1].superblockFactor, 4);
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 4.0);
}

TEST(SimdSimtCostModelTest, MixedScopeSuperBlockUsesSelectedFactorCost) {
  StageCostTable table;
  table.domain = "mixed_scope_superblock";
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, int64_t factor, double cycles) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, factor};
    cost.totalCycles = cycles;
    cost.modelName = "unit_test";
    cost.profileVersion = table.profileVersion;
    cost.source = "unit_test";
    return cost;
  };
  mlir::ascend::LogicalStageCost prefix;
  prefix.id = "simd_prefix";
  prefix.description = prefix.id;
  prefix.implementations = {
      makeCost(StageMode::SIMD, 1, 5.0), makeCost(StageMode::SIMT, 1, 50.0),
      makeCost(StageMode::SIMT, 2, 25.0), makeCost(StageMode::SIMT, 4, 12.5)};
  prefix.localSimtMaterializable = true;
  prefix.localSimtFactors = {1, 2, 4};

  mlir::ascend::LogicalStageCost payload;
  payload.id = "local_simt_payload";
  payload.description = payload.id;
  payload.implementations = {
      makeCost(StageMode::SIMD, 1, 100.0), makeCost(StageMode::SIMT, 1, 10.0),
      makeCost(StageMode::SIMT, 2, 1.0), makeCost(StageMode::SIMT, 4, 0.5)};
  payload.localSimtMaterializable = true;
  payload.localSimtFactors = {1, 2, 4};
  table.stages = {prefix, payload};
  mlir::ascend::LogicalPhaseCost phase;
  phase.id = "phase";
  phase.stages = table.stages;
  table.phases.push_back(std::move(phase));

  auto routes = solveStageRoutes(table, StageTransitionCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 5.5);
}

TEST(SimdSimtCostModelTest, StageWorkloadAnalysisConservesKernelWork) {
  const SimdSimtFeatureSummary features = gatherDotFeatures();
  auto result =
      StagePartitioner().partition(features, StagePartitionerOptions{});
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  ASSERT_TRUE(*result);
  const StagePartition &partition = **result;
  double operations = 0.0, bytes = 0.0, warps = 0.0, flops = 0.0, issue = 0.0;
  for (const auto &phase : partition.phases)
    for (const auto &stage : phase.stages) {
      for (const auto &entry : stage.workload.operationElements)
        operations += entry.second * stage.iterationCount;
      bytes += stage.workload.loadBytes * stage.iterationCount;
      warps += stage.workload.loadWarpInstructions * stage.iterationCount;
      flops += stage.workload.dotFlops * stage.iterationCount;
      issue += stage.workload.issueElements * stage.iterationCount;
    }
  EXPECT_DOUBLE_EQ(operations, 82.0);
  EXPECT_DOUBLE_EQ(bytes, 1088.0);
  EXPECT_DOUBLE_EQ(warps, 17.0);
  EXPECT_DOUBLE_EQ(flops, 8192.0);
  EXPECT_GT(issue, 0.0);
}

TEST(SimdSimtCostModelTest,
     OperationGraphBoundaryOwnsEveryRootAndDerivesLiveValues) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%pointer: i64) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %loaded = "tt.load"(%pointer) : (i64) -> tensor<16x16xf32>
        %result = scf.for %i = %c2 to %c16 step %c1
            iter_args(%state = %loaded) -> tensor<16x16xf32> {
          %next = arith.addf %state, %loaded : tensor<16x16xf32>
          scf.yield %next : tensor<16x16xf32>
        }
        "tt.store"(%pointer, %result) : (i64, tensor<16x16xf32>) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::Operation *recurrence = nullptr;
  module->walk([&](mlir::scf::ForOp loop) { recurrence = loop; });
  ASSERT_NE(recurrence, nullptr);

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = recurrence;
  anchor.scopeOperations.push_back(recurrence);
  anchor.scopeInsertionPoint = recurrence;
  anchor.kind = mlir::ascend::SimtAnchorKind::TriangularSolveLoop;
  anchor.facts =
      triangularBt16StageFeatures().simtAnchors.triangularSolves.front();
  anchor.lowerability.mixed = mlir::ascend::CandidateLoweringStatus::Native;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  SimdSimtFeatureSummary features = triangularBt16StageFeatures();
  auto phasePlan = mlir::ascend::PhaseBoundaryAnalysis().analyze(
      *module, anchorPlan, features, StagePartitionerOptions{});
  if (!phasePlan)
    FAIL() << llvm::toString(phasePlan.takeError());
  ASSERT_TRUE(*phasePlan);
  EXPECT_EQ((*phasePlan)->rootOperations.size(),
            (*phasePlan)->rootPhaseIds.size());
  EXPECT_EQ((*phasePlan)->rootPhaseIds,
            std::vector<std::string>({"head", "head", "head", "diagonal_load",
                                      "diagonal_inverse", "merge_store"}));

  auto result = StagePartitioner().partition(*module, anchorPlan, features,
                                             StagePartitionerOptions{});
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  ASSERT_TRUE(*result);
  const StagePartition &partition = **result;
  EXPECT_EQ(partition.boundarySource, "operation_graph");
  EXPECT_TRUE(partition.operationOwnershipComplete);

  int64_t ownedRootCount = 0;
  const LogicalStage *recurrenceStage = nullptr;
  for (const LogicalPhase &phase : partition.phases) {
    for (const LogicalStage &stage : phase.stages) {
      ownedRootCount += static_cast<int64_t>(stage.operations.size());
      if (stage.id == "diagonal_inverse_recurrence")
        recurrenceStage = &stage;
    }
  }
  EXPECT_EQ(ownedRootCount, partition.modeledOperationCount);
  ASSERT_NE(recurrenceStage, nullptr);
  EXPECT_EQ(recurrenceStage->operations.size(), 1u);
  EXPECT_FALSE(recurrenceStage->liveIns.empty());
  EXPECT_EQ(recurrenceStage->liveOuts.size(), 1u);
  EXPECT_EQ(recurrenceStage->simtAnchorIndices, std::vector<unsigned>({0}));
  EXPECT_TRUE(recurrenceStage->localSimtMaterializable);
  // The arith.addf is the scf.for body and therefore represents 256 element
  // additions on every one of the 14 dynamic recurrence iterations.  The
  // per-iteration Stage workload must remain 256, not be divided by 14.
  auto add = recurrenceStage->workload.operationElements.find("f32.add");
  ASSERT_NE(add, recurrenceStage->workload.operationElements.end());
  EXPECT_DOUBLE_EQ(add->second, 256.0);
}

TEST(SimdSimtCostModelTest,
     GenericStageDiscoveryFindsPaddedCopyBoundaryCandidates) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @padded_copy(%indices: i64, %input: i64, %output: i64) {
        %index = "tt.load"(%indices) : (i64) -> i32
        %c0 = arith.constant 0 : i32
        %condition = arith.cmpi sgt, %index, %c0 : i32
        %offset = scf.if %condition -> (i64) {
          %extended = arith.extsi %index : i32 to i64
          scf.yield %extended : i64
        } else {
          %zero = arith.constant 0 : i64
          scf.yield %zero : i64
        }
        %offsets = "tt.make_range"() : () -> tensor<8xi64>
        %input_ptrs = "tt.addptr"(%input, %offsets) :
            (i64, tensor<8xi64>) -> tensor<8xi64>
        %values = "tt.load"(%input_ptrs) :
            (tensor<8xi64>) -> tensor<8xf32>
        %scale = arith.constant dense<2.0> : tensor<8xf32>
        %scaled = arith.mulf %values, %scale : tensor<8xf32>
        %output_ptrs = "tt.addptr"(%output, %offsets) :
            (i64, tensor<8xi64>) -> tensor<8xi64>
        "tt.store"(%output_ptrs, %scaled) :
            (tensor<8xi64>, tensor<8xf32>) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::ascend::SimtAnchorPlan anchorPlan;
  StageDiscoveryOptions options;
  options.maximumSuperblockFactor = 4;
  auto graph = StageDiscovery().discover(*module, anchorPlan, options);
  if (!graph)
    FAIL() << llvm::toString(graph.takeError());
  ASSERT_GT(graph->dependenceGraph.units.size(), 2u);
  EXPECT_EQ(graph->boundaryCount(), graph->dependenceGraph.units.size() + 1);
  EXPECT_FALSE(graph->dependenceGraph.edges.empty());

  size_t copyBegin = graph->dependenceGraph.units.size();
  for (const auto &unit : graph->dependenceGraph.units)
    if (unit.operation->getName().getStringRef() == "tt.make_range")
      copyBegin = unit.index;
  ASSERT_LT(copyBegin, graph->dependenceGraph.units.size());

  const mlir::ascend::CandidateStage *prefix = nullptr;
  const mlir::ascend::CandidateStage *copy = nullptr;
  const mlir::ascend::CandidateStage *whole = nullptr;
  for (const auto &candidate : graph->candidates) {
    if (candidate.beginBoundary == 0 &&
        candidate.endBoundary == graph->dependenceGraph.units.size())
      whole = &candidate;
    if (candidate.beginBoundary == 0 && candidate.endBoundary == copyBegin)
      prefix = &candidate;
    if (candidate.beginBoundary == copyBegin &&
        candidate.endBoundary == graph->dependenceGraph.units.size())
      copy = &candidate;
  }
  ASSERT_NE(prefix, nullptr);
  ASSERT_NE(copy, nullptr);
  ASSERT_NE(whole, nullptr);
  EXPECT_EQ(prefix->stage.costModelKind, StageCostModelKind::ScalarControl);
  EXPECT_EQ(copy->stage.costModelKind,
            StageCostModelKind::ContinuousTileMemory);
  EXPECT_TRUE(prefix->stage.localSimtMaterializable);
  EXPECT_TRUE(copy->stage.localSimtMaterializable);
  EXPECT_EQ(whole->stage.legalSimtFactors, std::vector<int64_t>({1, 2, 4}));
}

TEST(SimdSimtCostModelTest,
     StageMaterializationMergesAdjacentSimtStagesAndThreadsResults) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel() -> i32 {
        %c1 = arith.constant 1 : i32
        %c2 = arith.constant 2 : i32
        %sum = arith.addi %c1, %c2 : i32
        %product = arith.muli %sum, %c2 : i32
        return %product : i32
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  ASSERT_TRUE(function);
  llvm::SmallVector<mlir::Operation *> roots;
  for (mlir::Operation &operation : function.getBody().front())
    if (!operation.hasTrait<mlir::OpTrait::IsTerminator>())
      roots.push_back(&operation);
  ASSERT_EQ(roots.size(), 4u);

  mlir::ascend::StageCostModelSummary summary;
  summary.mixed.legal = true;
  summary.mixed.implementations = {
      {StageMode::SIMD, 1}, {StageMode::SIMT, 1}, {StageMode::SIMT, 1}};
  summary.mixed.stageIndices = {0, 1, 2};
  mlir::ascend::LogicalStageCost simd;
  simd.beginBoundary = 0;
  simd.endBoundary = 2;
  simd.operations = {roots[0], roots[1]};
  mlir::ascend::LogicalStageCost firstSimt;
  firstSimt.beginBoundary = 2;
  firstSimt.endBoundary = 3;
  firstSimt.operations = {roots[2]};
  firstSimt.localSimtMaterializable = true;
  mlir::ascend::LogicalStageCost secondSimt;
  secondSimt.beginBoundary = 3;
  secondSimt.endBoundary = 4;
  secondSimt.operations = {roots[3]};
  secondSimt.localSimtMaterializable = true;
  summary.stages = {simd, firstSimt, secondSimt};

  auto plan = mlir::ascend::buildStageMaterializationPlan(summary);
  if (!plan)
    FAIL() << llvm::toString(plan.takeError());
  ASSERT_EQ(plan->ranges.size(), 1u);
  EXPECT_EQ(plan->ranges.front().operations.size(), 2u);
  EXPECT_TRUE(
      mlir::succeeded(mlir::ascend::materializeSimtStagePlan(*module, *plan)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  int64_t scopes = 0;
  module->walk([&](mlir::Operation *operation) {
    scopes += operation->getName().getStringRef() == "scope.scope";
  });
  EXPECT_EQ(scopes, 1);
}

TEST(SimdSimtCostModelTest,
     CompoundScopeOrderIsNormalizedBeforePhasePartitioning) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%pointer: i64) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %setup = arith.constant dense<0> : tensor<16xi32>
        %loaded = "tt.load"(%pointer) : (i64) -> tensor<16x16xf32>
        %result = scf.for %i = %c2 to %c16 step %c1
            iter_args(%state = %loaded) -> tensor<16x16xf32> {
          %next = arith.addf %state, %loaded : tensor<16x16xf32>
          scf.yield %next : tensor<16x16xf32>
        }
        "tt.store"(%pointer, %result) : (i64, tensor<16x16xf32>) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::Operation *setup = nullptr;
  mlir::Operation *recurrence = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "arith.constant" &&
        operation->getNumResults() == 1 &&
        mlir::isa<mlir::RankedTensorType>(operation->getResult(0).getType()))
      setup = operation;
    if (mlir::isa<mlir::scf::ForOp>(operation))
      recurrence = operation;
  });
  ASSERT_NE(setup, nullptr);
  ASSERT_NE(recurrence, nullptr);

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = recurrence;
  anchor.scopeOperations = {setup, recurrence};
  anchor.scopeInsertionPoint = recurrence;
  anchor.kind = mlir::ascend::SimtAnchorKind::TriangularSolveLoop;
  anchor.facts =
      triangularBt16StageFeatures().simtAnchors.triangularSolves.front();
  anchor.lowerability.mixed = mlir::ascend::CandidateLoweringStatus::Native;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  auto phasePlan = mlir::ascend::PhaseBoundaryAnalysis().analyze(
      *module, anchorPlan, triangularBt16StageFeatures(),
      StagePartitionerOptions{});
  if (!phasePlan)
    FAIL() << llvm::toString(phasePlan.takeError());
  ASSERT_TRUE(*phasePlan);
  EXPECT_EQ((*phasePlan)->rootPhaseIds,
            std::vector<std::string>({"head", "head", "head", "diagonal_load",
                                      "diagonal_inverse", "diagonal_inverse",
                                      "merge_store"}));

  auto partition = StagePartitioner().partition(*module, anchorPlan,
                                                triangularBt16StageFeatures(),
                                                StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());
  ASSERT_TRUE(*partition);
  const LogicalStage *loadStage = nullptr;
  const LogicalStage *recurrenceStage = nullptr;
  for (const LogicalPhase &phase : (**partition).phases)
    for (const LogicalStage &stage : phase.stages) {
      if (stage.id == "load_diagonal_tiles")
        loadStage = &stage;
      if (stage.id == "diagonal_inverse_recurrence")
        recurrenceStage = &stage;
    }
  ASSERT_NE(loadStage, nullptr);
  ASSERT_NE(recurrenceStage, nullptr);
  EXPECT_EQ(loadStage->operations.size(), 1u);
  EXPECT_EQ(recurrenceStage->operations.size(), 2u);
  EXPECT_EQ(recurrenceStage->simtAnchorIndices, std::vector<unsigned>({0}));
}

TEST(SimdSimtCostModelTest, PointerInductionLoopIsNotADataRecurrence) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%start: i64) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %step = arith.constant 16 : i64
        %address = scf.for %i = %c0 to %c8 step %c1
            iter_args(%current = %start) -> i64 {
          %value = "tt.load"(%current) : (i64) -> f32
          %next = arith.addi %current, %step : i64
          scf.yield %next : i64
        }
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::Operation *loop = nullptr;
  module->walk([&](mlir::scf::ForOp operation) { loop = operation; });
  ASSERT_NE(loop, nullptr);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalPhase phase;
  phase.id = "convert_store";
  LogicalStage stage =
      logicalStage("pointer_loop", StageCostModelKind::ConversionPack,
                   StageScheduleKind::IndependentPipelined, 8);
  stage.operations.push_back(loop);
  phase.stages.push_back(std::move(stage));
  partition.phases.push_back(std::move(phase));

  if (llvm::Error error = StageFeatureAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  if (llvm::Error error =
          mlir::ascend::StageKindClassifier().analyze(partition, 8192))
    FAIL() << llvm::toString(std::move(error));
  const LogicalStage &classified = partition.phases.front().stages.front();
  EXPECT_TRUE(classified.features.hasLoop);
  EXPECT_TRUE(classified.features.hasPointerInduction);
  EXPECT_FALSE(classified.features.hasLoopCarriedDataDependency);
  EXPECT_EQ(classified.costModelKind,
            StageCostModelKind::IndependentPipelinedLoop);
}

TEST(SimdSimtCostModelTest, ScalarSelectedRowCopyLoopRemainsContiguous) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%row_ptr: i64, %input: i64) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %step = arith.constant dense<8> : tensor<8xi32>
        %row = "tt.load"(%row_ptr) : (i64) -> i32
        %row_i64 = arith.extsi %row : i32 to i64
        %row_base = "tt.addptr"(%input, %row_i64) : (i64, i64) -> i64
        %base = "tt.splat"(%row_base) : (i64) -> tensor<8xi64>
        %initial = "tt.make_range"() : () -> tensor<8xi32>
        %offsets = scf.for %i = %c0 to %c4 step %c1
            iter_args(%current = %initial) -> tensor<8xi32> {
          %ptrs = "tt.addptr"(%base, %current) :
              (tensor<8xi64>, tensor<8xi32>) -> tensor<8xi64>
          %values = "tt.load"(%ptrs) : (tensor<8xi64>) -> tensor<8xf32>
          %next = arith.addi %current, %step : tensor<8xi32>
          scf.yield %next : tensor<8xi32>
        }
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::ascend::SimtAnchorPlan anchorPlan;
  StageDiscoveryOptions options;
  options.maximumSuperblockFactor = 4;
  auto graph = StageDiscovery().discover(*module, anchorPlan, options);
  if (!graph)
    FAIL() << llvm::toString(graph.takeError());

  size_t copyBegin = graph->dependenceGraph.units.size();
  for (const auto &unit : graph->dependenceGraph.units)
    if (unit.operation->getName().getStringRef() == "tt.make_range")
      copyBegin = unit.index;
  ASSERT_LT(copyBegin, graph->dependenceGraph.units.size());

  const mlir::ascend::CandidateStage *copy = nullptr;
  for (const auto &candidate : graph->candidates)
    if (candidate.beginBoundary == copyBegin &&
        candidate.endBoundary == graph->dependenceGraph.units.size())
      copy = &candidate;
  ASSERT_NE(copy, nullptr);
  EXPECT_EQ(copy->stage.costModelKind,
            StageCostModelKind::ContinuousTileMemory);
  EXPECT_TRUE(copy->stage.features.hasPointerInduction);
  EXPECT_FALSE(copy->stage.features.hasLoopCarriedDataDependency);
  EXPECT_FALSE(copy->stage.features.hasIndirectMemory);
  EXPECT_TRUE(copy->stage.features.hasContiguousMemory);
}

TEST(SimdSimtCostModelTest, IncompatibleDominantStructuresRequireStageSplit) {
  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalPhase phase;
  phase.id = "compound";
  LogicalStage stage =
      logicalStage("gather_dot", StageCostModelKind::TinyCubeRoofline,
                   StageScheduleKind::PartiallyDependent, 1);
  stage.features.hasDot = true;
  stage.features.hasIndirectMemory = true;
  phase.stages.push_back(std::move(stage));
  partition.phases.push_back(std::move(phase));

  llvm::Error error =
      mlir::ascend::StageKindClassifier().analyze(partition, 16384);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("requires_split"),
            std::string::npos);
}

TEST(SimdSimtCostModelTest, FeatureSummaryPartitionIsExplicitFallback) {
  auto result = StagePartitioner().partition(triangularBt16StageFeatures(),
                                             StagePartitionerOptions{});
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  ASSERT_TRUE(*result);
  EXPECT_EQ((**result).boundarySource, "feature_summary_fallback");
  EXPECT_FALSE((**result).operationOwnershipComplete);
  EXPECT_EQ((**result).modeledOperationCount, 0);
}
