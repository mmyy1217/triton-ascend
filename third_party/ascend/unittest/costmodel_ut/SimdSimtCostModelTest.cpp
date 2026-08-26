#include "AscendModel/StageModel/SimdSimtCostModel.h"
#include "AscendModel/StageModel/StageAnalysis.h"
#include "AscendModel/StageModel/StageCostModels.h"
#include "AscendModel/StageModel/StageDiscovery.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

using mlir::ascend::HardwareProfile;
using mlir::ascend::LogicalStage;
using mlir::ascend::solveStageRoutes;
using mlir::ascend::StageCostEvaluator;
using mlir::ascend::StageCostModelKind;
using mlir::ascend::StageCostModelRegistry;
using mlir::ascend::StageCostTable;
using mlir::ascend::StageDiscovery;
using mlir::ascend::StageDiscoveryOptions;
using mlir::ascend::StageFeatureAnalysis;
using mlir::ascend::StageMode;
using mlir::ascend::StageScheduleKind;
using mlir::ascend::StageTransitionCost;

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
  mlir::ascend::StageBoundaryGraph graph;
  graph.dependenceGraph.stages.resize(1);
  graph.dependenceGraph.cuttableBoundaries = {true, true};
  mlir::ascend::DiscoveredStage discovered;
  discovered.beginBoundary = 0;
  discovered.endBoundary = 1;
  discovered.stage = std::move(stage);
  graph.stages.push_back(std::move(discovered));
  return StageCostEvaluator().evaluate(graph, profile);
}

void makeLinearBoundaryGraph(StageCostTable &table) {
  table.boundaryCount = static_cast<int64_t>(table.stages.size() + 1);
  for (auto [index, stage] : llvm::enumerate(table.stages)) {
    stage.beginBoundary = static_cast<int64_t>(index);
    stage.endBoundary = static_cast<int64_t>(index + 1);
  }
}

} // namespace

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

TEST(SimdSimtCostModelTest, KernelMixedRouteComesFromAdjacentStageModes) {
  StageCostTable table;
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
  makeLinearBoundaryGraph(table);

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

TEST(SimdSimtCostModelTest, BoundaryGraphRouteUsesEveryStageExactlyOnce) {
  StageCostTable table;
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
    stage.localSimtMaterializable = true;
    stage.localSimtFactors = {1};
    stage.implementations = {makeCost(StageMode::SIMD, simd),
                             makeCost(StageMode::SIMT, simt)};
    return stage;
  };
  table.stages = {makeStage("scalar_prefix", 0, 1, 50.0, 5.0),
                  makeStage("copy_loop", 1, 2, 5.0, 50.0)};

  auto result = solveStageRoutes(table, StageTransitionCost{});
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  ASSERT_TRUE(result->mixed.legal);
  EXPECT_EQ(result->mixed.stageIndices, std::vector<size_t>({0, 1}));
  ASSERT_EQ(result->mixed.implementations.size(), 2u);
  EXPECT_EQ(result->mixed.implementations[0].mode, StageMode::SIMT);
  EXPECT_EQ(result->mixed.implementations[1].mode, StageMode::SIMD);
  EXPECT_DOUBLE_EQ(result->mixed.totalCycles, 10.0);
}

TEST(SimdSimtCostModelTest, MixedScopePaysExactBidirectionalUbHandoffCost) {
  StageCostTable table;
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
  makeLinearBoundaryGraph(table);

  StageTransitionCost transition;
  transition.simdUbLoadBytesPerCycle = 512.0;
  transition.simdUbStoreBytesPerCycle = 256.0;
  transition.simtUbLoadBytesPerCycle = 128.0;
  transition.simtUbStoreBytesPerCycle = 128.0;
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
  makeLinearBoundaryGraph(table);
  auto routes = solveStageRoutes(table, StageTransitionCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_TRUE(routes->allSimt.legal);
  EXPECT_FALSE(routes->mixed.legal);
}

TEST(SimdSimtCostModelTest, MixedRouteChargesOneMergedStageScope) {
  StageCostTable table;
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
  makeLinearBoundaryGraph(table);

  StageTransitionCost transition;
  transition.simdToSimtCycles = 10.0;
  transition.simtToSimdCycles = 10.0;
  auto routes = solveStageRoutes(table, transition);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  // Adjacent compatible anchor payloads materialize as one Candidate Scope.
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 23.0);
}

TEST(SimdSimtCostModelTest, ConservativeScopeSetupProxyPrefersOneScopeRun) {
  StageCostTable table;
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
  auto makeStage = [&](llvm::StringRef id, double simd, double simt) {
    mlir::ascend::LogicalStageCost stage;
    stage.id = id.str();
    stage.description = stage.id;
    stage.localSimtMaterializable = true;
    stage.localSimtFactors = {1};
    stage.implementations = {makeCost(StageMode::SIMD, simd),
                             makeCost(StageMode::SIMT, simt)};
    return stage;
  };
  table.stages = {makeStage("simt_head", 10.0, 1.0),
                  makeStage("simd_middle", 1.0, 10.0),
                  makeStage("simt_tail", 10.0, 1.0)};
  makeLinearBoundaryGraph(table);

  StageTransitionCost transition;
  transition.scopeSetupProxyCycles = 20.0;
  transition.scopeSetupProxyConfidence = "low";
  auto routes = solveStageRoutes(table, transition);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_EQ(routes->mixed.scopeRuns.size(), 2u);
  ASSERT_EQ(routes->conservativeMixed.scopeRuns.size(), 1u);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 3.0);
  EXPECT_DOUBLE_EQ(routes->conservativeMixed.totalCycles, 32.0);
}

TEST(SimdSimtCostModelTest, ScopeRunDerivesAggregateLiveValues) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%input: tensor<8xf32>) -> tensor<8xf32> {
        %head = arith.constant 0 : i32
        %seed = arith.constant dense<1.0> : tensor<8xf32>
        %sum = arith.addf %input, %seed : tensor<8xf32>
        return %sum : tensor<8xf32>
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
  ASSERT_EQ(roots.size(), 3u);

  StageCostTable table;
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
  for (auto [index, operation] : llvm::enumerate(roots)) {
    mlir::ascend::LogicalStageCost stage;
    stage.id = ("stage_" + std::to_string(index));
    stage.description = stage.id;
    stage.operations = {operation};
    stage.localSimtMaterializable = index > 0;
    stage.localSimtFactors = stage.localSimtMaterializable
                                 ? std::vector<int64_t>{1}
                                 : std::vector<int64_t>{};
    stage.implementations = {
        makeCost(StageMode::SIMD, index == 0 ? 1.0 : 1000.0),
        makeCost(StageMode::SIMT, index == 0 ? 1000.0 : 1.0)};
    table.stages.push_back(std::move(stage));
  }
  makeLinearBoundaryGraph(table);

  auto routes = solveStageRoutes(table, StageTransitionCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  ASSERT_EQ(routes->mixed.scopeRuns.size(), 1u);
  const auto &run = routes->mixed.scopeRuns.front();
  EXPECT_EQ(run.beginBoundary, 1);
  EXPECT_EQ(run.endBoundary, 3);
  EXPECT_EQ(run.liveInCount, 1);
  EXPECT_EQ(run.liveOutCount, 1);
  EXPECT_EQ(run.liveInTensorBytes, 32);
  EXPECT_EQ(run.liveOutTensorBytes, 32);
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
  makeLinearBoundaryGraph(table);

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
  makeLinearBoundaryGraph(table);

  auto routes = solveStageRoutes(table, StageTransitionCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 5.5);
}

TEST(SimdSimtCostModelTest,
     GenericStageDiscoveryBuildsOneStagePerSemanticUnit) {
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
  ASSERT_GT(graph->dependenceGraph.stages.size(), 2u);
  EXPECT_EQ(graph->boundaryCount(), graph->dependenceGraph.stages.size() + 1);
  ASSERT_EQ(graph->stages.size(), graph->dependenceGraph.stages.size());
  EXPECT_FALSE(graph->dependenceGraph.edges.empty());

  const mlir::ascend::DiscoveredStage *control = nullptr;
  const mlir::ascend::DiscoveredStage *vectorLoad = nullptr;
  for (auto [index, stage] : llvm::enumerate(graph->stages)) {
    EXPECT_EQ(stage.beginBoundary, index);
    EXPECT_EQ(stage.endBoundary, index + 1);
    ASSERT_EQ(stage.stage.operations.size(), 1u);
    EXPECT_EQ(stage.stage.operations.front(),
              graph->dependenceGraph.stages[index].operation);
    if (stage.stage.simtLegal)
      EXPECT_EQ(stage.stage.legalSimtFactors, std::vector<int64_t>({1, 2, 4}));
    mlir::Operation *operation = stage.stage.operations.front();
    if (operation->getName().getStringRef() == "scf.if")
      control = &stage;
    if (operation->getName().getStringRef() == "tt.load" &&
        llvm::isa<mlir::ShapedType>(operation->getResult(0).getType()))
      vectorLoad = &stage;
  }
  ASSERT_NE(control, nullptr);
  ASSERT_NE(vectorLoad, nullptr);
  EXPECT_EQ(control->stage.costModelKind, StageCostModelKind::ScalarControl);
  EXPECT_EQ(vectorLoad->stage.costModelKind,
            StageCostModelKind::ContinuousTileMemory);
  EXPECT_TRUE(control->stage.localSimtMaterializable);
  EXPECT_TRUE(vectorLoad->stage.localSimtMaterializable);
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

  LogicalStage stage =
      logicalStage("pointer_loop", StageCostModelKind::ConversionPack,
                   StageScheduleKind::IndependentPipelined, 8);
  stage.operations.push_back(loop);

  if (llvm::Error error = StageFeatureAnalysis().analyze(stage))
    FAIL() << llvm::toString(std::move(error));
  if (llvm::Error error =
          mlir::ascend::StageKindClassifier().analyze(stage, 8192))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_TRUE(stage.features.hasLoop);
  EXPECT_TRUE(stage.features.hasPointerInduction);
  EXPECT_FALSE(stage.features.hasLoopCarriedDataDependency);
  EXPECT_EQ(stage.costModelKind, StageCostModelKind::IndependentPipelinedLoop);
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

  const mlir::ascend::DiscoveredStage *loop = nullptr;
  for (const auto &stage : graph->stages)
    if (stage.stage.operations.front()->getName().getStringRef() == "scf.for")
      loop = &stage;
  ASSERT_NE(loop, nullptr);
  EXPECT_EQ(loop->stage.costModelKind,
            StageCostModelKind::ContinuousTileMemory);
  EXPECT_TRUE(loop->stage.features.hasPointerInduction);
  EXPECT_FALSE(loop->stage.features.hasLoopCarriedDataDependency);
  EXPECT_FALSE(loop->stage.features.hasIndirectMemory);
  EXPECT_TRUE(loop->stage.features.hasContiguousMemory);
}

TEST(SimdSimtCostModelTest, IncompatibleDominantStructuresRequireStageSplit) {
  LogicalStage stage =
      logicalStage("gather_dot", StageCostModelKind::TinyCubeRoofline,
                   StageScheduleKind::PartiallyDependent, 1);
  stage.features.hasDot = true;
  stage.features.hasIndirectMemory = true;

  llvm::Error error = mlir::ascend::StageKindClassifier().analyze(stage, 16384);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("requires_split"),
            std::string::npos);
}
