#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/Transforms/Passes.h"
#include "Utils/SimtSelection.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "llvm/Support/JSON.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>

using mlir::ModuleOp;
using mlir::Operation;
using mlir::OwningOpRef;
using mlir::Pass;
using mlir::PassManager;
using mlir::StringAttr;
using mlir::ascend::createAssignOpIDsPass;
using mlir::ascend::createEstimateCyclesPass;
using mlir::ascend::createMaterializeSimtScopesPass;
using mlir::ascend::createPerfReportPass;
using mlir::ascend::createPipelineAnalysisPass;
using mlir::ascend::createSelectSimdSimtCostModelPass;
using mlir::ascend::EstimateCyclesPassOptions;
using mlir::ascend::SelectSimdSimtCostModelPassOptions;

namespace {

constexpr const char *kVectorModule = R"mlir(
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = ascend.vector_load %arg0 {bytes = 16 : i64} : tensor<4xf32> -> tensor<4xf32>
    %1 = ascend.add %0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    ascend.vector_store %1 {bytes = 16 : i64} : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
)mlir";

constexpr const char *kOutOfSimdSimtCoverageModule = R"mlir(
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir";

void registerDialects(mlir::MLIRContext &context) {
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::ascend::AscendModelDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::scope::ScopeDialect>();
}

OwningOpRef<ModuleOp> parseModule(mlir::MLIRContext &context,
                                  llvm::StringRef source) {
  registerDialects(context);
  return mlir::parseSourceString<ModuleOp>(source, &context);
}

template <typename... PassTs>
bool runPasses(ModuleOp module, PassTs &&...passes) {
  PassManager pm(module.getContext());
  (pm.addPass(std::forward<PassTs>(passes)), ...);
  return mlir::succeeded(pm.run(module));
}

Operation *findFirstOp(ModuleOp module, llvm::StringRef name) {
  Operation *result = nullptr;
  module.walk([&](Operation *op) {
    if (!result && op->getName().getStringRef() == name)
      result = op;
  });
  return result;
}

int64_t getI64Attr(Operation *op, llvm::StringRef name) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
  return attr ? attr.getInt() : -1;
}

} // namespace

TEST(CostModelPassesTest, AssignOpIDsPassAnnotatesAscendOpsOnly) {
  mlir::MLIRContext context;
  auto module = parseModule(context, R"mlir(
module {
  func.func @main(%arg0: i32, %arg1: i32, %arg2: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = arith.addi %arg0, %arg1 : i32
    %0 = ascend.add %arg2, %arg2 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  ASSERT_TRUE(runPasses(*module, createAssignOpIDsPass()));

  auto totalOps = module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
      "ascend.total_ops");
  ASSERT_TRUE(totalOps);
  EXPECT_EQ(totalOps.getInt(), 1);

  Operation *addOp = findFirstOp(*module, "ascend.add");
  ASSERT_NE(addOp, nullptr);
  EXPECT_EQ(getI64Attr(addOp, "op_id"), 0);

  Operation *arithOp = findFirstOp(*module, "arith.addi");
  ASSERT_NE(arithOp, nullptr);
  EXPECT_FALSE(arithOp->hasAttr("op_id"));
}

TEST(CostModelPassesTest, EstimateCyclesAnnotatesComputeAndTransferOps) {
  mlir::MLIRContext context;
  auto module = parseModule(context, kVectorModule);
  ASSERT_TRUE(module);

  ASSERT_TRUE(runPasses(*module, createEstimateCyclesPass()));

  Operation *loadOp = findFirstOp(*module, "ascend.vector_load");
  Operation *addOp = findFirstOp(*module, "ascend.add");
  Operation *storeOp = findFirstOp(*module, "ascend.vector_store");
  ASSERT_NE(loadOp, nullptr);
  ASSERT_NE(addOp, nullptr);
  ASSERT_NE(storeOp, nullptr);

  EXPECT_GT(getI64Attr(loadOp, "estimated_cycles"), 0);
  EXPECT_GT(getI64Attr(addOp, "estimated_cycles"), 0);
  EXPECT_GT(getI64Attr(storeOp, "estimated_cycles"), 0);
  EXPECT_EQ(getI64Attr(loadOp, "bytes"), 16);
  EXPECT_EQ(getI64Attr(storeOp, "bytes"), 16);
  EXPECT_EQ(getI64Attr(addOp, "flops"), 4);
  EXPECT_TRUE(loadOp->getAttrOfType<StringAttr>("hw_unit"));
  EXPECT_TRUE(addOp->getAttrOfType<StringAttr>("hw_unit"));
  EXPECT_TRUE(storeOp->getAttrOfType<StringAttr>("hw_unit"));
}

TEST(CostModelPassesTest,
     DavidF32AddConsumesSharedThroughputInAbsoluteModel) {
  mlir::MLIRContext context;
  auto module = parseModule(context, R"mlir(
module {
  func.func @main(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = ascend.add %arg0, %arg0
      : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  EstimateCyclesPassOptions options;
  options.hardwareConfigPath = TRITON_ASCEND_DAVID_TEST_CONFIG_PATH;
  ASSERT_TRUE(runPasses(*module, createEstimateCyclesPass(options)));

  Operation *addOp = findFirstOp(*module, "ascend.add");
  ASSERT_NE(addOp, nullptr);
  // 1024 f32 values / 64 lanes = 16 vector instructions.
  // Shared rate: 3.30 inst/SYS_CNT-cycle * 988.9/1650
  //             = 1.9778 inst/device-cycle.
  // ceil(16 / 1.9778) + 35 startup = 44 device cycles.
  EXPECT_EQ(getI64Attr(addOp, "estimated_cycles"), 44);
}

TEST(CostModelPassesTest, EstimateCyclesReportsInvalidArgBindings) {
  mlir::MLIRContext context;
  auto module = parseModule(context, kVectorModule);
  ASSERT_TRUE(module);

  EstimateCyclesPassOptions options;
  options.argBindingsStr = "arg0";

  EXPECT_FALSE(runPasses(*module, createEstimateCyclesPass(options)));
}

TEST(CostModelPassesTest, PipelineAnalysisSetsCycleSummaryAttrs) {
  mlir::MLIRContext context;
  auto module = parseModule(context, kVectorModule);
  ASSERT_TRUE(module);

  ASSERT_TRUE(runPasses(*module, createAssignOpIDsPass(),
                        createEstimateCyclesPass(),
                        createPipelineAnalysisPass()));

  auto scheduled =
      module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
          "ascend.scheduled_cycles_one_iter");
  auto roofline =
      module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
          "ascend.roofline_cycles");
  auto simple =
      module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
          "ascend.simple_sum_cycles");
  ASSERT_TRUE(scheduled);
  ASSERT_TRUE(roofline);
  ASSERT_TRUE(simple);
  EXPECT_GT(scheduled.getInt(), 0);
  EXPECT_GT(roofline.getInt(), 0);
  EXPECT_GT(simple.getInt(), 0);
}

TEST(CostModelPassesTest, PerfReportPassAcceptsEstimatedPipeline) {
  mlir::MLIRContext context;
  auto module = parseModule(context, kVectorModule);
  ASSERT_TRUE(module);

  EXPECT_TRUE(runPasses(*module, createAssignOpIDsPass(),
                        createEstimateCyclesPass(),
                        createPipelineAnalysisPass(),
                        createPerfReportPass()));
}

TEST(CostModelPassesTest,
     SimdSimtCoverageShortCircuitIsAutoOnly) {
  auto configureOptions =
      [](SelectSimdSimtCostModelPassOptions &options,
         llvm::StringRef mode) {
    options.mode = mode.str();
    options.profilePath =
        TRITON_ASCEND_SIMD_SIMT_TEST_PROFILE_PATH;
    options.actualTarget = "Ascend950PR_9579";
    options.numWarps = 4;
    options.marginRatio = 0.10;
    options.compileOn91095 = true;
  };

  mlir::MLIRContext autoContext;
  auto autoModule =
      parseModule(autoContext, kOutOfSimdSimtCoverageModule);
  ASSERT_TRUE(autoModule);
  SelectSimdSimtCostModelPassOptions autoOptions;
  configureOptions(autoOptions, "auto");
  ASSERT_TRUE(runPasses(
      *autoModule,
      createSelectSimdSimtCostModelPass(autoOptions)));

  auto autoEffective = (*autoModule)
                           ->getAttrOfType<StringAttr>(
                               "ascend.simt_costmodel.effective");
  auto autoRecommended = (*autoModule)
                             ->getAttrOfType<StringAttr>(
                                 "ascend.simt_costmodel.recommended");
  auto autoReport = (*autoModule)
                        ->getAttrOfType<StringAttr>(
                            "ascend.simt_costmodel.report_json");
  ASSERT_TRUE(autoEffective);
  ASSERT_TRUE(autoRecommended);
  ASSERT_TRUE(autoReport);
  EXPECT_EQ(autoEffective.getValue(), "backend_default");
  EXPECT_EQ(autoRecommended.getValue(), "backend_default");
  EXPECT_FALSE(
      (*autoModule)->hasAttr("ascend.simt_costmodel.all_simd_score"));
  auto autoJSON = llvm::json::parse(autoReport.getValue());
  ASSERT_TRUE(autoJSON);
  auto *autoObject = autoJSON->getAsObject();
  ASSERT_NE(autoObject, nullptr);
  auto autoEvaluated =
      autoObject->getBoolean("candidate_costs_evaluated");
  ASSERT_TRUE(autoEvaluated);
  EXPECT_FALSE(*autoEvaluated);
  auto autoReason = autoObject->getString("application_reason");
  ASSERT_TRUE(autoReason);
  EXPECT_EQ(*autoReason, "selection_score_invalid");

  mlir::MLIRContext reportContext;
  auto reportModule =
      parseModule(reportContext, kOutOfSimdSimtCoverageModule);
  ASSERT_TRUE(reportModule);
  SelectSimdSimtCostModelPassOptions reportOptions;
  configureOptions(reportOptions, "report");
  ASSERT_TRUE(runPasses(
      *reportModule,
      createSelectSimdSimtCostModelPass(reportOptions)));

  auto reportEffective = (*reportModule)
                             ->getAttrOfType<StringAttr>(
                                 "ascend.simt_costmodel.effective");
  auto reportJSONAttr = (*reportModule)
                            ->getAttrOfType<StringAttr>(
                                "ascend.simt_costmodel.report_json");
  ASSERT_TRUE(reportEffective);
  ASSERT_TRUE(reportJSONAttr);
  EXPECT_EQ(reportEffective.getValue(), "backend_default");
  EXPECT_TRUE(
      (*reportModule)->hasAttr("ascend.simt_costmodel.all_simd_score"));
  auto reportJSON = llvm::json::parse(reportJSONAttr.getValue());
  ASSERT_TRUE(reportJSON);
  auto *reportObject = reportJSON->getAsObject();
  ASSERT_NE(reportObject, nullptr);
  auto reportEvaluated =
      reportObject->getBoolean("candidate_costs_evaluated");
  ASSERT_TRUE(reportEvaluated);
  EXPECT_TRUE(*reportEvaluated);
  auto reportReason = reportObject->getString("application_reason");
  ASSERT_TRUE(reportReason);
  EXPECT_EQ(*reportReason, "report_mode");
}

TEST(CostModelPassesTest, MaterializeSimtScopePreservesEscapingSSAResult) {
  mlir::MLIRContext context;
  auto module = parseModule(context, R"mlir(
module attributes {
  ascend.simt_costmodel.effective = "mixed_simd_simt"
} {
  func.func @main(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 {ascend.simt_costmodel.selected} : i32
    %1 = arith.muli %0, %arg1 : i32
    return %1 : i32
  }
}
)mlir");
  ASSERT_TRUE(module);

  ASSERT_TRUE(runPasses(*module, createMaterializeSimtScopesPass()));

  Operation *scopeOp = findFirstOp(*module, "scope.scope");
  ASSERT_NE(scopeOp, nullptr);
  ASSERT_EQ(scopeOp->getNumRegions(), 1u);
  ASSERT_EQ(scopeOp->getNumResults(), 1u);
  ASSERT_TRUE(scopeOp->getAttrOfType<StringAttr>("vec_mode"));
  EXPECT_EQ(scopeOp->getAttrOfType<StringAttr>("vec_mode").getValue(), "simt");

  auto &scopeBody = scopeOp->getRegion(0).front();
  Operation *selectedAdd = nullptr;
  Operation *scopeReturn = nullptr;
  for (Operation &nested : scopeBody) {
    if (nested.getName().getStringRef() == "arith.addi")
      selectedAdd = &nested;
    if (nested.getName().getStringRef() == "scope.return")
      scopeReturn = &nested;
  }
  ASSERT_NE(selectedAdd, nullptr);
  ASSERT_NE(scopeReturn, nullptr);
  EXPECT_TRUE(selectedAdd->hasAttr(
      mlir::ascend::simt_selection::kSelectedForSimtAttr));
  ASSERT_EQ(scopeReturn->getNumOperands(), 1u);
  EXPECT_EQ(scopeReturn->getOperand(0), selectedAdd->getResult(0));

  Operation *mulOp = findFirstOp(*module, "arith.muli");
  ASSERT_NE(mulOp, nullptr);
  ASSERT_EQ(mulOp->getNumOperands(), 2u);
  EXPECT_EQ(mulOp->getOperand(0), scopeOp->getResult(0));
  EXPECT_NE(mulOp->getParentOp(), scopeOp);

  auto materialized = module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
      "ascend.simt_costmodel.scope_materialized");
  ASSERT_TRUE(materialized);
  EXPECT_EQ(materialized.getInt(), 1);
}

TEST(CostModelPassesTest, NativeWholeBodySimtScopeDetectionAndInlining) {
  mlir::MLIRContext context;
  auto module = parseModule(context, R"mlir(
module {
  func.func public @main(%arg0: i32) {
    %c1 = arith.constant 1 : i32
    "scope.scope"() ({
      "scope.scope"() ({
        %0 = arith.addi %arg0, %c1 : i32
        "scope.return"() : () -> ()
      }) {vec_mode = "simt"} : () -> ()
      "scope.return"() : () -> ()
    }) {vec_mode = "simt"} : () -> ()
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  Operation *scope =
      mlir::ascend::simt_selection::findWholeBodyVoidSimtScope(
          module->getOperation());
  ASSERT_NE(scope, nullptr);
  EXPECT_EQ(mlir::ascend::simt_selection::
                inlineVoidSimtScopesForPureSimt(module->getOperation()),
            2);
  EXPECT_EQ(findFirstOp(*module, "scope.scope"), nullptr);
  EXPECT_NE(findFirstOp(*module, "arith.addi"), nullptr);
  EXPECT_EQ(
      mlir::ascend::simt_selection::findWholeBodyVoidSimtScope(
          module->getOperation()),
      nullptr);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));

  auto resultBearing = parseModule(context, R"mlir(
module {
  func.func public @main(%arg0: i32) -> i32 {
    %0 = "scope.scope"() ({
      "scope.return"(%arg0) : (i32) -> ()
    }) {vec_mode = "simt"} : () -> i32
    return %0 : i32
  }
}
)mlir");
  ASSERT_TRUE(resultBearing);
  EXPECT_EQ(mlir::ascend::simt_selection::findWholeBodyVoidSimtScope(
                resultBearing->getOperation()),
            nullptr);
  EXPECT_EQ(mlir::ascend::simt_selection::
                inlineVoidSimtScopesForPureSimt(
                    resultBearing->getOperation()),
            0);
  EXPECT_NE(findFirstOp(*resultBearing, "scope.scope"), nullptr);
}

TEST(CostModelPassesTest, ModelControlledRoutingIgnoresLegacyGlobalForce) {
  mlir::MLIRContext context;
  auto module = parseModule(context, R"mlir(
module attributes {
  ascend.simt_costmodel.effective = "all_simd"
} {
  func.func @main(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
}
)mlir");
  ASSERT_TRUE(module);

  Operation *addOp = findFirstOp(*module, "arith.addi");
  ASSERT_NE(addOp, nullptr);
  EXPECT_TRUE(mlir::ascend::simt_selection::isModelControlled(addOp));
  EXPECT_FALSE(mlir::ascend::simt_selection::shouldUseSimtTemplate(
      addOp, /*legacyForceSimt=*/true));

  addOp->setAttr(mlir::ascend::simt_selection::kSelectedForSimtAttr,
                 mlir::UnitAttr::get(&context));
  EXPECT_TRUE(mlir::ascend::simt_selection::shouldUseSimtTemplate(
      addOp, /*legacyForceSimt=*/false));

  addOp->removeAttr(mlir::ascend::simt_selection::kSelectedForSimtAttr);
  (*module)->setAttr(mlir::ascend::simt_selection::kEffectiveExecutionAttr,
                     mlir::StringAttr::get(&context, "backend_default"));
  EXPECT_FALSE(mlir::ascend::simt_selection::isModelControlled(addOp));
  EXPECT_TRUE(mlir::ascend::simt_selection::shouldUseSimtTemplate(
      addOp, /*legacyForceSimt=*/true));
}
