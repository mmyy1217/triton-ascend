//===- SelectSimdSimtCostModel.cpp - C++ SIMD/SIMT selection ------------===//
//
// This pass is the online owner of SIMD/SIMT candidate selection.  Python
// only schedules the pass and reacts to its machine-readable execution intent.
// Feature extraction, stage scoring, candidate legality, and mixed-operation
// planning stay in C++.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/StageModel/SimdSimtCostModel.h"
#include "AscendModel/StageModel/SimtAnchorAnalysis.h"
#include "AscendModel/StageModel/SimtSelection.h"
#include "AscendModel/StageModel/StageDiscovery.h"
#include "AscendModel/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_SELECTSIMDSIMTCOSTMODELPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

using namespace simt_selection;

inline constexpr llvm::StringLiteral kRecommendedExecutionAttr =
    "ascend.simt_costmodel.recommended";
inline constexpr llvm::StringLiteral kSelectionSourceAttr =
    "ascend.simt_costmodel.selection_source";
inline constexpr llvm::StringLiteral kAllSimdScoreAttr =
    "ascend.simt_costmodel.all_simd_score";
inline constexpr llvm::StringLiteral kAllSimtScoreAttr =
    "ascend.simt_costmodel.all_simt_score";
inline constexpr llvm::StringLiteral kMixedScoreAttr =
    "ascend.simt_costmodel.mixed_score";
inline constexpr llvm::StringLiteral kReportJSONAttr =
    "ascend.simt_costmodel.report_json";
inline constexpr llvm::StringLiteral kSuperblockFactorAttr =
    "ascend.simt_costmodel.superblock_factor";

static bool containsExplicitVectorScope(ModuleOp module) {
  bool found = false;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "scope.scope") {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static void clearPreviousSelection(ModuleOp module) {
  module->removeAttr(kEffectiveExecutionAttr);
  module->removeAttr(kRecommendedExecutionAttr);
  module->removeAttr(kSelectionSourceAttr);
  module->removeAttr(kAllSimdScoreAttr);
  module->removeAttr(kAllSimtScoreAttr);
  module->removeAttr(kMixedScoreAttr);
  module->removeAttr(kReportJSONAttr);
  module->removeAttr(kSuperblockFactorAttr);
}

static SimtAnchorPlan
buildSelectedMixedAnchorPlan(const StageCostModelSummary &stageModel,
                             const SimtAnchorPlan &completePlan) {
  SimtAnchorPlan selected;
  selected.kernelLowerability = completePlan.kernelLowerability;
  if (!stageModel.mixed.legal || stageModel.mixed.implementations.size() !=
                                     stageModel.mixed.stageIndices.size())
    return selected;

  llvm::DenseSet<unsigned> included;
  for (size_t position = 0; position < stageModel.mixed.implementations.size();
       ++position) {
    size_t stageIndex = stageModel.mixed.stageIndices[position];
    if (stageIndex >= stageModel.stages.size())
      continue;
    const LogicalStageCost &stage = stageModel.stages[stageIndex];
    const StageImplementation &implementation =
        stageModel.mixed.implementations[position];
    if (implementation.mode != StageMode::SIMT)
      continue;

    for (unsigned index : stage.simtAnchorIndices) {
      if (index >= completePlan.anchors.size() ||
          !included.insert(index).second)
        continue;
      selected.anchors.push_back(completePlan.anchors[index]);
    }
  }
  return selected;
}

static LogicalResult appendJSONLine(llvm::StringRef path,
                                    llvm::StringRef json) {
  if (path.empty())
    return success();
  std::error_code error;
  llvm::raw_fd_ostream os(path, error, llvm::sys::fs::OF_Append);
  if (error)
    return failure();
  os << json << '\n';
  return success();
}

static std::string hashTTIR(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  llvm::ArrayRef<uint8_t> bytes(reinterpret_cast<const uint8_t *>(text.data()),
                                text.size());
  auto digest = llvm::SHA256::hash(bytes);
  return llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
}

struct SelectSimdSimtCostModelPass
    : public impl::SelectSimdSimtCostModelPassBase<
          SelectSimdSimtCostModelPass> {
  using SelectSimdSimtCostModelPassBase::SelectSimdSimtCostModelPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    clearPreviousSelection(module);
    const std::string ttirSha256 = hashTTIR(module);
    const bool autoMode = mode.getValue() == "auto";

    SimdSimtCostModelOptions options;
    options.profilePath = profilePath.getValue();
    options.actualTarget = actualTarget.getValue();
    options.numWarps =
        static_cast<unsigned>(std::max<int64_t>(1, numWarps.getValue()));
    options.includeFeaturesInJSON = true;
    options.compileOn91095 = compileOn91095.getValue();
    options.wholeKernelSuperblockMaterializable =
        wholeKernelSuperblockMaterializable.getValue();
    options.scopeSuperblockMaterializable =
        scopeSuperblockMaterializable.getValue();

    SimtAnchorPlan anchorPlan =
        buildMixedSimtAnchorPlan(module, options.compileOn91095);
    auto reportOr = analyzeSimdSimtCandidates(module, anchorPlan, options);
    if (!reportOr) {
      module.emitError("C++ SIMD/SIMT cost model failed: ")
          << llvm::toString(reportOr.takeError());
      signalPassFailure();
      return;
    }
    SimdSimtCostReport report = std::move(*reportOr);

    std::string recommended = stringifySimdSimtCandidate(report.decision).str();
    std::string effective = kBackendDefault.str();
    std::string selectionSource = "backend_default";
    std::string applicationReason;
    SmallVector<Operation *> mixedAnchors;
    SimtAnchorPlan selectedMixedAnchorPlan;
    std::optional<StageMaterializationPlan> selectedStagePlan;
    int64_t selectedSuperblockFactor = 1;
    if (report.stageModel.applied) {
      if (report.decision == SimdSimtCandidateKind::AllSIMD)
        selectedSuperblockFactor =
            report.stageModel.allSimd.routeSuperblockFactor;
      else if (report.decision == SimdSimtCandidateKind::AllSIMTOnly)
        selectedSuperblockFactor =
            report.stageModel.allSimt.routeSuperblockFactor;
      else
        selectedSuperblockFactor =
            report.stageModel.mixed.routeSuperblockFactor;
    }

    bool actionSupported = true;
    bool hasExplicitScope = containsExplicitVectorScope(module);
    if (recommended == kMixedSimdSimt) {
      if (hasExplicitScope) {
        actionSupported = false;
        applicationReason = "explicit_scope_present";
      } else if (report.stageModel.boundarySource == "stage_boundary_graph") {
        auto plan = buildStageMaterializationPlan(report.stageModel);
        if (!plan) {
          llvm::consumeError(plan.takeError());
          actionSupported = false;
          applicationReason = "no_materializable_mixed_stage_plan";
        } else {
          selectedStagePlan = std::move(*plan);
        }
      } else {
        selectedMixedAnchorPlan =
            buildSelectedMixedAnchorPlan(report.stageModel, anchorPlan);
        mixedAnchors = selectedMixedAnchorPlan.materializableRoots();
        if (mixedAnchors.empty()) {
          actionSupported = false;
          applicationReason = "no_materializable_mixed_anchor";
        }
      }
      // A factor>1 mixed route needs batching of the surrounding SIMD
      // producer/consumer phases, not just a scope attribute.  Keep the
      // recommendation visible but do not apply it until ScopeSuperBlockPass
      // implements that exact materialization.
      if (selectedSuperblockFactor > 1 &&
          !options.scopeSuperblockMaterializable) {
        actionSupported = false;
        applicationReason = "scope_superblock_not_materializable";
      }
    } else if (recommended == kAllSimtOnly && hasExplicitScope) {
      // Preserve explicit local SIMD/SIMT/cube scope semantics instead of
      // replacing the whole kernel with a pure-SIMT route.
      actionSupported = false;
      applicationReason = "explicit_scope_present";
    }
    if (selectedSuperblockFactor > 1 &&
        selectedSuperblockFactor * options.numWarps > 64) {
      actionSupported = false;
      applicationReason = "superblock_warp_limit_exceeded";
    }
    if (recommended == kAllSimtOnly && selectedSuperblockFactor > 1 &&
        !report.features.autoBlockifyV1Applied &&
        !options.wholeKernelSuperblockMaterializable) {
      actionSupported = false;
      applicationReason = "superblock_requires_auto_blockify_v1";
    }

    if (autoMode && actionSupported) {
      effective = recommended;
      selectionSource = "cpp_cost_model";
      applicationReason = "minimum_cost_candidate";
    } else if (!autoMode) {
      applicationReason = "report_mode";
    } else if (applicationReason.empty()) {
      applicationReason = "candidate_not_materializable";
    }

    Builder builder(module.getContext());
    module->setAttr(kRecommendedExecutionAttr,
                    builder.getStringAttr(recommended));
    module->setAttr(kEffectiveExecutionAttr, builder.getStringAttr(effective));
    module->setAttr(kSelectionSourceAttr,
                    builder.getStringAttr(selectionSource));
    module->setAttr(kAllSimdScoreAttr,
                    builder.getF64FloatAttr(report.candidateCosts.allSimd));
    module->setAttr(kAllSimtScoreAttr,
                    builder.getF64FloatAttr(report.candidateCosts.allSimtOnly));
    module->setAttr(kMixedScoreAttr, builder.getF64FloatAttr(
                                         report.candidateCosts.mixedSimdSimt));
    module->setAttr(kSuperblockFactorAttr,
                    builder.getI64IntegerAttr(selectedSuperblockFactor));

    // Selector and Materializer consume the same immutable anchor plan in one
    // pass invocation.  No per-operation marker is persisted in TTIR.
    if (effective == kMixedSimdSimt) {
      LogicalResult materialized =
          selectedStagePlan
              ? materializeSimtStagePlan(module, *selectedStagePlan)
              : materializeSimtAnchorPlan(module, selectedMixedAnchorPlan);
      if (failed(materialized)) {
        signalPassFailure();
        return;
      }
    }

    llvm::json::Object reportJSON = report.toJSON();
    reportJSON["stage_model_snapshot_version"] = "1.0";
    reportJSON["ttir_sha256"] = ttirSha256;
    llvm::json::Object snapshotConfig;
    snapshotConfig["actual_target"] = options.actualTarget;
    snapshotConfig["num_warps"] = static_cast<int64_t>(options.numWarps);
    snapshotConfig["compile_on_91095"] = options.compileOn91095;
    snapshotConfig["whole_kernel_superblock_materializable"] =
        options.wholeKernelSuperblockMaterializable;
    snapshotConfig["scope_superblock_materializable"] =
        options.scopeSuperblockMaterializable;
    reportJSON["stage_model_config"] = std::move(snapshotConfig);
    reportJSON["mode"] = mode.getValue();
    reportJSON["recommended_decision_kind"] = recommended;
    reportJSON["effective_decision_kind"] = effective;
    reportJSON["selection_source"] = selectionSource;
    reportJSON["application_reason"] = applicationReason;
    reportJSON["action_supported"] = actionSupported;
    reportJSON["materialized_simt_anchor_count"] =
        static_cast<int64_t>(mixedAnchors.size());
    reportJSON["materialized_simt_stage_scope_count"] =
        selectedStagePlan
            ? static_cast<int64_t>(selectedStagePlan->ranges.size())
            : 0;
    if (selectedStagePlan)
      reportJSON["stage_materialization_plan"] = selectedStagePlan->toJSON();
    reportJSON["selected_superblock_factor"] = selectedSuperblockFactor;
    std::string json =
        llvm::formatv("{0}", llvm::json::Value(std::move(reportJSON))).str();
    module->setAttr(kReportJSONAttr, builder.getStringAttr(json));

    if (failed(appendJSONLine(reportFile.getValue(), json)))
      module.emitWarning("failed to append C++ SIMD/SIMT report to ")
          << reportFile.getValue();
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
