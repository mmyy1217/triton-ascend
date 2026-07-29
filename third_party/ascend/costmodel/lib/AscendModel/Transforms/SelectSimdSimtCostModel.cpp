//===- SelectSimdSimtCostModel.cpp - C++ SIMD/SIMT selection ------------===//
//
// This pass is the online owner of SIMD/SIMT candidate selection.  Python
// only schedules the pass and reacts to its machine-readable execution intent.
// Feature extraction, calibrated scoring, confidence/target/margin gates, and
// mixed-operation planning stay in C++.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/SimdSimtCostModel.h"
#include "AscendModel/Transforms/Passes.h"
#include "Utils/SimtSelection.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
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
inline constexpr llvm::StringLiteral kRankingConfidenceAttr =
    "ascend.simt_costmodel.ranking_confidence";
inline constexpr llvm::StringLiteral kAllSimdScoreAttr =
    "ascend.simt_costmodel.all_simd_score";
inline constexpr llvm::StringLiteral kAllSimtScoreAttr =
    "ascend.simt_costmodel.all_simt_score";
inline constexpr llvm::StringLiteral kMixedScoreAttr =
    "ascend.simt_costmodel.mixed_score";
inline constexpr llvm::StringLiteral kReportJSONAttr =
    "ascend.simt_costmodel.report_json";

static bool isPlainOneDimensionalCumsum(Operation *op) {
  if (op->getName().getStringRef() != "tt.scan" || op->getNumOperands() == 0)
    return false;

  auto axis = op->getAttrOfType<IntegerAttr>("axis");
  auto sourceType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!axis || !sourceType || !sourceType.hasStaticShape())
    return false;

  int64_t axisValue = axis.getInt();
  if (axisValue < 0 || axisValue >= sourceType.getRank())
    return false;
  for (auto [index, extent] : llvm::enumerate(sourceType.getShape())) {
    if (static_cast<int64_t>(index) != axisValue && extent != 1)
      return false;
  }

  int64_t realCombineOps = 0;
  bool isAdd = false;
  op->walk([&](Operation *nested) {
    if (nested == op)
      return;
    llvm::StringRef name = nested->getName().getStringRef();
    if (name == "tt.scan.return" || name == "arith.extf" ||
        name == "arith.truncf" || name == "arith.bitcast")
      return;
    ++realCombineOps;
    isAdd = name == "arith.addf" || name == "arith.addi";
  });
  return realCombineOps == 1 && isAdd;
}

static bool hasTensorPointerOperand(Operation *op) {
  if (op->getNumOperands() == 0)
    return false;
  auto type = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  return type && type.hasStaticShape() && type.getRank() <= 5;
}

static bool pointerDependsOnLoadedIndex(Operation *memoryOp) {
  SmallVector<Value> worklist{memoryOp->getOperand(0)};
  llvm::DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;

    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Operation *parent = argument.getOwner()->getParentOp();
      unsigned number = argument.getArgNumber();
      if (parent && parent->getName().getStringRef() == "scf.for" &&
          number > 0 && number + 2 < parent->getNumOperands()) {
        worklist.push_back(parent->getOperand(number + 2));
        Operation *yield = argument.getOwner()->getTerminator();
        if (yield && number - 1 < yield->getNumOperands())
          worklist.push_back(yield->getOperand(number - 1));
      }
      continue;
    }

    Operation *producer = value.getDefiningOp();
    if (!producer)
      continue;

    llvm::StringRef name = producer->getName().getStringRef();
    if (name == "tt.load" || name == "tt.gather")
      return true;
    llvm::append_range(worklist, producer->getOperands());
  }
  return false;
}

static bool isMixedSimtAnchor(Operation *op, bool compileOn91095) {
  if (!compileOn91095)
    return false;
  llvm::StringRef name = op->getName().getStringRef();
  if (name == "tt.gather" || name == "tt.histogram")
    return true;
  if (name == "tt.scan")
    return isPlainOneDimensionalCumsum(op);
  if (name.starts_with("tt.atomic"))
    return hasTensorPointerOperand(op);
  if (name == "tt.load" || name == "tt.store")
    return hasTensorPointerOperand(op) && pointerDependsOnLoadedIndex(op);
  return false;
}

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

static SmallVector<Operation *>
collectMixedSimtAnchors(ModuleOp module, bool compileOn91095) {
  SmallVector<Operation *> anchors;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isMixedSimtAnchor(op, compileOn91095)) {
      anchors.push_back(op);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  return anchors;
}

static void clearPreviousSelection(ModuleOp module) {
  module->removeAttr(kEffectiveExecutionAttr);
  module->removeAttr(kRecommendedExecutionAttr);
  module->removeAttr(kSelectionSourceAttr);
  module->removeAttr(kRankingConfidenceAttr);
  module->removeAttr(kAllSimdScoreAttr);
  module->removeAttr(kAllSimtScoreAttr);
  module->removeAttr(kMixedScoreAttr);
  module->removeAttr(kReportJSONAttr);
  module.walk(
      [](Operation *op) { op->removeAttr(kSelectedForSimtAttr); });
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

struct SelectSimdSimtCostModelPass
    : public impl::SelectSimdSimtCostModelPassBase<
          SelectSimdSimtCostModelPass> {
  using SelectSimdSimtCostModelPassBase::
      SelectSimdSimtCostModelPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    clearPreviousSelection(module);
    bool autoMode = mode.getValue() == "auto";

    SimdSimtCostModelOptions options;
    options.profilePath = profilePath.getValue();
    options.actualTarget = actualTarget.getValue();
    options.numWarps =
        static_cast<unsigned>(std::max<int64_t>(1, numWarps.getValue()));
    options.marginRatio =
        std::max(0.0, std::min(1.0, marginRatio.getValue()));
    options.includeFeaturesInJSON = true;
    options.scoreOutsideCalibrationCoverage = !autoMode;

    auto reportOr = analyzeSimdSimtCandidates(module, options);
    if (!reportOr) {
      module.emitError("C++ SIMD/SIMT cost model failed: ")
          << llvm::toString(reportOr.takeError());
      signalPassFailure();
      return;
    }
    SimdSimtCostReport report = std::move(*reportOr);

    std::string recommended =
        report.candidateCostsEvaluated
            ? stringifySimdSimtCandidate(report.decision).str()
            : kBackendDefault.str();
    std::string effective = kBackendDefault.str();
    std::string selectionSource = "backend_default";
    std::string applicationReason;
    SmallVector<Operation *> mixedAnchors;

    bool actionSupported = true;
    bool hasExplicitScope = containsExplicitVectorScope(module);
    if (recommended == kMixedSimdSimt) {
      if (hasExplicitScope) {
        actionSupported = false;
        applicationReason = "explicit_scope_present";
      } else {
        mixedAnchors =
            collectMixedSimtAnchors(module, compileOn91095.getValue());
        if (mixedAnchors.empty()) {
          actionSupported = false;
          applicationReason = "no_materializable_mixed_anchor";
        }
      }
    } else if (recommended == kAllSimtOnly && hasExplicitScope) {
      // Preserve explicit local SIMD/SIMT/cube scope semantics instead of
      // replacing the whole kernel with a pure-SIMT route.
      actionSupported = false;
      applicationReason = "explicit_scope_present";
    }

    if (autoMode && report.gatePassed && actionSupported) {
      effective = recommended;
      selectionSource = "cpp_cost_model";
      applicationReason = "cpp_cost_model_admitted";
      if (effective == kMixedSimdSimt) {
        for (Operation *anchor : mixedAnchors)
          anchor->setAttr(kSelectedForSimtAttr,
                          UnitAttr::get(module.getContext()));
      }
    } else if (!autoMode) {
      applicationReason = "report_mode";
    } else if (!report.gatePassed) {
      applicationReason =
          report.gateReasons.empty() ? "model_gate_rejected"
                                     : report.gateReasons.front();
    }

    Builder builder(module.getContext());
    module->setAttr(kRecommendedExecutionAttr,
                    builder.getStringAttr(recommended));
    module->setAttr(kEffectiveExecutionAttr,
                    builder.getStringAttr(effective));
    module->setAttr(kSelectionSourceAttr,
                    builder.getStringAttr(selectionSource));
    module->setAttr(kRankingConfidenceAttr,
                    builder.getStringAttr(report.rankingConfidence));
    if (report.candidateCostsEvaluated) {
      module->setAttr(kAllSimdScoreAttr,
                      builder.getF64FloatAttr(report.candidateCosts.allSimd));
      module->setAttr(
          kAllSimtScoreAttr,
          builder.getF64FloatAttr(report.candidateCosts.allSimtOnly));
      module->setAttr(
          kMixedScoreAttr,
          builder.getF64FloatAttr(report.candidateCosts.mixedSimdSimt));
    }

    llvm::json::Object reportJSON = report.toJSON();
    reportJSON["mode"] = mode.getValue();
    reportJSON["recommended_decision_kind"] = recommended;
    reportJSON["effective_decision_kind"] = effective;
    reportJSON["selection_source"] = selectionSource;
    reportJSON["application_reason"] = applicationReason;
    reportJSON["action_supported"] = actionSupported;
    reportJSON["selected_simt_anchor_count"] =
        static_cast<int64_t>(mixedAnchors.size());
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
