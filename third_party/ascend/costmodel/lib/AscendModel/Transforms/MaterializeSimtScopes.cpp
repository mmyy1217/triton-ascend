//===- MaterializeSimtScopes.cpp - Materialize local SIMT scopes --------===//
//
// This pass is intentionally a materializer, not a selector.  The C++ cost
// model records an admitted execution kind on the module/function and marks
// the exact operations assigned to SIMT.  This pass turns those markers into
// SSA-safe scope.scope regions consumed by the Ascend lowering pipeline.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Transforms/Passes.h"
#include "Utils/SimtSelection.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_MATERIALIZESIMTSCOPESPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

using namespace simt_selection;

static bool hasSelectedAncestor(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isSelectedForSimt(parent))
      return true;
  }
  return false;
}

static bool isMaterializable(Operation *op) {
  return op->getBlock() && !isa<ModuleOp>(op) &&
         !op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         !op->hasTrait<OpTrait::IsTerminator>() &&
         op->getName().getStringRef() != "scope.scope" &&
         op->getName().getStringRef() != "scope.return";
}

/// Wrap one operation and thread all of its SSA results through scope.return.
///
/// Scope regions are not isolated from above, so operands remain legal
/// captures.  Moving only the selected operation keeps SIMD producers and
/// consumers outside the SIMT region.
static LogicalResult wrapSelectedOperation(Operation *op) {
  OpBuilder builder(op);
  OperationState scopeState(op->getLoc(), "scope.scope");
  scopeState.addTypes(op->getResultTypes());
  scopeState.addAttribute(kVectorModeAttr, builder.getStringAttr("simt"));
  scopeState.addRegion();
  Operation *scopeOp = builder.create(scopeState);

  Region &scopeRegion = scopeOp->getRegion(0);
  auto *scopeBody = new Block();
  scopeRegion.push_back(scopeBody);

  SmallVector<Value> originalResults(op->getResults());
  op->moveBefore(scopeBody, scopeBody->end());

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(scopeBody);
  OperationState returnState(op->getLoc(), "scope.return");
  returnState.addOperands(originalResults);
  Operation *returnOp = bodyBuilder.create(returnState);

  if (scopeOp->getNumResults() != originalResults.size())
    return op->emitError("SIMT scope result count does not match selected op");

  for (auto [original, replacement] :
       llvm::zip_equal(originalResults, scopeOp->getResults())) {
    original.replaceAllUsesExcept(replacement, returnOp);
  }
  return success();
}

struct MaterializeSimtScopesPass
    : public impl::MaterializeSimtScopesPassBase<
          MaterializeSimtScopesPass> {
  using MaterializeSimtScopesPassBase::MaterializeSimtScopesPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> selectedOps;
    bool sawMixedDecision = false;
    bool sawInvalidSelection = false;
    int64_t alreadyMaterialized = 0;

    module.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isMixedModelDecision(op))
        sawMixedDecision = true;
      if (!isMixedModelDecision(op) || !isSelectedForSimt(op))
        return WalkResult::advance();

      // A selected parent scope already covers nested operations.  Likewise,
      // preserve a user/model-provided local SIMT scope without nesting a
      // second wrapper.
      if (hasSelectedAncestor(op) || hasEnclosingVectorMode(op, "simt")) {
        ++alreadyMaterialized;
        return WalkResult::skip();
      }
      if (!isMaterializable(op)) {
        op->emitError(
            "ascend.simt_costmodel.selected is not materializable as a local "
            "SIMT scope");
        sawInvalidSelection = true;
        return WalkResult::skip();
      }
      selectedOps.push_back(op);
      return WalkResult::skip();
    });

    if (sawInvalidSelection) {
      signalPassFailure();
      return;
    }

    int64_t materialized = alreadyMaterialized;
    for (Operation *op : selectedOps) {
      if (failed(wrapSelectedOperation(op))) {
        signalPassFailure();
        return;
      }
      ++materialized;
    }

    module->setAttr(
        kScopeMaterializedAttr,
        IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                         materialized));

    // An admitted mixed decision with no local SIMT operation is never a valid
    // materialization.  Fail here instead of silently compiling a different
    // execution plan than the model selected.
    if (sawMixedDecision && materialized == 0) {
      module.emitError(
          "mixed_simd_simt was selected but no operation carries "
          "ascend.simt_costmodel.selected");
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
