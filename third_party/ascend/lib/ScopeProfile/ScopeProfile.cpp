#include "ascend/include/ScopeProfile/ScopeProfile.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>

#define DEBUG_TYPE "ta-scope-profile"

namespace mlir::triton {
#define GEN_PASS_DEF_TASCOPEPROFILE
#include "ascend/include/ScopeProfile/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace mlir::triton {
namespace {

constexpr llvm::StringLiteral kAutoBlockifyAttr = "ta.auto_blockify_v1";
constexpr llvm::StringLiteral kAutoBlockifyLoopAttr =
    "ta.auto_blockify_v1.loop";
constexpr llvm::StringLiteral kAutoBlockifyScheduleAttr =
    "ta.auto_blockify_v1.schedule";
constexpr llvm::StringLiteral kAutoBlockifyFactorAttr =
    "ta.auto_blockify_v1.superblock_factor";
constexpr llvm::StringLiteral kManifestAttr =
    "ascend.scope_profile.manifest_json";
constexpr llvm::StringLiteral kMaterializedAttr =
    "ascend.scope_profile.materialized";

struct ProfileRoot {
  Operation *operation = nullptr;
  unsigned index = 0;
  unsigned segment = 0;
};

struct RootSegment {
  unsigned index = 0;
  unsigned begin = 0;
  unsigned end = 0;
};

struct ScopeCandidate {
  int64_t id = 0;
  unsigned segment = 0;
  unsigned begin = 0;
  unsigned end = 0;
};

static std::string sha256(llvm::StringRef content) {
  llvm::ArrayRef<uint8_t> bytes(
      reinterpret_cast<const uint8_t *>(content.data()), content.size());
  auto digest = llvm::SHA256::hash(bytes);
  return llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
}

static std::string printType(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return stream.str();
}

static std::string printOperation(Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->print(stream, OpPrintingFlags().enableDebugInfo(false));
  return stream.str();
}

static std::string fingerprint(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream, OpPrintingFlags().enableDebugInfo(false));
  return sha256(stream.str());
}

static bool isInside(Value value, const llvm::DenseSet<Operation *> &inside) {
  if (Operation *definition = value.getDefiningOp())
    return inside.contains(definition);
  auto argument = dyn_cast<BlockArgument>(value);
  Operation *parent = argument ? argument.getOwner()->getParentOp() : nullptr;
  return parent && inside.contains(parent);
}

static bool hasInsideAncestor(Operation *operation,
                              const llvm::DenseSet<Operation *> &inside) {
  for (Operation *owner = operation; owner; owner = owner->getParentOp())
    if (inside.contains(owner))
      return true;
  return false;
}

static llvm::json::Array valueTypes(ArrayRef<Value> values) {
  llvm::json::Array types;
  for (Value value : values)
    types.push_back(printType(value.getType()));
  return types;
}

static llvm::json::Object boundaryJSON(ArrayRef<ProfileRoot> roots,
                                       const ScopeCandidate &candidate) {
  llvm::DenseSet<Operation *> inside;
  for (unsigned index = candidate.begin; index < candidate.end; ++index) {
    Operation *root = roots[index].operation;
    inside.insert(root);
    root->walk([&](Operation *nested) { inside.insert(nested); });
  }

  llvm::SetVector<Value> liveIns;
  llvm::SetVector<Value> liveOuts;
  for (Operation *operation : inside)
    for (Value operand : operation->getOperands())
      if (!isInside(operand, inside))
        liveIns.insert(operand);
  for (unsigned index = candidate.begin; index < candidate.end; ++index)
    for (Value result : roots[index].operation->getResults())
      if (llvm::any_of(result.getUsers(), [&](Operation *user) {
            return !hasInsideAncestor(user, inside);
          }))
        liveOuts.insert(result);

  bool pointerLiveOut = llvm::any_of(liveOuts, [](Value value) {
    return llvm::StringRef(printType(value.getType())).contains("!tt.ptr");
  });
  llvm::json::Object boundary;
  boundary["live_in_count"] = static_cast<int64_t>(liveIns.size());
  boundary["live_out_count"] = static_cast<int64_t>(liveOuts.size());
  boundary["live_in_types"] = valueTypes(liveIns.getArrayRef());
  boundary["live_out_types"] = valueTypes(liveOuts.getArrayRef());
  boundary["pointer_like_live_out"] = pointerLiveOut;
  return boundary;
}

static llvm::json::Object planJSON(ArrayRef<ProfileRoot> roots,
                                   const ScopeCandidate &candidate) {
  llvm::json::Object plan;
  plan["id"] = candidate.id;
  if (candidate.id == 0) {
    plan["kind"] = "empty";
    plan["begin"] = 0;
    plan["end"] = 0;
    plan["segment"] = -1;
    plan["root_indices"] = llvm::json::Array();
    llvm::json::Object boundary;
    boundary["live_in_count"] = 0;
    boundary["live_out_count"] = 0;
    boundary["live_in_types"] = llvm::json::Array();
    boundary["live_out_types"] = llvm::json::Array();
    boundary["pointer_like_live_out"] = false;
    plan["boundary"] = std::move(boundary);
    return plan;
  }

  plan["kind"] = "single_scope";
  plan["begin"] = static_cast<int64_t>(candidate.begin);
  plan["end"] = static_cast<int64_t>(candidate.end);
  plan["segment"] = static_cast<int64_t>(candidate.segment);
  llvm::json::Array indices;
  for (unsigned index = candidate.begin; index < candidate.end; ++index)
    indices.push_back(static_cast<int64_t>(index));
  plan["root_indices"] = std::move(indices);
  plan["boundary"] = boundaryJSON(roots, candidate);
  return plan;
}

static LogicalResult materializeCandidate(ArrayRef<ProfileRoot> roots,
                                          const ScopeCandidate &candidate) {
  if (candidate.id == 0)
    return success();

  SmallVector<Operation *> operations;
  for (unsigned index = candidate.begin; index < candidate.end; ++index)
    operations.push_back(roots[index].operation);
  if (operations.empty())
    return failure();

  llvm::DenseSet<Operation *> planned;
  for (Operation *operation : operations)
    planned.insert(operation);
  auto isInsideRange = [&](Operation *user) {
    for (Operation *owner = user; owner; owner = owner->getParentOp())
      if (planned.contains(owner))
        return true;
    return false;
  };

  SmallVector<Value> escaping;
  llvm::DenseSet<Value> seen;
  for (Operation *operation : operations)
    for (Value result : operation->getResults())
      for (OpOperand &use : result.getUses())
        if (!isInsideRange(use.getOwner()) && seen.insert(result).second) {
          escaping.push_back(result);
          break;
        }

  Operation *insertionPoint = operations.front();
  OpBuilder builder(insertionPoint);
  OperationState scopeState(insertionPoint->getLoc(), "scope.scope");
  for (Value value : escaping)
    scopeState.addTypes(value.getType());
  scopeState.addAttribute("vector_mode", builder.getStringAttr("simt"));
  scopeState.addRegion();
  Operation *scopeOperation = builder.create(scopeState);

  Region &region = scopeOperation->getRegion(0);
  auto *body = new Block();
  region.push_back(body);
  for (Operation *operation : operations)
    operation->moveBefore(body, body->end());

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  OperationState returnState(insertionPoint->getLoc(), "scope.return");
  returnState.addOperands(escaping);
  Operation *returnOperation = bodyBuilder.create(returnState);

  if (scopeOperation->getNumResults() != escaping.size())
    return failure();
  for (auto [original, replacement] :
       llvm::zip_equal(escaping, scopeOperation->getResults()))
    for (OpOperand &use : llvm::make_early_inc_range(original.getUses()))
      if (use.getOwner() != returnOperation &&
          !isInsideRange(use.getOwner()))
        use.set(replacement);
  return success();
}

class TAScopeProfilePass
    : public impl::TAScopeProfileBase<TAScopeProfilePass> {
public:
  using Base = impl::TAScopeProfileBase<TAScopeProfilePass>;
  using Base::Base;

  explicit TAScopeProfilePass(const TAScopeProfileOptions &options)
      : Base(options) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module->removeAttr(kManifestAttr);
    module->removeAttr(kMaterializedAttr);

    if (planId < -1) {
      module.emitError("ScopeProfile expected plan-id >= -1, got ") << planId;
      return signalPassFailure();
    }

    SmallVector<FuncOp> functions;
    for (FuncOp function : module.getOps<FuncOp>())
      functions.push_back(function);
    if (functions.size() != 1) {
      module.emitError("ScopeProfile requires exactly one TTIR function, found ")
          << functions.size();
      return signalPassFailure();
    }
    FuncOp function = functions.front();
    if (!function.isPublic() || !function.getResultTypes().empty()) {
      function.emitError(
          "ScopeProfile requires one public void entry kernel; use the normal "
          "compiler path for helper or result-bearing functions");
      return signalPassFailure();
    }
    if (!function->hasAttr(kAutoBlockifyAttr)) {
      function.emitError(
          "ScopeProfile requires TA AutoBlockify V1 F1 before discovery");
      return signalPassFailure();
    }
    auto factor = function->getAttrOfType<IntegerAttr>(kAutoBlockifyFactorAttr);
    if (!factor || factor.getInt() != 1) {
      function.emitError("ScopeProfile requires AutoBlockify factor 1, got ")
          << (factor ? factor.getInt() : -1);
      return signalPassFailure();
    }

    bool hasExistingScope = false;
    module.walk([&](scope::ScopeOp) { hasExistingScope = true; });
    if (hasExistingScope) {
      module.emitError(
          "ScopeProfile expected scope-free input; remove existing scope.scope "
          "operations before profiling");
      return signalPassFailure();
    }

    SmallVector<scf::ForOp> scheduleLoops;
    function.walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kAutoBlockifyLoopAttr))
        scheduleLoops.push_back(loop);
    });
    if (scheduleLoops.size() != 1) {
      function.emitError(
          "ScopeProfile requires exactly one AutoBlockify scheduling loop, found ")
          << scheduleLoops.size();
      return signalPassFailure();
    }

    SmallVector<ProfileRoot> roots;
    SmallVector<RootSegment> segments;
    std::optional<unsigned> segmentBegin;
    auto closeSegment = [&]() {
      if (!segmentBegin)
        return;
      segments.push_back(
          {static_cast<unsigned>(segments.size()), *segmentBegin,
           static_cast<unsigned>(roots.size())});
      segmentBegin.reset();
    };

    for (Operation &operation : *scheduleLoops.front().getBody()) {
      if (operation.hasTrait<OpTrait::IsTerminator>() ||
          operation.hasAttr(kAutoBlockifyScheduleAttr)) {
        closeSegment();
        continue;
      }
      if (!segmentBegin)
        segmentBegin = roots.size();
      roots.push_back({&operation, static_cast<unsigned>(roots.size()),
                       static_cast<unsigned>(segments.size())});
    }
    closeSegment();
    if (roots.empty()) {
      function.emitError(
          "ScopeProfile found no original operations after AutoBlockify; verify "
          "that the kernel uses a program id and has an algorithm body");
      return signalPassFailure();
    }

    SmallVector<ScopeCandidate> candidates;
    candidates.push_back({0, 0, 0, 0});
    int64_t nextId = 1;
    for (const RootSegment &segment : segments)
      for (unsigned begin = segment.begin; begin < segment.end; ++begin)
        for (unsigned end = begin + 1; end <= segment.end; ++end)
          candidates.push_back({nextId++, segment.index, begin, end});

    const ScopeCandidate *selected = nullptr;
    if (planId >= 0) {
      auto iterator = llvm::find_if(candidates, [&](const ScopeCandidate &item) {
        return item.id == planId;
      });
      if (iterator == candidates.end()) {
        module.emitError("ScopeProfile plan-id ")
            << planId << " does not exist; expected 0.."
            << candidates.back().id;
        return signalPassFailure();
      }
      selected = &*iterator;
    }

    std::string irFingerprint = fingerprint(module);
    llvm::json::Object manifest;
    manifest["schema_version"] = 1;
    manifest["search_model"] = "single_contiguous_scope_v1";
    manifest["search_complete"] = true;
    manifest["kernel"] = function.getSymName().str();
    manifest["ir_fingerprint"] = irFingerprint;
    manifest["root_count"] = static_cast<int64_t>(roots.size());
    manifest["segment_count"] = static_cast<int64_t>(segments.size());
    manifest["candidate_count"] = static_cast<int64_t>(candidates.size());
    manifest["selected_plan_id"] = static_cast<int64_t>(planId);

    llvm::json::Array rootArray;
    for (const ProfileRoot &root : roots) {
      llvm::json::Object object;
      object["index"] = static_cast<int64_t>(root.index);
      object["segment"] = static_cast<int64_t>(root.segment);
      object["op_name"] = root.operation->getName().getStringRef().str();
      object["has_regions"] = root.operation->getNumRegions() != 0;
      object["operation_sha256"] = sha256(printOperation(root.operation));
      llvm::json::Array resultTypes;
      for (Type type : root.operation->getResultTypes())
        resultTypes.push_back(printType(type));
      object["result_types"] = std::move(resultTypes);
      rootArray.push_back(std::move(object));
    }
    manifest["roots"] = std::move(rootArray);

    llvm::json::Array segmentArray;
    for (const RootSegment &segment : segments) {
      llvm::json::Object object;
      object["index"] = static_cast<int64_t>(segment.index);
      object["begin"] = static_cast<int64_t>(segment.begin);
      object["end"] = static_cast<int64_t>(segment.end);
      segmentArray.push_back(std::move(object));
    }
    manifest["segments"] = std::move(segmentArray);

    llvm::json::Array planArray;
    for (const ScopeCandidate &candidate : candidates)
      planArray.push_back(planJSON(roots, candidate));
    manifest["plans"] = std::move(planArray);

    std::string manifestText =
        llvm::formatv("{0:2}", llvm::json::Value(std::move(manifest))).str();
    OpBuilder builder(module.getContext());
    module->setAttr(kManifestAttr, builder.getStringAttr(manifestText));

    if (selected && failed(materializeCandidate(roots, *selected))) {
      module.emitError("ScopeProfile failed to preserve SSA while materializing plan ")
          << selected->id;
      return signalPassFailure();
    }
    module->setAttr(kMaterializedAttr,
                    builder.getI32IntegerAttr(selected && selected->id != 0));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createTAScopeProfilePass(const TAScopeProfileOptions &options) {
  return std::make_unique<TAScopeProfilePass>(options);
}

} // namespace mlir::triton
