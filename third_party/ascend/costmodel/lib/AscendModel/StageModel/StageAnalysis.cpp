//===- StageAnalysis.cpp - Operation-derived Stage analysis -------------===//

#include "AscendModel/StageModel/StageAnalysis.h"

#include "AscendModel/StageModel/SimtAnchorAnalysis.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <system_error>

using namespace mlir;
using namespace mlir::ascend;

namespace {

static void recomputeIssueElements(StageWorkload &work) {
  double elements = work.scalarOperations + work.predicateElements;
  for (const auto &entry : work.operationElements)
    elements += entry.second;
  elements += 32.0 * (work.loadWarpInstructions + work.storeWarpInstructions);
  work.issueElements = elements;
}

static double getTypeElementCount(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    if (!shaped.hasStaticShape())
      return 1.0;
    return static_cast<double>(std::max<int64_t>(1, shaped.getNumElements()));
  }
  return 1.0;
}

static Type getScalarElementType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.getElementType();
  return type;
}

static std::string typeToString(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << type;
  return text;
}

static bool isPointerLikeType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  return llvm::StringRef(typeToString(type)).contains("!tt.ptr");
}

static bool isAddressOnlyLoopValue(Value root) {
  llvm::SmallVector<Value, 8> worklist{root};
  llvm::DenseSet<Value> visited;
  bool reachesAddressUse = false;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      llvm::StringRef name = user->getName().getStringRef();
      if (name == "scf.yield" || name == "scf.condition")
        continue;
      if ((name == "tt.load" || name == "tt.store" ||
           name.starts_with("tt.atomic")) &&
          use.getOperandNumber() == 0) {
        reachesAddressUse = true;
        continue;
      }
      bool forwarding =
          name == "tt.addptr" || name == "tt.advance" || name == "tt.splat" ||
          name == "tt.broadcast" || name == "tt.expand_dims" ||
          name == "arith.addi" || name == "arith.subi" ||
          name == "arith.muli" || name == "arith.index_cast";
      if (!forwarding)
        return false;
      if (name == "tt.addptr" || name == "tt.advance")
        reachesAddressUse = true;
      llvm::append_range(worklist, user->getResults());
    }
  }
  return reachesAddressUse;
}

static bool isAffineIndexInduction(BlockArgument argument) {
  Type type = getScalarElementType(argument.getType());
  if (!isa<IntegerType, IndexType>(type))
    return false;
  Operation *loop = argument.getOwner()->getParentOp();
  if (!loop || loop->getName().getStringRef() != "scf.for" ||
      argument.getArgNumber() == 0)
    return false;
  Operation *yield = argument.getOwner()->getTerminator();
  unsigned resultIndex = argument.getArgNumber() - 1;
  if (!yield || resultIndex >= yield->getNumOperands())
    return false;
  Operation *update = yield->getOperand(resultIndex).getDefiningOp();
  if (!update || update->getNumOperands() != 2)
    return false;
  llvm::StringRef name = update->getName().getStringRef();
  if (name != "arith.addi" && name != "arith.subi")
    return false;
  bool lhsIsArgument = update->getOperand(0) == argument;
  bool rhsIsArgument = update->getOperand(1) == argument;
  if (lhsIsArgument == rhsIsArgument ||
      (name == "arith.subi" && !lhsIsArgument))
    return false;
  Value step = lhsIsArgument ? update->getOperand(1) : update->getOperand(0);
  Operation *producer = step.getDefiningOp();
  return !producer || producer->getBlock() != argument.getOwner();
}

static int64_t getScalarBitWidth(Type type) {
  type = getScalarElementType(type);
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth();
  if (auto floating = dyn_cast<FloatType>(type))
    return floating.getWidth();
  return isa<IndexType>(type) ? 64 : 0;
}

static double getValueBytes(Value value) {
  int64_t bits = getScalarBitWidth(value.getType());
  return bits > 0 ? getTypeElementCount(value.getType()) * bits / 8.0 : 0.0;
}

static double getOperationElements(Operation *operation) {
  double elements = 1.0;
  for (Type type : operation->getResultTypes())
    elements = std::max(elements, getTypeElementCount(type));
  if (operation->getNumResults() == 0)
    for (Value value : operation->getOperands())
      elements = std::max(elements, getTypeElementCount(value.getType()));
  return elements;
}

static bool hasTensorResult(Operation *operation) {
  return llvm::any_of(operation->getResultTypes(),
                      [](Type type) { return isa<ShapedType>(type); });
}

static llvm::StringRef getProfileOperationName(Operation *operation) {
  llvm::StringRef name = operation->getName().getStringRef();
  return llvm::StringSwitch<llvm::StringRef>(name)
      .Cases("arith.addf", "tt.add", "f32.add")
      .Case("arith.subf", "f32.sub")
      .Case("arith.mulf", "f32.mul")
      .Case("arith.divf", "f32.div")
      .Cases("arith.maximumf", "arith.maxnumf", "f32.max")
      .Cases("math.absf", "tt.abs", "f32.abs")
      .Cases("math.exp", "tt.exp", "f32.exp")
      .Cases("math.log", "tt.log", "f32.log")
      .Cases("arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp",
             "convert.cast")
      .Cases("arith.fptosi", "arith.fptoui", "convert.cast")
      .Default("generic.issue");
}

static void accumulateDotWorkload(Operation *operation, StageWorkload &work) {
  if (operation->getNumOperands() < 2)
    return;
  auto lhs = dyn_cast<ShapedType>(operation->getOperand(0).getType());
  auto rhs = dyn_cast<ShapedType>(operation->getOperand(1).getType());
  if (!lhs || !rhs || !lhs.hasStaticShape() || !rhs.hasStaticShape() ||
      lhs.getRank() < 2 || rhs.getRank() < 2)
    return;
  int64_t m = lhs.getShape()[lhs.getRank() - 2];
  int64_t k = lhs.getShape()[lhs.getRank() - 1];
  int64_t n = rhs.getShape()[rhs.getRank() - 1];
  if (m > 0 && n > 0 && k > 0)
    work.dotFlops += 2.0 * static_cast<double>(m) * n * k;
}

static void accumulateReductionWorkload(Operation *operation,
                                        StageWorkload &work) {
  if (operation->getNumOperands() == 0)
    return;
  auto input = dyn_cast<ShapedType>(operation->getOperand(0).getType());
  auto axis = operation->getAttrOfType<IntegerAttr>("axis");
  if (!input || !input.hasStaticShape() || !axis || input.getRank() == 0)
    return;
  int64_t dimension = axis.getInt();
  if (dimension < 0)
    dimension += input.getRank();
  if (dimension < 0 || dimension >= input.getRank())
    return;
  int64_t extent = input.getShape()[dimension];
  if (extent > 1)
    work.shuffleLaneSteps +=
        getTypeElementCount(input) * std::ceil(std::log2(extent));
}

static void accumulateOneOperation(Operation *operation, StageWorkload &work) {
  if (!operation || operation->hasTrait<OpTrait::IsTerminator>())
    return;
  llvm::StringRef name = operation->getName().getStringRef();
  double elements = getOperationElements(operation);
  if ((name == "tt.load" || name == "tt.gather") &&
      operation->getNumResults() > 0) {
    work.loadBytes += getValueBytes(operation->getResult(0));
    work.loadWarpInstructions += std::ceil(elements / 32.0);
    return;
  }
  if ((name == "tt.store" || name.starts_with("tt.atomic")) &&
      operation->getNumOperands() > 1) {
    Value value = operation->getOperand(1);
    work.storeBytes += getValueBytes(value);
    work.storeWarpInstructions +=
        std::ceil(getTypeElementCount(value.getType()) / 32.0);
    return;
  }
  if (name == "tt.dot") {
    accumulateDotWorkload(operation, work);
    return;
  }
  if (name == "tt.reduce" || name == "tt.scan")
    accumulateReductionWorkload(operation, work);
  if (name == "arith.cmpi" || name == "arith.cmpf") {
    work.predicateElements += elements;
    return;
  }
  if (name == "scf.for" || name == "scf.if" || name == "scf.while")
    return;
  if (!hasTensorResult(operation)) {
    work.scalarOperations += 1.0;
    return;
  }
  work.operationElements[getProfileOperationName(operation)] += elements;
}

static void mergeWorkload(StageWorkload &into, const StageWorkload &from) {
  into.scalarOperations += from.scalarOperations;
  into.loadBytes += from.loadBytes;
  into.storeBytes += from.storeBytes;
  into.loadWarpInstructions += from.loadWarpInstructions;
  into.storeWarpInstructions += from.storeWarpInstructions;
  into.predicateElements += from.predicateElements;
  into.shuffleLaneSteps += from.shuffleLaneSteps;
  into.dotFlops += from.dotFlops;
  into.estimatedSpillTransactions += from.estimatedSpillTransactions;
  for (const auto &[name, elements] : from.operationElements)
    into.operationElements[name] += elements;
  recomputeIssueElements(into);
}

static void scaleWorkload(StageWorkload &work, double scale) {
  work.scalarOperations *= scale;
  work.loadBytes *= scale;
  work.storeBytes *= scale;
  work.loadWarpInstructions *= scale;
  work.storeWarpInstructions *= scale;
  work.predicateElements *= scale;
  work.shuffleLaneSteps *= scale;
  work.dotFlops *= scale;
  work.estimatedSpillTransactions *= scale;
  for (auto &entry : work.operationElements)
    entry.second *= scale;
  recomputeIssueElements(work);
}

static std::optional<int64_t> getConstantInteger(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  auto attribute = definition->getAttrOfType<IntegerAttr>("value");
  return attribute ? std::optional<int64_t>(attribute.getInt()) : std::nullopt;
}

static int64_t getLoopTripCount(Operation *operation,
                                int64_t fallbackTripCount) {
  llvm::StringRef name = operation->getName().getStringRef();
  if (name == "scf.for" && operation->getNumOperands() >= 3) {
    auto lower = getConstantInteger(operation->getOperand(0));
    auto upper = getConstantInteger(operation->getOperand(1));
    auto step = getConstantInteger(operation->getOperand(2));
    if (lower && upper && step && *step > 0 && *upper > *lower)
      return (*upper - *lower + *step - 1) / *step;
  }
  return name == "scf.for" || name == "scf.while"
             ? std::max<int64_t>(1, fallbackTripCount)
             : 1;
}

static void accumulateDynamicOperationTree(Operation *operation,
                                           StageWorkload &work,
                                           double multiplicity,
                                           int64_t fallbackTripCount) {
  if (!operation)
    return;
  StageWorkload local;
  accumulateOneOperation(operation, local);
  scaleWorkload(local, multiplicity);
  mergeWorkload(work, local);
  if (operation->hasAttr("ta.auto_blockify_v1.loop"))
    return;
  double childMultiplicity =
      multiplicity * getLoopTripCount(operation, fallbackTripCount);
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      for (Operation &nested : block)
        accumulateDynamicOperationTree(&nested, work, childMultiplicity,
                                       fallbackTripCount);
}

static int64_t countAlgorithmLoops(const LogicalStage &stage) {
  int64_t count = 0;
  for (Operation *root : stage.operations)
    if (root)
      root->walk([&](Operation *operation) {
        llvm::StringRef name = operation->getName().getStringRef();
        if ((name == "scf.for" || name == "scf.while") &&
            !operation->hasAttr("ta.auto_blockify_v1.loop"))
          ++count;
      });
  return count;
}

static void makePerIteration(LogicalStage &stage) {
  double count = std::max<int64_t>(1, stage.iterationCount);
  StageWorkload &work = stage.workload;
  work.scalarOperations /= count;
  work.loadBytes /= count;
  work.storeBytes /= count;
  work.loadWarpInstructions /= count;
  work.storeWarpInstructions /= count;
  work.predicateElements /= count;
  work.shuffleLaneSteps /= count;
  work.dotFlops /= count;
  work.estimatedSpillTransactions /= count;
  for (auto &entry : work.operationElements)
    entry.second /= count;
  recomputeIssueElements(work);
}

static void collectOwnedOperationTree(Operation *root,
                                      llvm::DenseSet<Operation *> &owned) {
  if (root)
    root->walk([&](Operation *operation) { owned.insert(operation); });
}

} // namespace

llvm::Error StageWorkloadAnalysis::analyze(LogicalStage &stage) const {
  if (stage.operations.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageWorkloadAnalysis requires operation ownership");
  StageWorkload work;
  work.paysKernelSetup = stage.workload.paysKernelSetup;
  int64_t loopCount = countAlgorithmLoops(stage);
  int64_t fallbackTripCount =
      loopCount > 0 ? std::max<int64_t>(1, stage.iterationCount / loopCount) : 1;
  for (Operation *root : stage.operations)
    accumulateDynamicOperationTree(root, work, 1.0, fallbackTripCount);
  recomputeIssueElements(work);
  stage.workload = std::move(work);
  makePerIteration(stage);
  if (!stage.workload.isFiniteAndNonNegative())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "Stage '%s' has invalid operation-derived workload", stage.id.c_str());
  return llvm::Error::success();
}

llvm::Error StageFeatureAnalysis::analyze(LogicalStage &stage) const {
  if (stage.operations.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageFeatureAnalysis requires operation ownership");
  StageModelFeatures &facts = stage.features;
  double activeLaneRatio = facts.activeLaneRatio;
  facts = StageModelFeatures{};
  facts.activeLaneRatio = activeLaneRatio;
  llvm::DenseSet<Operation *> owned;
  for (Operation *root : stage.operations)
    collectOwnedOperationTree(root, owned);
  bool hasMemory = false;
  int64_t algorithmLoopCount = 0;
  for (Operation *operation : owned) {
    llvm::StringRef name = operation->getName().getStringRef();
    if (name == "scf.for" || name == "scf.while") {
      facts.hasLoop = true;
      ++facts.loopBackedgeCount;
      if (!operation->hasAttr("ta.auto_blockify_v1.loop"))
        ++algorithmLoopCount;
      if (!operation->hasAttr("ta.auto_blockify_v1.loop") &&
          operation->getNumRegions() > 0 && !operation->getRegion(0).empty()) {
        Block &body = operation->getRegion(0).front();
        unsigned firstCarriedArgument = name == "scf.for" ? 1 : 0;
        for (unsigned index = firstCarriedArgument;
             index < body.getNumArguments(); ++index) {
          BlockArgument argument = body.getArgument(index);
          if (argument.use_empty())
            continue;
          if (isPointerLikeType(argument.getType()) ||
              isAffineIndexInduction(argument) ||
              isAddressOnlyLoopValue(argument))
            facts.hasPointerInduction = true;
          else
            facts.hasLoopCarriedDataDependency = true;
        }
        if (name == "scf.for" && body.getNumArguments() > 0 &&
            isAddressOnlyLoopValue(body.getArgument(0)))
          facts.hasPointerInduction = true;
      }
    }
    if (name == "scf.if" || name == "cf.cond_br") {
      ++facts.conditionalBranchCount;
      ++facts.divergentBranchCount;
    }
    if (name.contains("barrier") || name.contains("sync"))
      ++facts.synchronizationCount;
    if (name == "tt.load" || name == "tt.store" || name == "tt.gather" ||
        name.starts_with("tt.atomic")) {
      hasMemory = true;
      facts.hasIndirectMemory |= isLoadedIndexDependentMemoryOp(operation) ||
                                 name == "tt.gather" ||
                                 name.starts_with("tt.atomic");
    }
    facts.hasReduction |= name == "tt.reduce" || name == "tt.scan" ||
                          name == "linalg.reduce";
    facts.hasDot |= name == "tt.dot" || name.contains("matmul") ||
                    name.contains("mmad");
    facts.hasConversionPack |=
        name == "arith.extf" || name == "arith.truncf" ||
        name == "arith.fptosi" || name == "arith.fptoui" ||
        name == "arith.sitofp" || name == "arith.uitofp" ||
        name == "tt.fp_to_fp" || name.contains("convert") ||
        name.contains("pack") || name.contains("unpack");
  }
  facts.hasContiguousMemory = hasMemory && !facts.hasIndirectMemory;
  if (algorithmLoopCount > 0 && stage.iterationCount > 1) {
    if (facts.hasLoopCarriedDataDependency)
      facts.parallelRecurrenceGroupCount = algorithmLoopCount;
    facts.loopBackedgeCount = 1;
    facts.conditionalBranchCount = std::max<int64_t>(
        facts.conditionalBranchCount > 0 ? 1 : 0,
        facts.conditionalBranchCount / algorithmLoopCount);
    facts.divergentBranchCount = std::max<int64_t>(
        facts.divergentBranchCount > 0 ? 1 : 0,
        facts.divergentBranchCount / algorithmLoopCount);
  }
  facts.source = "exact post-layout TTIR operation graph";
  if (!facts.isValid())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Stage '%s' has invalid features",
                                   stage.id.c_str());
  return llvm::Error::success();
}

llvm::Error StageKindClassifier::analyze(LogicalStage &stage,
                                         int64_t tinyDotFlopsMax) const {
  const StageModelFeatures &facts = stage.features;
  if (facts.hasDot && (facts.hasReduction || facts.hasIndirectMemory ||
                       facts.hasLoopCarriedDataDependency))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "requires_split: Stage '%s' owns incompatible dominant structures",
        stage.id.c_str());

  if (facts.hasDot)
    stage.costModelKind =
        stage.workload.dotFlops * stage.iterationCount <=
                std::max<int64_t>(1, tinyDotFlopsMax)
            ? StageCostModelKind::TinyCubeRoofline
            : StageCostModelKind::CubeRoofline;
  else if (facts.hasReduction)
    stage.costModelKind = StageCostModelKind::RowwiseReduction;
  else if (facts.hasConversionPack)
    stage.costModelKind = StageCostModelKind::ConversionPack;
  else if (facts.hasLoop) {
    stage.costModelKind = facts.hasLoopCarriedDataDependency
                              ? StageCostModelKind::LoopCarriedRecurrence
                              : StageCostModelKind::IndependentPipelinedLoop;
    stage.scheduleKind = facts.hasLoopCarriedDataDependency
                             ? StageScheduleKind::LoopCarriedSerial
                             : StageScheduleKind::IndependentPipelined;
  } else if (facts.hasIndirectMemory)
    stage.costModelKind = StageCostModelKind::IndirectGatherMemory;
  else if (facts.hasContiguousMemory)
    stage.costModelKind = stage.workload.storeBytes > 0.0 &&
                                  stage.workload.loadBytes == 0.0
                              ? StageCostModelKind::ContinuousTileStore
                              : StageCostModelKind::ContinuousTileMemory;
  else
    stage.costModelKind = StageCostModelKind::ScalarIssue;
  return llvm::Error::success();
}
