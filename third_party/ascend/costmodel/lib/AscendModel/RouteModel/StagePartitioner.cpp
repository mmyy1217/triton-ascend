//===- StagePartitioner.cpp - Build semantic Phase/Stage IR -------------===//

#include "AscendModel/RouteModel/StagePartitioner.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <system_error>
#include <utility>

using namespace mlir;
using namespace mlir::ascend;

namespace {

static double mapValue(const llvm::StringMap<int64_t> &map,
                       llvm::StringRef name) {
  auto iterator = map.find(name);
  return iterator == map.end()
             ? 0.0
             : static_cast<double>(std::max<int64_t>(0, iterator->second));
}

static StageWorkload buildWorkload(const SimtAnchorFeatureSummary &features) {
  StageWorkload work;
  const std::pair<llvm::StringLiteral, llvm::StringLiteral> names[] = {
      {"f32.add", "add"},     {"f32.sub", "sub"}, {"f32.mul", "mul"},
      {"f32.div", "div"},     {"f32.max", "max"}, {"f32.abs", "abs"},
      {"f32.exp", "exp"},     {"f32.log", "log"}, {"convert.cast", "cast"},
      {"f32.clamp", "clamp"},
  };
  for (const auto &[profileName, featureName] : names) {
    const double elements = mapValue(features.opElements, featureName);
    if (elements > 0.0)
      work.operationElements[profileName] = elements;
  }
  work.scalarOperations = mapValue(features.weightedOps, "scalar");
  work.loadBytes = features.loadBytes;
  work.storeBytes = features.storeBytes;
  work.loadWarpInstructions =
      static_cast<double>(features.loadWarpInstructions);
  work.storeWarpInstructions =
      static_cast<double>(features.storeWarpInstructions);
  work.predicateElements = static_cast<double>(std::max<int64_t>(
      features.predicateLaneEvaluations, features.predicateElements));
  work.shuffleLaneSteps = static_cast<double>(features.shuffleLaneSteps);
  work.dotFlops = static_cast<double>(features.dotFlops);
  return work;
}

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
  stream.flush();
  return text;
}

static bool isPointerLikeType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  return llvm::StringRef(typeToString(type)).contains("!tt.ptr");
}

/// True when a loop argument only participates in address induction.  Such a
/// value is an implementation recurrence that later pointer canonicalization
/// can eliminate; it is not an algorithmic loop-carried dependency and must
/// not disable the SIMD independent-loop roofline model.
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
      const llvm::StringRef name = user->getName().getStringRef();
      if (name == "scf.yield" || name == "scf.condition")
        continue;
      if ((name == "tt.load" || name == "tt.store" ||
           name.starts_with("tt.atomic")) &&
          use.getOperandNumber() == 0) {
        reachesAddressUse = true;
        continue;
      }
      const bool addressForwarding =
          name == "tt.addptr" || name == "tt.advance" || name == "tt.splat" ||
          name == "tt.broadcast" || name == "tt.expand_dims" ||
          name == "arith.addi" || name == "arith.subi" ||
          name == "arith.muli" || name == "arith.index_cast";
      if (!addressForwarding)
        return false;
      if (name == "tt.addptr" || name == "tt.advance")
        reachesAddressUse = true;
      llvm::append_range(worklist, user->getResults());
    }
  }
  return reachesAddressUse;
}

static bool isAffineIndexInduction(BlockArgument argument) {
  Type type = argument.getType();
  if (auto shaped = dyn_cast<ShapedType>(type))
    type = shaped.getElementType();
  if (!isa<IntegerType, IndexType>(type))
    return false;
  Operation *loop = argument.getOwner()->getParentOp();
  if (!loop || loop->getName().getStringRef() != "scf.for" ||
      argument.getArgNumber() == 0)
    return false;
  Operation *yield = argument.getOwner()->getTerminator();
  const unsigned resultIndex = argument.getArgNumber() - 1;
  if (!yield || resultIndex >= yield->getNumOperands())
    return false;
  Operation *update = yield->getOperand(resultIndex).getDefiningOp();
  if (!update || update->getNumOperands() != 2)
    return false;
  llvm::StringRef name = update->getName().getStringRef();
  if (name != "arith.addi" && name != "arith.subi")
    return false;
  const bool lhsIsArgument = update->getOperand(0) == argument;
  const bool rhsIsArgument = update->getOperand(1) == argument;
  if (lhsIsArgument == rhsIsArgument ||
      (name == "arith.subi" && !lhsIsArgument))
    return false;
  Value step = lhsIsArgument ? update->getOperand(1) : update->getOperand(0);
  Operation *stepProducer = step.getDefiningOp();
  return !stepProducer || stepProducer->getBlock() != argument.getOwner();
}

static int64_t getScalarBitWidth(Type type) {
  type = getScalarElementType(type);
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth();
  if (auto floating = dyn_cast<FloatType>(type))
    return floating.getWidth();
  if (isa<IndexType>(type))
    return 64;
  return 0;
}

static double getValueBytes(Value value) {
  const int64_t bits = getScalarBitWidth(value.getType());
  return bits > 0 ? getTypeElementCount(value.getType()) *
                        static_cast<double>(bits) / 8.0
                  : 0.0;
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
  const llvm::StringRef name = operation->getName().getStringRef();
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
  const int64_t m = lhs.getShape()[lhs.getRank() - 2];
  const int64_t k = lhs.getShape()[lhs.getRank() - 1];
  const int64_t n = rhs.getShape()[rhs.getRank() - 1];
  if (m > 0 && n > 0 && k > 0)
    work.dotFlops += 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                     static_cast<double>(k);
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
  const int64_t extent = input.getShape()[dimension];
  if (extent <= 1)
    return;
  const double depth = std::ceil(std::log2(static_cast<double>(extent)));
  work.shuffleLaneSteps += getTypeElementCount(input) * depth;
}

static void accumulateOneOperation(Operation *operation, StageWorkload &work) {
  if (!operation || operation->hasTrait<OpTrait::IsTerminator>())
    return;
  const llvm::StringRef name = operation->getName().getStringRef();
  const double elements = getOperationElements(operation);

  if ((name == "tt.load" || name == "tt.gather") &&
      operation->getNumResults() > 0) {
    Value result = operation->getResult(0);
    work.loadBytes += getValueBytes(result);
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

static void mergeWorkload(StageWorkload &into, StageWorkload from);

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
  if (!attribute)
    return std::nullopt;
  return attribute.getInt();
}

static int64_t getLoopTripCount(Operation *operation,
                                int64_t stageIterationCount) {
  const llvm::StringRef name = operation->getName().getStringRef();
  if (name == "scf.for" && operation->getNumOperands() >= 3) {
    const std::optional<int64_t> lower =
        getConstantInteger(operation->getOperand(0));
    const std::optional<int64_t> upper =
        getConstantInteger(operation->getOperand(1));
    const std::optional<int64_t> step =
        getConstantInteger(operation->getOperand(2));
    if (lower && upper && step && *step > 0 && *upper > *lower)
      return (*upper - *lower + *step - 1) / *step;
  }
  if (name == "scf.for" || name == "scf.while")
    return std::max<int64_t>(1, stageIterationCount);
  return 1;
}

/// Accumulate dynamic work, not merely the number of syntactic TTIR ops.
/// A loop body appears once in TTIR but executes `tripCount` times.  The
/// resulting total dynamic work is normalized to one Stage iteration by
/// makePerIteration().  Consequently N_iter * C_body accounts for every
/// loop iteration instead of accidentally counting the body once.
/// AutoBlockify V1 is the exception: its loop is a scheduling shell and its
/// direct body operations are already separate semantic roots.
static void accumulateDynamicOperationTree(Operation *operation,
                                           StageWorkload &work,
                                           double multiplicity,
                                           int64_t fallbackLoopTripCount) {
  if (!operation)
    return;
  StageWorkload local;
  accumulateOneOperation(operation, local);
  scaleWorkload(local, multiplicity);
  mergeWorkload(work, std::move(local));

  if (operation->hasAttr("ta.auto_blockify_v1.loop"))
    return;
  const double childMultiplicity =
      multiplicity *
      static_cast<double>(getLoopTripCount(operation, fallbackLoopTripCount));
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      for (Operation &nested : block.getOperations())
        accumulateDynamicOperationTree(&nested, work, childMultiplicity,
                                       fallbackLoopTripCount);
}

static int64_t countAlgorithmLoops(const LogicalStage &stage) {
  int64_t count = 0;
  for (Operation *root : stage.operations) {
    if (!root || root->hasAttr("ta.auto_blockify_v1.loop"))
      continue;
    root->walk([&](Operation *operation) {
      const llvm::StringRef name = operation->getName().getStringRef();
      if ((name == "scf.for" || name == "scf.while") &&
          !operation->hasAttr("ta.auto_blockify_v1.loop"))
        ++count;
    });
  }
  return count;
}

static double consume(double &remaining, double requested) {
  const double value =
      std::min(std::max(0.0, remaining), std::max(0.0, requested));
  remaining -= value;
  return value;
}

static StageWorkload consumeExact(StageWorkload &remaining,
                                  const StageWorkload &requested) {
  StageWorkload result;
  result.scalarOperations =
      consume(remaining.scalarOperations, requested.scalarOperations);
  result.loadBytes = consume(remaining.loadBytes, requested.loadBytes);
  result.storeBytes = consume(remaining.storeBytes, requested.storeBytes);
  result.loadWarpInstructions =
      consume(remaining.loadWarpInstructions, requested.loadWarpInstructions);
  result.storeWarpInstructions =
      consume(remaining.storeWarpInstructions, requested.storeWarpInstructions);
  result.predicateElements =
      consume(remaining.predicateElements, requested.predicateElements);
  result.shuffleLaneSteps =
      consume(remaining.shuffleLaneSteps, requested.shuffleLaneSteps);
  result.dotFlops = consume(remaining.dotFlops, requested.dotFlops);
  result.estimatedSpillTransactions =
      consume(remaining.estimatedSpillTransactions,
              requested.estimatedSpillTransactions);
  for (const auto &[name, elements] : requested.operationElements) {
    double &available = remaining.operationElements[name];
    const double owned = consume(available, elements);
    if (owned > 0.0)
      result.operationElements[name] = owned;
  }
  recomputeIssueElements(result);
  recomputeIssueElements(remaining);
  return result;
}

static StageWorkload takeScalarAndPredicate(StageWorkload &remaining) {
  StageWorkload result;
  result.scalarOperations = std::exchange(remaining.scalarOperations, 0.0);
  result.predicateElements = std::exchange(remaining.predicateElements, 0.0);
  recomputeIssueElements(result);
  recomputeIssueElements(remaining);
  return result;
}

static StageWorkload takeLoads(StageWorkload &remaining) {
  StageWorkload result;
  result.loadBytes = std::exchange(remaining.loadBytes, 0.0);
  result.loadWarpInstructions =
      std::exchange(remaining.loadWarpInstructions, 0.0);
  recomputeIssueElements(result);
  recomputeIssueElements(remaining);
  return result;
}

static StageWorkload takeStores(StageWorkload &remaining) {
  StageWorkload result;
  result.storeBytes = std::exchange(remaining.storeBytes, 0.0);
  result.storeWarpInstructions =
      std::exchange(remaining.storeWarpInstructions, 0.0);
  recomputeIssueElements(result);
  recomputeIssueElements(remaining);
  return result;
}

static StageWorkload takeDot(StageWorkload &remaining) {
  StageWorkload result;
  result.dotFlops = std::exchange(remaining.dotFlops, 0.0);
  recomputeIssueElements(result);
  recomputeIssueElements(remaining);
  return result;
}

static void moveOperation(StageWorkload &from, StageWorkload &to,
                          llvm::StringRef name) {
  auto iterator = from.operationElements.find(name);
  if (iterator == from.operationElements.end())
    return;
  to.operationElements[name] += iterator->second;
  from.operationElements.erase(iterator);
}

static StageWorkload takeAllOperations(StageWorkload &remaining) {
  StageWorkload result;
  result.operationElements = std::move(remaining.operationElements);
  remaining.operationElements.clear();
  recomputeIssueElements(result);
  recomputeIssueElements(remaining);
  return result;
}

static void mergeWorkload(StageWorkload &into, StageWorkload from) {
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

static void makePerIteration(LogicalStage &stage) {
  const double count =
      static_cast<double>(std::max<int64_t>(1, stage.iterationCount));
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

static LogicalStage makeStage(llvm::StringRef id, llvm::StringRef description,
                              StageCostModelKind kind,
                              StageScheduleKind schedule, int64_t iterations,
                              StageWorkload workload) {
  LogicalStage stage;
  stage.id = id.str();
  stage.description = description.str();
  stage.costModelKind = kind;
  stage.scheduleKind = schedule;
  stage.iterationCount = std::max<int64_t>(1, iterations);
  stage.workload = std::move(workload);
  makePerIteration(stage);
  return stage;
}

static LogicalStage withControl(LogicalStage stage, int64_t branches,
                                int64_t divergent, double activeLaneRatio) {
  stage.features.conditionalBranchCount = std::max<int64_t>(0, branches);
  stage.features.divergentBranchCount = std::max<int64_t>(0, divergent);
  stage.features.activeLaneRatio = std::clamp(activeLaneRatio, 0.0, 1.0);
  return stage;
}

static LogicalStage asLocalSIMT(LogicalStage stage) {
  stage.localSimtMaterializable = true;
  // StageModeLegalityAnalysis opens F2/F4 only when backend integration has
  // the AutoBlockify V1 Scope SuperBlock wrapper available.
  stage.localSimtFactors = {1};
  return stage;
}

static void addPhase(StagePartition &partition, llvm::StringRef id,
                     llvm::StringRef description, LogicalStage stage) {
  LogicalPhase phase;
  phase.id = id.str();
  phase.description = description.str();
  phase.stages.push_back(std::move(stage));
  partition.phases.push_back(std::move(phase));
}

static bool operationTreeContainsName(Operation *root, llvm::StringRef name);

static bool hasWork(const StageWorkload &work) {
  if (work.scalarOperations > 0.0 || work.loadBytes > 0.0 ||
      work.storeBytes > 0.0 || work.predicateElements > 0.0 ||
      work.shuffleLaneSteps > 0.0 || work.dotFlops > 0.0 ||
      work.estimatedSpillTransactions > 0.0)
    return true;
  return llvm::any_of(work.operationElements,
                      [](const auto &entry) { return entry.second > 0.0; });
}

static void prependAutoBlockifyStages(StagePartition &partition,
                                      StageWorkload &remaining,
                                      const SimdSimtFeatureSummary &features,
                                      bool operationGraphHasAutoBlockify) {
  if (!features.autoBlockifyV1Applied && !operationGraphHasAutoBlockify)
    return;
  StageWorkload dispatch;
  dispatch.scalarOperations =
      consume(remaining.scalarOperations,
              static_cast<double>(features.autoBlockifyV1ScheduleOpCount));
  recomputeIssueElements(dispatch);
  LogicalPhase phase;
  phase.id = "auto_blockify_dispatch";
  phase.description = "AutoBlockify V1 physical/logical program dispatch";
  phase.stages.push_back(makeStage(
      "physical_program_dispatch", "Map physical PID to logical block range",
      StageCostModelKind::AutoBlockifyDispatch, StageScheduleKind::StraightLine,
      1, std::move(dispatch)));
  if (features.autoBlockifyV1LoopCount > 0 || operationGraphHasAutoBlockify) {
    StageWorkload loop;
    loop.scalarOperations =
        consume(remaining.scalarOperations,
                static_cast<double>(features.autoBlockifyV1LoopCount));
    recomputeIssueElements(loop);
    phase.stages.push_back(makeStage(
        "logical_program_loop", "Iterate AutoBlockify V1 logical programs",
        StageCostModelKind::AutoBlockifyLoop,
        StageScheduleKind::IndependentPipelined,
        std::max<int64_t>(1, features.autoBlockifyV1LoopCount),
        std::move(loop)));
  }
  partition.phases.push_back(std::move(phase));
}

static StagePartition
partitionTriangular(const SimdSimtFeatureSummary &features,
                    const TriangularSolveFacts &facts,
                    const PhaseBoundaryPlan *operationGraphPlan) {
  StagePartition partition;
  partition.domain = "triangular_recurrence";
  StageWorkload remaining = buildKernelStageWorkload(features);
  StageWorkload anchor = buildWorkload(features.simtAnchors);
  anchor = consumeExact(remaining, anchor);

  const bool graphHasAutoBlockify =
      operationGraphPlan && llvm::is_contained(operationGraphPlan->rootPhaseIds,
                                               "auto_blockify_dispatch");
  prependAutoBlockifyStages(partition, remaining, features,
                            graphHasAutoBlockify);

  StageWorkload head = takeScalarAndPredicate(remaining);
  mergeWorkload(head, takeAllOperations(remaining));
  head.paysKernelSetup = true;
  addPhase(partition, "head", "Tile offset and triangular mask setup",
           withControl(makeStage("head_index_mask",
                                 "Scalar indices and triangular masks",
                                 StageCostModelKind::PredicateMask,
                                 StageScheduleKind::StraightLine, 1,
                                 std::move(head)),
                       features.conditionalBranchCount -
                           features.simtAnchors.conditionalBranchCount,
                       features.divergentBranchCount -
                           features.simtAnchors.divergentBranchCount,
                       features.activeLaneRatio));

  StageWorkload loads = takeLoads(remaining);
  const bool graphHasDiagonalLoad =
      operationGraphPlan &&
      llvm::is_contained(operationGraphPlan->rootPhaseIds, "diagonal_load");
  if (hasWork(loads) || graphHasDiagonalLoad)
    addPhase(partition, "diagonal_load", "Load diagonal tile data",
             makeStage("load_diagonal_tiles", "Continuous diagonal tile loads",
                       StageCostModelKind::ContinuousTileMemory,
                       StageScheduleKind::IndependentPipelined, 1,
                       std::move(loads)));

  const int64_t recurrenceIterations =
      std::max<int64_t>(1, facts.recurrenceLoopCount);
  // The feature-summary fallback has no owned operation graph from which to
  // count sibling loops.  Derive the number of independent 16x16 recurrence
  // groups from the structural triangular facts.  Production operation-graph
  // analysis below recomputes the same field directly from owned loops.
  LogicalStage recurrence = asLocalSIMT(withControl(
      makeStage("diagonal_inverse_recurrence",
                "Predicate, short reduction and recurrent state update",
                StageCostModelKind::LoopCarriedRecurrence,
                StageScheduleKind::LoopCarriedSerial, recurrenceIterations,
                std::move(anchor)),
      features.simtAnchors.conditionalBranchCount,
      features.simtAnchors.divergentBranchCount,
      features.simtAnchors.activeLaneRatio));
  const int64_t iterationsPerGroup =
      std::max<int64_t>(1, facts.blockRows - facts.recurrenceStartRow);
  recurrence.features.parallelRecurrenceGroupCount = std::max<int64_t>(
      1, (recurrenceIterations + iterationsPerGroup - 1) / iterationsPerGroup);
  addPhase(partition, "diagonal_inverse",
           "Loop-carried blockwise triangular recurrence",
           std::move(recurrence));

  StageWorkload dot = takeDot(remaining);
  StageWorkload stores = takeStores(remaining);
  mergeWorkload(stores, std::move(remaining));
  bool graphHasDot = false;
  bool graphHasStore = false;
  if (operationGraphPlan)
    for (auto indexedRoot :
         llvm::enumerate(operationGraphPlan->rootOperations)) {
      if (operationGraphPlan->rootPhaseIds[indexedRoot.index()] !=
          "merge_store")
        continue;
      graphHasDot |= operationTreeContainsName(indexedRoot.value(), "tt.dot");
      graphHasStore |=
          operationTreeContainsName(indexedRoot.value(), "tt.store");
    }
  if (hasWork(dot) || hasWork(stores) || graphHasDot || graphHasStore) {
    LogicalPhase mergeStore;
    mergeStore.id = "merge_store";
    mergeStore.description = "Dense off-diagonal merge and result store";
    if (hasWork(dot) || graphHasDot)
      mergeStore.stages.push_back(makeStage(
          "dense_dot_tail", "Dense dot tail preserved for Cube",
          StageCostModelKind::CubeRoofline,
          StageScheduleKind::IndependentPipelined,
          std::max<int64_t>(1, facts.denseDotTailOps), std::move(dot)));
    if (hasWork(stores) || graphHasStore)
      mergeStore.stages.push_back(
          makeStage("store_inverse_tile", "Continuous result tile store",
                    StageCostModelKind::ContinuousTileStore,
                    StageScheduleKind::StraightLine, 1, std::move(stores)));
    partition.phases.push_back(std::move(mergeStore));
  }
  return partition;
}

static StagePartition
partitionRowwise(const SimdSimtFeatureSummary &features,
                 const PhaseBoundaryPlan *operationGraphPlan) {
  StagePartition partition;
  partition.domain = "loaded_index_rowwise_reduction";
  StageWorkload remaining = buildKernelStageWorkload(features);
  const bool graphHasAutoBlockify =
      operationGraphPlan && llvm::is_contained(operationGraphPlan->rootPhaseIds,
                                               "auto_blockify_dispatch");
  prependAutoBlockifyStages(partition, remaining, features,
                            graphHasAutoBlockify);

  StageWorkload index = takeScalarAndPredicate(remaining);
  index.paysKernelSetup = true;
  addPhase(partition, "row_dispatch", "Resolve token and row indices",
           withControl(
               makeStage("row_index_generation", "Scalar row index and masks",
                         StageCostModelKind::IndexGeneration,
                         StageScheduleKind::StraightLine, 1, std::move(index)),
               features.conditionalBranchCount, features.divergentBranchCount,
               features.activeLaneRatio));

  addPhase(partition, "row_load", "Loaded-index-dependent row access",
           asLocalSIMT(makeStage("indirect_row_gather",
                                 "Gather the selected row tile",
                                 StageCostModelKind::IndirectGatherMemory,
                                 StageScheduleKind::PartiallyDependent, 1,
                                 takeLoads(remaining))));

  StageWorkload reduction;
  reduction.shuffleLaneSteps = std::exchange(remaining.shuffleLaneSteps, 0.0);
  moveOperation(remaining, reduction, "f32.max");
  recomputeIssueElements(reduction);
  const int64_t reductionIterations =
      std::max<int64_t>(1, features.staticLoopTripCountMax);
  addPhase(partition, "row_reduction", "Reduce each selected row",
           makeStage("rowwise_reduction", "Row-local reduction tree",
                     StageCostModelKind::RowwiseReduction,
                     StageScheduleKind::PartiallyDependent, reductionIterations,
                     std::move(reduction)));

  StageWorkload convert = takeStores(remaining);
  mergeWorkload(convert, takeAllOperations(remaining));
  mergeWorkload(convert, std::move(remaining));
  addPhase(partition, "convert_store", "Scale, convert, pack and store",
           makeStage("conversion_pack_store", "Conversion and packed output",
                     StageCostModelKind::ConversionPack,
                     StageScheduleKind::IndependentPipelined,
                     reductionIterations, std::move(convert)));
  return partition;
}

static StagePartition
partitionIndirectDot(const SimdSimtFeatureSummary &features,
                     const PhaseBoundaryPlan *operationGraphPlan) {
  StagePartition partition;
  partition.domain = "indirect_underfilled_dot";
  StageWorkload remaining = buildKernelStageWorkload(features);
  const bool graphHasAutoBlockify =
      operationGraphPlan && llvm::is_contained(operationGraphPlan->rootPhaseIds,
                                               "auto_blockify_dispatch");
  prependAutoBlockifyStages(partition, remaining, features,
                            graphHasAutoBlockify);

  StageWorkload index = takeScalarAndPredicate(remaining);
  mergeWorkload(index, takeAllOperations(remaining));
  index.paysKernelSetup = true;
  addPhase(partition, "index_setup", "Generate gather indices and masks",
           withControl(
               makeStage("index_generation", "Index and predicate generation",
                         StageCostModelKind::IndexGeneration,
                         StageScheduleKind::StraightLine, 1, std::move(index)),
               features.conditionalBranchCount, features.divergentBranchCount,
               features.activeLaneRatio));

  addPhase(partition, "gather_tiles", "Gather dot input tiles",
           asLocalSIMT(makeStage("indirect_tile_gather",
                                 "Loaded-index-dependent operand gathers",
                                 StageCostModelKind::IndirectGatherMemory,
                                 StageScheduleKind::PartiallyDependent, 1,
                                 takeLoads(remaining))));

  addPhase(partition, "dot", "Under-filled matrix product",
           makeStage("tiny_cube_dot", "Small dot with Cube underfill",
                     StageCostModelKind::TinyCubeRoofline,
                     StageScheduleKind::IndependentPipelined, 1,
                     takeDot(remaining)));

  StageWorkload store = takeStores(remaining);
  mergeWorkload(store, std::move(remaining));
  addPhase(partition, "output_store", "Write dot result",
           makeStage("store_dot_result", "Continuous result store",
                     StageCostModelKind::ContinuousTileStore,
                     StageScheduleKind::StraightLine, 1, std::move(store)));
  return partition;
}

static bool anchorMatchesStage(const SimtAnchorDescriptor &anchor,
                               const LogicalStage &stage) {
  if (!anchor.materializable || !stage.localSimtMaterializable)
    return false;
  if (stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence)
    return anchor.kind == SimtAnchorKind::TriangularSolveLoop;
  if (stage.costModelKind == StageCostModelKind::IndirectGatherMemory ||
      stage.costModelKind == StageCostModelKind::IndirectScalarMemory)
    return anchor.kind == SimtAnchorKind::DirectGather ||
           anchor.kind == SimtAnchorKind::LoadedIndexDependentMemory;
  return false;
}

static Operation *getTopLevelSemanticRoot(Operation *operation);

static void attachAnchorOperationOwnership(StagePartition &partition,
                                           const SimtAnchorPlan &anchorPlan) {
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      if (!stage.localSimtMaterializable)
        continue;
      for (auto indexedAnchor : llvm::enumerate(anchorPlan.anchors)) {
        const SimtAnchorDescriptor &anchor = indexedAnchor.value();
        if (!anchorMatchesStage(anchor, stage))
          continue;
        stage.simtAnchorIndices.push_back(
            static_cast<unsigned>(indexedAnchor.index()));
        if (anchor.scopeOperations.empty()) {
          if (anchor.operation &&
              !llvm::is_contained(stage.operations, anchor.operation))
            stage.operations.push_back(anchor.operation);
          continue;
        }
        for (Operation *operation : anchor.scopeOperations)
          if (operation && !llvm::is_contained(stage.operations, operation))
            stage.operations.push_back(operation);
      }
      // A production mixed Stage must be backed by the exact operations that
      // the materializer will consume.  Synthetic feature-only tests use the
      // overload without an anchor plan and retain fallback behavior.
      stage.localSimtMaterializable = !stage.operations.empty();
      if (!stage.localSimtMaterializable)
        stage.localSimtFactors.clear();
    }
  }
}

static bool stageOwnsAnchor(const LogicalStage &stage,
                            const SimtAnchorDescriptor &anchor) {
  if (!anchorMatchesStage(anchor, stage))
    return false;
  auto owns = [&](Operation *operation) {
    Operation *root = getTopLevelSemanticRoot(operation);
    return root && llvm::is_contained(stage.operations, root);
  };
  if (anchor.scopeOperations.empty())
    return owns(anchor.operation);
  return llvm::all_of(anchor.scopeOperations, owns);
}

/// Associate exact anchor descriptors with their owning Stage after complete
/// operation ownership has been established.  This is deliberately separate
/// from operation assignment: anchors are materialization evidence, not a
/// second source of Stage boundaries.
static void attachExactAnchorOwnership(StagePartition &partition,
                                       const SimtAnchorPlan &anchorPlan) {
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      if (!stage.localSimtMaterializable)
        continue;
      stage.simtAnchorIndices.clear();
      for (auto indexedAnchor : llvm::enumerate(anchorPlan.anchors)) {
        const SimtAnchorDescriptor &anchor = indexedAnchor.value();
        if (anchor.materializable && stageOwnsAnchor(stage, anchor))
          stage.simtAnchorIndices.push_back(
              static_cast<unsigned>(indexedAnchor.index()));
      }
      stage.localSimtMaterializable = !stage.simtAnchorIndices.empty();
      if (!stage.localSimtMaterializable)
        stage.localSimtFactors.clear();
    }
  }
}

static bool isFunctionLikeTTIROp(Operation *operation) {
  if (!operation)
    return false;
  const llvm::StringRef name = operation->getName().getStringRef();
  return name == "tt.func" || name == "func.func";
}

static Operation *getTopLevelSemanticRoot(Operation *operation) {
  if (!operation)
    return nullptr;
  Operation *root = operation;
  while (Operation *parent = root->getParentOp()) {
    if (isFunctionLikeTTIROp(parent) ||
        parent->hasAttr("ta.auto_blockify_v1.loop"))
      return root;
    root = parent;
  }
  return nullptr;
}

static std::vector<Operation *> collectTopLevelSemanticRoots(ModuleOp module) {
  std::vector<Operation *> result;
  auto appendBlock = [&](Block &block) {
    for (Operation &nested : block.getOperations()) {
      if (nested.hasTrait<OpTrait::IsTerminator>())
        continue;
      result.push_back(&nested);
      // AutoBlockify V1's scf.for is a scheduling shell.  Own the shell as
      // loop control, then expose its direct body operations as semantic
      // roots.  Other structured operations remain atomic roots so their
      // nested recurrence/reduction work is not double-owned.
      if (!nested.hasAttr("ta.auto_blockify_v1.loop") ||
          nested.getNumRegions() == 0)
        continue;
      for (Block &body : nested.getRegion(0))
        for (Operation &bodyOperation : body.getOperations())
          if (!bodyOperation.hasTrait<OpTrait::IsTerminator>())
            result.push_back(&bodyOperation);
    }
  };
  for (Operation &operation : module.getBody()->getOperations()) {
    if (!isFunctionLikeTTIROp(&operation) || operation.getNumRegions() == 0)
      continue;
    for (Block &block : operation.getRegion(0))
      appendBlock(block);
  }
  return result;
}

static bool operationTreeContainsName(Operation *root, llvm::StringRef name) {
  bool found = root && root->getName().getStringRef() == name;
  if (!root || found)
    return found;
  root->walk([&](Operation *nested) {
    found |= nested->getName().getStringRef() == name;
  });
  return found;
}

static bool operationTreeContainsLoadedIndexMemory(Operation *root) {
  bool found = root && isLoadedIndexDependentMemoryOp(root);
  if (!root || found)
    return found;
  root->walk([&](Operation *nested) {
    if (!found)
      found = isLoadedIndexDependentMemoryOp(nested);
  });
  return found;
}

/// PhaseBoundaryAnalysis owns the algorithm-level serial cut.  Each root is
/// assigned exactly one Phase id in execution order.  The state machines are
/// monotone: after a boundary is crossed, a later root cannot move back to an
/// earlier Phase.  Cost and candidate mode are intentionally absent here.
static llvm::Error assignRootPhaseIds(PhaseBoundaryPlan &plan) {
  enum class TriangularPhase { Head, Load, Recurrence, MergeStore };
  enum class RowwisePhase { Index, Gather, Reduction, ConvertStore };
  enum class IndirectDotPhase { Index, Gather, Dot, OutputStore };
  TriangularPhase triangular = TriangularPhase::Head;
  RowwisePhase rowwise = RowwisePhase::Index;
  IndirectDotPhase indirectDot = IndirectDotPhase::Index;
  llvm::DenseSet<Operation *> anchorRoots(plan.localSimtAnchorRoots.begin(),
                                          plan.localSimtAnchorRoots.end());
  std::optional<size_t> firstAnchorIndex;
  std::optional<size_t> lastAnchorIndex;
  for (auto indexedRoot : llvm::enumerate(plan.rootOperations)) {
    if (!anchorRoots.contains(indexedRoot.value()))
      continue;
    if (!firstAnchorIndex)
      firstAnchorIndex = indexedRoot.index();
    lastAnchorIndex = indexedRoot.index();
  }
  if (plan.domain == PhaseBoundaryDomain::TriangularRecurrence &&
      !firstAnchorIndex)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "triangular PhaseBoundaryAnalysis requires an exact anchor region");
  if (plan.domain == PhaseBoundaryDomain::TriangularRecurrence &&
      firstAnchorIndex && lastAnchorIndex) {
    for (size_t index = *firstAnchorIndex; index <= *lastAnchorIndex; ++index)
      if (!anchorRoots.contains(plan.rootOperations[index]))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "triangular PhaseBoundaryAnalysis requires a contiguous planned "
            "scope region");
  }

  plan.rootPhaseIds.clear();
  plan.rootPhaseIds.reserve(plan.rootOperations.size());
  for (auto indexedRoot : llvm::enumerate(plan.rootOperations)) {
    Operation *root = indexedRoot.value();
    if (!root)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "PhaseBoundaryAnalysis received a null semantic root");
    if (root->hasAttr("ta.auto_blockify_v1.loop") ||
        root->hasAttr("ta.auto_blockify_v1.schedule")) {
      plan.rootPhaseIds.push_back("auto_blockify_dispatch");
      continue;
    }

    switch (plan.domain) {
    case PhaseBoundaryDomain::TriangularRecurrence:
      if (firstAnchorIndex && indexedRoot.index() >= *firstAnchorIndex &&
          indexedRoot.index() <= *lastAnchorIndex)
        triangular = TriangularPhase::Recurrence;
      else if (lastAnchorIndex && indexedRoot.index() > *lastAnchorIndex)
        triangular = TriangularPhase::MergeStore;
      else if (operationTreeContainsName(root, "tt.dot") ||
               operationTreeContainsName(root, "tt.store"))
        triangular = TriangularPhase::MergeStore;
      else if (triangular == TriangularPhase::Head &&
               operationTreeContainsName(root, "tt.load"))
        triangular = TriangularPhase::Load;
      switch (triangular) {
      case TriangularPhase::Head:
        plan.rootPhaseIds.push_back("head");
        break;
      case TriangularPhase::Load:
        plan.rootPhaseIds.push_back("diagonal_load");
        break;
      case TriangularPhase::Recurrence:
        plan.rootPhaseIds.push_back("diagonal_inverse");
        break;
      case TriangularPhase::MergeStore:
        plan.rootPhaseIds.push_back("merge_store");
        break;
      }
      break;
    case PhaseBoundaryDomain::LoadedIndexRowwiseReduction:
      if (operationTreeContainsName(root, "tt.reduce"))
        rowwise = RowwisePhase::Reduction;
      else if (rowwise == RowwisePhase::Reduction ||
               operationTreeContainsName(root, "tt.store"))
        rowwise = RowwisePhase::ConvertStore;
      else if (rowwise == RowwisePhase::Index &&
               operationTreeContainsLoadedIndexMemory(root))
        rowwise = RowwisePhase::Gather;
      switch (rowwise) {
      case RowwisePhase::Index:
        plan.rootPhaseIds.push_back("row_dispatch");
        break;
      case RowwisePhase::Gather:
        plan.rootPhaseIds.push_back("row_load");
        break;
      case RowwisePhase::Reduction:
        plan.rootPhaseIds.push_back("row_reduction");
        break;
      case RowwisePhase::ConvertStore:
        plan.rootPhaseIds.push_back("convert_store");
        break;
      }
      break;
    case PhaseBoundaryDomain::IndirectUnderfilledDot:
      if (operationTreeContainsName(root, "tt.dot"))
        indirectDot = IndirectDotPhase::Dot;
      else if (indirectDot == IndirectDotPhase::Dot ||
               operationTreeContainsName(root, "tt.store"))
        indirectDot = IndirectDotPhase::OutputStore;
      else if (indirectDot == IndirectDotPhase::Index &&
               operationTreeContainsLoadedIndexMemory(root))
        indirectDot = IndirectDotPhase::Gather;
      switch (indirectDot) {
      case IndirectDotPhase::Index:
        plan.rootPhaseIds.push_back("index_setup");
        break;
      case IndirectDotPhase::Gather:
        plan.rootPhaseIds.push_back("gather_tiles");
        break;
      case IndirectDotPhase::Dot:
        plan.rootPhaseIds.push_back("dot");
        break;
      case IndirectDotPhase::OutputStore:
        plan.rootPhaseIds.push_back("output_store");
        break;
      }
      break;
    }
  }
  if (plan.rootPhaseIds.size() != plan.rootOperations.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "PhaseBoundaryAnalysis did not own every semantic root");
  llvm::StringSet<> closedPhases;
  llvm::StringRef currentPhase;
  for (const std::string &phaseId : plan.rootPhaseIds) {
    if (phaseId == currentPhase)
      continue;
    if (!currentPhase.empty())
      closedPhases.insert(currentPhase);
    if (closedPhases.contains(phaseId))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "PhaseBoundaryAnalysis produced a non-contiguous Phase '%s'",
          phaseId.c_str());
    currentPhase = phaseId;
  }
  return llvm::Error::success();
}

static LogicalStage *findStage(StagePartition &partition, llvm::StringRef id) {
  for (LogicalPhase &phase : partition.phases)
    for (LogicalStage &stage : phase.stages)
      if (stage.id == id)
        return &stage;
  return nullptr;
}

static int64_t getStageOrdinal(const StagePartition &partition,
                               const LogicalStage *target) {
  int64_t ordinal = 0;
  for (const LogicalPhase &phase : partition.phases) {
    for (const LogicalStage &stage : phase.stages) {
      if (&stage == target)
        return ordinal;
      ++ordinal;
    }
  }
  return -1;
}

static llvm::Error assignOperation(LogicalStage *stage, Operation *operation,
                                   llvm::DenseSet<Operation *> &owned) {
  if (!stage || !operation)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageBoundaryAnalysis could not map a TTIR operation");
  if (!owned.insert(operation).second)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageBoundaryAnalysis assigned a TTIR operation more than once");
  stage->operations.push_back(operation);
  return llvm::Error::success();
}

static llvm::Error
attachCompleteOperationOwnership(StagePartition &partition,
                                 const PhaseBoundaryPlan &plan) {
  if (!plan.hasOperationGraph())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageBoundaryAnalysis requires complete Phase root ownership");
  llvm::DenseSet<Operation *> owned;
  int64_t lastStageOrdinal = -1;
  bool mergeStoreReached = findStage(partition, "dense_dot_tail") == nullptr;

  for (auto indexedRoot : llvm::enumerate(plan.rootOperations)) {
    Operation *root = indexedRoot.value();
    const llvm::StringRef phaseId = plan.rootPhaseIds[indexedRoot.index()];
    LogicalStage *target = nullptr;
    if (phaseId == "auto_blockify_dispatch") {
      if (root->hasAttr("ta.auto_blockify_v1.loop"))
        target = findStage(partition, "logical_program_loop");
      else if (!root->hasAttr("ta.auto_blockify_v1.schedule"))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "AutoBlockify Phase contains a root without V1 provenance");
      Operation *parent = root->getParentOp();
      if (!target)
        target = findStage(partition,
                           parent && parent->hasAttr("ta.auto_blockify_v1.loop")
                               ? "logical_program_loop"
                               : "physical_program_dispatch");
    }

    if (!target) {
      switch (plan.domain) {
      case PhaseBoundaryDomain::TriangularRecurrence:
        if (phaseId == "head")
          target = findStage(partition, "head_index_mask");
        else if (phaseId == "diagonal_load")
          target = findStage(partition, "load_diagonal_tiles");
        else if (phaseId == "diagonal_inverse")
          target = findStage(partition, "diagonal_inverse_recurrence");
        else if (phaseId == "merge_store") {
          if (operationTreeContainsName(root, "tt.store"))
            mergeStoreReached = true;
          target = findStage(partition, mergeStoreReached ? "store_inverse_tile"
                                                          : "dense_dot_tail");
        }
        break;
      case PhaseBoundaryDomain::LoadedIndexRowwiseReduction:
        if (phaseId == "row_dispatch")
          target = findStage(partition, "row_index_generation");
        else if (phaseId == "row_load")
          target = findStage(partition, "indirect_row_gather");
        else if (phaseId == "row_reduction")
          target = findStage(partition, "rowwise_reduction");
        else if (phaseId == "convert_store")
          target = findStage(partition, "conversion_pack_store");
        break;
      case PhaseBoundaryDomain::IndirectUnderfilledDot:
        if (phaseId == "index_setup")
          target = findStage(partition, "index_generation");
        else if (phaseId == "gather_tiles")
          target = findStage(partition, "indirect_tile_gather");
        else if (phaseId == "dot")
          target = findStage(partition, "tiny_cube_dot");
        else if (phaseId == "output_store")
          target = findStage(partition, "store_dot_result");
        break;
      }
    }
    const int64_t ordinal = getStageOrdinal(partition, target);
    if (ordinal < 0)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "StageBoundaryAnalysis selected a missing Stage for Phase '%s' "
          "and root '%s'",
          phaseId.str().c_str(), root->getName().getStringRef().str().c_str());
    if (ordinal < lastStageOrdinal)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "StageBoundaryAnalysis produced non-contiguous Stage ownership");
    lastStageOrdinal = ordinal;
    if (llvm::Error error = assignOperation(target, root, owned))
      return error;
  }

  if (owned.size() != plan.rootOperations.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageBoundaryAnalysis did not conserve TTIR operation ownership");
  partition.boundarySource = "operation_graph";
  partition.operationOwnershipComplete = true;
  partition.modeledOperationCount =
      static_cast<int64_t>(plan.rootOperations.size());
  return llvm::Error::success();
}

static void collectOwnedOperationTree(Operation *root,
                                      llvm::DenseSet<Operation *> &owned) {
  if (!root)
    return;
  owned.insert(root);
  // The AutoBlockify loop is intentionally split into a scheduling shell and
  // direct semantic body roots.  Treating the shell as the owner of its body
  // would double-own every algorithm operation.
  if (root->hasAttr("ta.auto_blockify_v1.loop"))
    return;
  root->walk([&](Operation *nested) {
    if (nested != root)
      owned.insert(nested);
  });
}

static bool isValueDefinedInside(Value value,
                                 const llvm::DenseSet<Operation *> &owned) {
  if (Operation *definition = value.getDefiningOp())
    return owned.contains(definition);
  auto argument = dyn_cast<BlockArgument>(value);
  Operation *parent = argument ? argument.getOwner()->getParentOp() : nullptr;
  return parent && owned.contains(parent);
}

static int64_t staticTensorBytes(Value value) {
  auto shaped = dyn_cast<ShapedType>(value.getType());
  if (!shaped || !shaped.hasStaticShape())
    return 0;
  Type elementType = shaped.getElementType();
  if (!isa<IntegerType, FloatType>(elementType))
    return 0;
  const int64_t elements = shaped.getNumElements();
  const int64_t bits = elementType.getIntOrFloatBitWidth();
  if (elements <= 0 || bits <= 0)
    return 0;
  constexpr int64_t maximum = std::numeric_limits<int64_t>::max();
  if (elements > (maximum - 7) / bits)
    return maximum;
  return (elements * bits + 7) / 8;
}

static int64_t staticTensorBytes(llvm::ArrayRef<Value> values) {
  int64_t total = 0;
  for (Value value : values) {
    const int64_t bytes = staticTensorBytes(value);
    if (bytes > std::numeric_limits<int64_t>::max() - total)
      return std::numeric_limits<int64_t>::max();
    total += bytes;
  }
  return total;
}

/// Derive the exact SSA contract of every Stage from operation ownership.
/// Values defined outside and consumed inside are live-ins; values defined
/// inside and consumed by any operation outside are live-outs.
static void deriveStageLiveValues(StagePartition &partition) {
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      llvm::DenseSet<Operation *> owned;
      for (Operation *root : stage.operations)
        collectOwnedOperationTree(root, owned);

      llvm::SetVector<Value> liveIns;
      llvm::SetVector<Value> liveOuts;
      for (Operation *operation : owned) {
        for (Value operand : operation->getOperands())
          if (!isValueDefinedInside(operand, owned))
            liveIns.insert(operand);
        for (Value result : operation->getResults())
          if (llvm::any_of(result.getUsers(), [&](Operation *user) {
                return !owned.contains(user);
              }))
            liveOuts.insert(result);
      }
      stage.liveIns.assign(liveIns.begin(), liveIns.end());
      stage.liveOuts.assign(liveOuts.begin(), liveOuts.end());
      stage.liveInBytes = staticTensorBytes(stage.liveIns);
      stage.liveOutBytes = staticTensorBytes(stage.liveOuts);
    }
  }
}

/// Derive the physical tensor traffic of the exact local SIMT scopes that the
/// materializer will create.  Stage live values are intentionally not used:
/// a Stage can own SIMD operations around a much smaller local scope, and
/// charging its complete live-out footprint would invent UB traffic.
static void deriveLocalSimtScopeTraffic(StagePartition &partition,
                                        const SimtAnchorPlan &anchorPlan) {
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      stage.localSimtScopeCount = 0;
      stage.scopeInputTensorBytes = 0;
      stage.scopeOutputTensorBytes = 0;
      llvm::DenseSet<Operation *> coveredByRange;

      for (unsigned anchorIndex : stage.simtAnchorIndices) {
        if (anchorIndex >= anchorPlan.anchors.size())
          continue;
        const SimtAnchorDescriptor &anchor = anchorPlan.anchors[anchorIndex];
        if (!anchor.materializable || !anchor.operation ||
            coveredByRange.contains(anchor.operation))
          continue;

        llvm::SmallVector<Operation *> roots;
        const bool isRange = anchor.scopeOperations.size() > 1;
        if (isRange) {
          llvm::append_range(roots, anchor.scopeOperations);
          for (Operation *operation : roots)
            coveredByRange.insert(operation);
        } else {
          roots.push_back(anchor.operation);
        }

        llvm::DenseSet<Operation *> inside;
        for (Operation *root : roots) {
          if (!root)
            continue;
          inside.insert(root);
          root->walk([&](Operation *nested) { inside.insert(nested); });
        }

        llvm::SetVector<Value> captured;
        for (Operation *operation : inside)
          for (Value operand : operation->getOperands())
            if (!isValueDefinedInside(operand, inside))
              captured.insert(operand);

        llvm::SetVector<Value> returned;
        for (Operation *root : roots) {
          if (!root)
            continue;
          for (Value result : root->getResults()) {
            // A single-op scope returns every result.  A range scope mirrors
            // wrapAnchorRange and returns only values with an outside user.
            if (!isRange ||
                llvm::any_of(result.getUsers(), [&](Operation *user) {
                  return !inside.contains(user);
                }))
              returned.insert(result);
          }
        }

        ++stage.localSimtScopeCount;
        stage.scopeInputTensorBytes +=
            staticTensorBytes(captured.getArrayRef());
        stage.scopeOutputTensorBytes +=
            staticTensorBytes(returned.getArrayRef());
      }
    }
  }
}

static double totalOperationElements(const StageWorkload &work) {
  double result = 0.0;
  for (const auto &entry : work.operationElements)
    result += entry.second;
  return result;
}

static StageWorkload accumulatePartition(const StagePartition &partition) {
  StageWorkload total;
  for (const LogicalPhase &phase : partition.phases) {
    for (const LogicalStage &stage : phase.stages) {
      StageWorkload work = stage.workload;
      const double count =
          static_cast<double>(std::max<int64_t>(1, stage.iterationCount));
      work.scalarOperations *= count;
      work.loadBytes *= count;
      work.storeBytes *= count;
      work.loadWarpInstructions *= count;
      work.storeWarpInstructions *= count;
      work.predicateElements *= count;
      work.shuffleLaneSteps *= count;
      work.dotFlops *= count;
      work.estimatedSpillTransactions *= count;
      for (auto &entry : work.operationElements)
        entry.second *= count;
      mergeWorkload(total, std::move(work));
    }
  }
  return total;
}

static bool near(double lhs, double rhs) {
  return std::abs(lhs - rhs) <= 1.0e-6 * std::max({1.0, lhs, rhs});
}

} // namespace

StageWorkload
mlir::ascend::buildKernelStageWorkload(const SimdSimtFeatureSummary &features) {
  SimtAnchorFeatureSummary kernel;
  kernel.opElements = features.opElements;
  kernel.weightedOps = features.weightedOps;
  kernel.loadBytes = features.loadBytes;
  kernel.storeBytes = features.storeBytes;
  kernel.loadWarpInstructions = features.loadWarpInstructions;
  kernel.storeWarpInstructions = features.storeWarpInstructions;
  kernel.predicateElements = features.predicateElements;
  kernel.predicateLaneEvaluations = features.predicateLaneEvaluations;
  kernel.shuffleLaneSteps = features.shuffleLaneSteps;
  kernel.dotFlops = features.dotFlops;
  StageWorkload work = buildWorkload(kernel);
  work.scalarOperations = static_cast<double>(features.scalarOps);
  recomputeIssueElements(work);
  return work;
}

llvm::Expected<ProgramStructure>
ProgramStructureAnalysis::analyze(ModuleOp module,
                                  const SimtAnchorPlan &anchorPlan) const {
  if (!module)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ProgramStructureAnalysis requires ModuleOp");
  ProgramStructure structure;
  structure.rootOperations = collectTopLevelSemanticRoots(module);
  if (structure.rootOperations.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ProgramStructureAnalysis found no top-level TTIR operation");

  // Compound scopes may legally move pure tensor setup across input loads.
  // Stage boundaries must describe the program that the selected route will
  // materialize, rather than treating the pre-materialization textual order
  // as immutable.  Normalize each compound anchor to its planned insertion
  // point before PhaseBoundaryAnalysis performs a serial cut.  The operation
  // objects are not mutated here; only the analysis view is reordered.
  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    if (!anchor.materializable || anchor.scopeOperations.size() < 2 ||
        !anchor.scopeInsertionPoint)
      continue;

    llvm::SmallVector<Operation *, 8> scopeRoots;
    for (Operation *operation : anchor.scopeOperations) {
      Operation *root = getTopLevelSemanticRoot(operation);
      if (root && llvm::is_contained(structure.rootOperations, root) &&
          !llvm::is_contained(scopeRoots, root))
        scopeRoots.push_back(root);
    }
    Operation *insertionRoot =
        getTopLevelSemanticRoot(anchor.scopeInsertionPoint);
    auto insertionIt = llvm::find(structure.rootOperations, insertionRoot);
    if (scopeRoots.empty() || insertionIt == structure.rootOperations.end())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ProgramStructureAnalysis cannot normalize a compound SIMT scope");

    const size_t insertionPosition =
        static_cast<size_t>(insertionIt - structure.rootOperations.begin());
    std::vector<Operation *> reordered;
    reordered.reserve(structure.rootOperations.size());
    size_t normalizedInsertionPosition = 0;
    for (auto indexedRoot : llvm::enumerate(structure.rootOperations)) {
      if (indexedRoot.index() < insertionPosition &&
          !llvm::is_contained(scopeRoots, indexedRoot.value()))
        ++normalizedInsertionPosition;
      if (!llvm::is_contained(scopeRoots, indexedRoot.value()))
        reordered.push_back(indexedRoot.value());
    }
    reordered.insert(reordered.begin() + normalizedInsertionPosition,
                     scopeRoots.begin(), scopeRoots.end());
    structure.rootOperations = std::move(reordered);
  }

  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    if (!anchor.materializable)
      continue;
    auto addRoot = [&](Operation *operation) {
      Operation *root = getTopLevelSemanticRoot(operation);
      if (root && llvm::is_contained(structure.rootOperations, root) &&
          !llvm::is_contained(structure.localSimtAnchorRoots, root))
        structure.localSimtAnchorRoots.push_back(root);
    };
    if (anchor.scopeOperations.empty())
      addRoot(anchor.operation);
    else
      for (Operation *operation : anchor.scopeOperations)
        addRoot(operation);
  }
  return structure;
}

llvm::Expected<std::optional<PhaseBoundaryPlan>>
PhaseBoundaryAnalysis::analyze(const SimdSimtFeatureSummary &features,
                               const StagePartitionerOptions &options) const {
  if (features.simtAnchors.triangularSolves.size() == 1 &&
      features.simtAnchors.count > 0) {
    PhaseBoundaryPlan plan{PhaseBoundaryDomain::TriangularRecurrence,
                           "triangular_recurrence",
                           features.simtAnchors.triangularSolves.front()};
    return std::optional<PhaseBoundaryPlan>{std::move(plan)};
  }
  if (features.dotOps == 0 && features.reduceOps > 0 &&
      features.loadedIndexDependentMemoryOps > 0 && features.loadOps > 0 &&
      features.storeOps > 0) {
    PhaseBoundaryPlan plan{PhaseBoundaryDomain::LoadedIndexRowwiseReduction,
                           "loaded_index_rowwise_reduction", std::nullopt};
    return std::optional<PhaseBoundaryPlan>{std::move(plan)};
  }
  if (features.dotOps > 0 && features.reduceOps == 0 &&
      features.loadedIndexDependentMemoryOps > 0 &&
      features.dotFlops <= options.tinyDotFlopsMax && features.loadOps > 0 &&
      features.storeOps > 0) {
    PhaseBoundaryPlan plan{PhaseBoundaryDomain::IndirectUnderfilledDot,
                           "indirect_underfilled_dot", std::nullopt};
    return std::optional<PhaseBoundaryPlan>{std::move(plan)};
  }
  return std::optional<PhaseBoundaryPlan>{};
}

llvm::Expected<std::optional<PhaseBoundaryPlan>>
PhaseBoundaryAnalysis::analyze(ModuleOp module,
                               const SimtAnchorPlan &anchorPlan,
                               const SimdSimtFeatureSummary &features,
                               const StagePartitionerOptions &options) const {
  auto plan = analyze(features, options);
  if (!plan)
    return plan.takeError();
  if (!*plan)
    return plan;
  auto structure = ProgramStructureAnalysis().analyze(module, anchorPlan);
  if (!structure)
    return structure.takeError();
  (*plan)->rootOperations = std::move(structure->rootOperations);
  (*plan)->localSimtAnchorRoots = std::move(structure->localSimtAnchorRoots);
  if (llvm::Error error = assignRootPhaseIds(**plan))
    return std::move(error);
  return plan;
}

llvm::Expected<StagePartition>
StageBoundaryAnalysis::analyze(const PhaseBoundaryPlan &phasePlan,
                               const SimdSimtFeatureSummary &features,
                               const SimtAnchorPlan *anchorPlan) const {
  StagePartition partition;
  switch (phasePlan.domain) {
  case PhaseBoundaryDomain::TriangularRecurrence:
    if (!phasePlan.triangularSolve)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "triangular PhaseBoundaryPlan has no recurrence facts");
    partition = partitionTriangular(features, *phasePlan.triangularSolve,
                                    phasePlan.hasOperationGraph() ? &phasePlan
                                                                  : nullptr);
    break;
  case PhaseBoundaryDomain::LoadedIndexRowwiseReduction:
    partition = partitionRowwise(
        features, phasePlan.hasOperationGraph() ? &phasePlan : nullptr);
    break;
  case PhaseBoundaryDomain::IndirectUnderfilledDot:
    partition = partitionIndirectDot(
        features, phasePlan.hasOperationGraph() ? &phasePlan : nullptr);
    break;
  }
  partition.domain = phasePlan.domainName;
  if (phasePlan.hasOperationGraph()) {
    if (llvm::Error error =
            attachCompleteOperationOwnership(partition, phasePlan))
      return std::move(error);
    if (anchorPlan)
      attachExactAnchorOwnership(partition, *anchorPlan);
    deriveStageLiveValues(partition);
    if (anchorPlan)
      deriveLocalSimtScopeTraffic(partition, *anchorPlan);
  } else if (anchorPlan) {
    attachAnchorOperationOwnership(partition, *anchorPlan);
  }
  return partition;
}

llvm::Error StageFeatureAnalysis::analyze(StagePartition &partition) const {
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      StageModelFeatures &facts = stage.features;
      if (partition.operationOwnershipComplete) {
        const double activeLaneRatio = facts.activeLaneRatio;
        facts = StageModelFeatures{};
        facts.activeLaneRatio = activeLaneRatio;
        llvm::DenseSet<Operation *> owned;
        for (Operation *root : stage.operations)
          collectOwnedOperationTree(root, owned);
        bool hasMemory = false;
        int64_t algorithmLoopCount = 0;
        for (Operation *operation : owned) {
          const llvm::StringRef name = operation->getName().getStringRef();
          if (name == "scf.for" || name == "scf.while") {
            facts.hasLoop = true;
            ++facts.loopBackedgeCount;
            if (!operation->hasAttr("ta.auto_blockify_v1.loop"))
              ++algorithmLoopCount;
            if (!operation->hasAttr("ta.auto_blockify_v1.loop") &&
                operation->getNumRegions() > 0 &&
                !operation->getRegion(0).empty()) {
              Block &body = operation->getRegion(0).front();
              const unsigned firstCarriedArgument = name == "scf.for" ? 1 : 0;
              for (unsigned argumentIndex = firstCarriedArgument;
                   argumentIndex < body.getNumArguments(); ++argumentIndex) {
                BlockArgument argument = body.getArgument(argumentIndex);
                if (argument.use_empty())
                  continue;
                if (isPointerLikeType(argument.getType()) ||
                    isAffineIndexInduction(argument) ||
                    isAddressOnlyLoopValue(argument))
                  facts.hasPointerInduction = true;
                else
                  facts.hasLoopCarriedDataDependency = true;
              }
              // The scf.for induction variable itself commonly feeds pointer
              // arithmetic.  Record that work without treating it as a data
              // recurrence.
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
            facts.hasIndirectMemory |=
                isLoadedIndexDependentMemoryOp(operation) ||
                name == "tt.gather" || name.starts_with("tt.atomic");
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
        // Counts consumed by StageCostEvaluator are per logical iteration.
        // Multiple sibling recurrence loops are flattened into one Stage
        // iteration space, so identical control events must be normalized by
        // the number of sibling loops rather than charged once per loop on
        // every flattened iteration.
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
        facts.source =
            "exact post-layout/post-AutoBlockify-V1 TTIR operation graph";
        if (!facts.isValid())
          return llvm::createStringError(std::errc::invalid_argument,
                                         "Stage '%s' has invalid features",
                                         stage.id.c_str());
        continue;
      }
      facts.hasLoop =
          stage.iterationCount > 1 ||
          stage.costModelKind == StageCostModelKind::AutoBlockifyLoop ||
          stage.costModelKind == StageCostModelKind::IndependentPipelinedLoop ||
          stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence;
      facts.loopBackedgeCount = facts.hasLoop ? 1 : 0;
      facts.hasLoopCarriedDataDependency =
          stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence;
      facts.hasPointerInduction =
          facts.hasLoop && !facts.hasLoopCarriedDataDependency;
      facts.hasContiguousMemory =
          llvm::is_contained({StageCostModelKind::ContinuousTileMemory,
                              StageCostModelKind::ContinuousTileStore,
                              StageCostModelKind::ContinuousShortLoad,
                              StageCostModelKind::CachePolicyStore},
                             stage.costModelKind);
      facts.hasIndirectMemory =
          llvm::is_contained({StageCostModelKind::IndirectScalarMemory,
                              StageCostModelKind::IndirectGatherMemory},
                             stage.costModelKind);
      facts.hasReduction =
          llvm::is_contained({StageCostModelKind::LoopCarriedRecurrence,
                              StageCostModelKind::RowwiseReduction},
                             stage.costModelKind);
      facts.hasDot = llvm::is_contained({StageCostModelKind::CubeRoofline,
                                         StageCostModelKind::TinyCubeRoofline},
                                        stage.costModelKind);
      facts.hasConversionPack =
          stage.costModelKind == StageCostModelKind::ConversionPack;
      facts.source = "feature-summary fallback structural facts";
      if (!facts.isValid())
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' has invalid features",
                                       stage.id.c_str());
    }
  }
  return llvm::Error::success();
}

llvm::Error StageKindClassifier::analyze(StagePartition &partition,
                                         int64_t tinyDotFlopsMax) const {
  if (!partition.operationOwnershipComplete)
    return llvm::Error::success();
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      const StageModelFeatures &facts = stage.features;
      // Auxiliary scalar/index/predicate and memory work may live in a
      // specialized Stage, but two independent dominant formulas may not.
      // Reaching this check means StageBoundaryAnalysis missed a structural
      // cut.  Silently selecting the first matching Kind would hide or
      // double-count work, so report a stable boundary diagnostic instead.
      // Conversion operations are not a split condition by themselves:
      // predicate-to-float and accumulator casts are often auxiliary work of
      // a recurrence/reduction/dot Stage.  ConversionPack becomes dominant
      // only when no stronger structure owns the Stage (see deriveKind()).
      const bool requiresSplit =
          facts.hasDot && (facts.hasReduction || facts.hasIndirectMemory ||
                           facts.hasLoopCarriedDataDependency);
      if (requiresSplit)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "requires_split: Stage '%s' owns incompatible dominant "
            "structures (carried=%d, indirect=%d, reduction=%d, dot=%d, "
            "conversion=%d, roots=%zu)",
            stage.id.c_str(), facts.hasLoopCarriedDataDependency,
            facts.hasIndirectMemory, facts.hasReduction, facts.hasDot,
            facts.hasConversionPack, stage.operations.size());
      auto kindMatchesFacts = [&](StageCostModelKind kind) {
        switch (kind) {
        case StageCostModelKind::AutoBlockifyDispatch:
        case StageCostModelKind::AutoBlockifyLoop:
        case StageCostModelKind::ScalarIssue:
        case StageCostModelKind::ScalarControl:
        case StageCostModelKind::ScalarMath:
        case StageCostModelKind::IndexGeneration:
        case StageCostModelKind::PredicateMask:
        case StageCostModelKind::LoopPredicate:
          return true;
        case StageCostModelKind::LoopCarriedRecurrence:
          return facts.hasLoopCarriedDataDependency;
        case StageCostModelKind::IndependentPipelinedLoop:
          return facts.hasLoop && !facts.hasLoopCarriedDataDependency;
        case StageCostModelKind::RowwiseReduction:
          return facts.hasReduction;
        case StageCostModelKind::CubeRoofline:
        case StageCostModelKind::TinyCubeRoofline:
          return facts.hasDot;
        case StageCostModelKind::IndirectScalarMemory:
        case StageCostModelKind::IndirectGatherMemory:
          return facts.hasIndirectMemory;
        case StageCostModelKind::ContinuousTileMemory:
        case StageCostModelKind::ContinuousTileStore:
        case StageCostModelKind::ContinuousShortLoad:
        case StageCostModelKind::CachePolicyStore:
          return facts.hasContiguousMemory;
        case StageCostModelKind::ConversionPack:
          return facts.hasConversionPack;
        }
        return false;
      };
      auto deriveKind = [&]() {
        // Preserve semantic specializations when their defining evidence is
        // present.  Otherwise classify from the exact Stage operation graph;
        // no workload name or experiment identity participates here.
        if (facts.hasDot)
          return stage.workload.dotFlops * stage.iterationCount <=
                         static_cast<double>(
                             std::max<int64_t>(1, tinyDotFlopsMax))
                     ? StageCostModelKind::TinyCubeRoofline
                     : StageCostModelKind::CubeRoofline;
        if (facts.hasReduction)
          return StageCostModelKind::RowwiseReduction;
        if (facts.hasConversionPack)
          return StageCostModelKind::ConversionPack;
        if (facts.hasLoop)
          return facts.hasLoopCarriedDataDependency
                     ? StageCostModelKind::LoopCarriedRecurrence
                     : StageCostModelKind::IndependentPipelinedLoop;
        if (facts.hasIndirectMemory)
          return StageCostModelKind::IndirectGatherMemory;
        if (facts.hasContiguousMemory)
          return stage.workload.storeBytes > 0.0 &&
                         stage.workload.loadBytes == 0.0
                     ? StageCostModelKind::ContinuousTileStore
                     : StageCostModelKind::ContinuousTileMemory;
        return StageCostModelKind::ScalarIssue;
      };
      if (!kindMatchesFacts(stage.costModelKind)) {
        stage.costModelKind = deriveKind();
        if (stage.costModelKind == StageCostModelKind::IndependentPipelinedLoop)
          stage.scheduleKind = StageScheduleKind::IndependentPipelined;
        else if (stage.costModelKind ==
                 StageCostModelKind::LoopCarriedRecurrence)
          stage.scheduleKind = StageScheduleKind::LoopCarriedSerial;
      }
      const StageCostModelKind kind = stage.costModelKind;
      auto mismatch = [&]() {
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Stage '%s' operation graph does not match StageCostModelKind "
            "'%s' (loop=%d, carried=%d, contiguous=%d, indirect=%d, "
            "reduction=%d, dot=%d, conversion=%d, roots=%zu)",
            stage.id.c_str(), stringifyStageCostModel(kind).str().c_str(),
            facts.hasLoop, facts.hasLoopCarriedDataDependency,
            facts.hasContiguousMemory, facts.hasIndirectMemory,
            facts.hasReduction, facts.hasDot, facts.hasConversionPack,
            stage.operations.size());
      };
      switch (kind) {
      case StageCostModelKind::AutoBlockifyDispatch:
      case StageCostModelKind::AutoBlockifyLoop:
        break;
      case StageCostModelKind::LoopCarriedRecurrence:
        if (!facts.hasLoopCarriedDataDependency)
          return mismatch();
        break;
      case StageCostModelKind::IndependentPipelinedLoop:
        if (!facts.hasLoop || facts.hasLoopCarriedDataDependency)
          return mismatch();
        break;
      case StageCostModelKind::RowwiseReduction:
        if (!facts.hasReduction)
          return mismatch();
        break;
      case StageCostModelKind::CubeRoofline:
        if (!facts.hasDot)
          return mismatch();
        break;
      case StageCostModelKind::TinyCubeRoofline:
        if (!facts.hasDot ||
            stage.workload.dotFlops * stage.iterationCount >
                static_cast<double>(std::max<int64_t>(1, tinyDotFlopsMax)))
          return mismatch();
        break;
      case StageCostModelKind::IndirectScalarMemory:
      case StageCostModelKind::IndirectGatherMemory:
        if (!facts.hasIndirectMemory)
          return mismatch();
        break;
      case StageCostModelKind::ContinuousTileMemory:
      case StageCostModelKind::ContinuousTileStore:
      case StageCostModelKind::ContinuousShortLoad:
      case StageCostModelKind::CachePolicyStore:
        if (!facts.hasContiguousMemory)
          return mismatch();
        break;
      case StageCostModelKind::ConversionPack:
        if (!facts.hasConversionPack)
          return mismatch();
        break;
      case StageCostModelKind::ScalarIssue:
      case StageCostModelKind::ScalarControl:
      case StageCostModelKind::ScalarMath:
      case StageCostModelKind::IndexGeneration:
      case StageCostModelKind::PredicateMask:
      case StageCostModelKind::LoopPredicate:
        // These kinds may legitimately contain auxiliary address, predicate,
        // control, or short metadata memory work.  Their dominant semantics
        // is established by the contiguous boundary plan.
        break;
      }
    }
  }
  return llvm::Error::success();
}

llvm::Error
StageWorkloadAnalysis::verify(const StagePartition &partition,
                              const StageWorkload &kernelWorkload) const {
  if (!kernelWorkload.isFiniteAndNonNegative())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "kernel StageWorkload is invalid");
  const StageWorkload owned = accumulatePartition(partition);
  if (!near(owned.scalarOperations, kernelWorkload.scalarOperations) ||
      !near(owned.loadBytes, kernelWorkload.loadBytes) ||
      !near(owned.storeBytes, kernelWorkload.storeBytes) ||
      !near(owned.predicateElements, kernelWorkload.predicateElements) ||
      !near(owned.shuffleLaneSteps, kernelWorkload.shuffleLaneSteps) ||
      !near(owned.dotFlops, kernelWorkload.dotFlops) ||
      !near(totalOperationElements(owned),
            totalOperationElements(kernelWorkload)))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StagePartition does not conserve post-transform TTIR workload");
  return llvm::Error::success();
}

llvm::Error StageWorkloadAnalysis::analyze(StagePartition &partition) const {
  if (!partition.operationOwnershipComplete)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageWorkloadAnalysis requires complete operation ownership");
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      StageWorkload work;
      work.paysKernelSetup = stage.workload.paysKernelSetup;
      const int64_t loopCount = countAlgorithmLoops(stage);
      const int64_t fallbackLoopTripCount =
          loopCount > 0 ? std::max<int64_t>(1, stage.iterationCount / loopCount)
                        : 1;
      for (Operation *root : stage.operations)
        accumulateDynamicOperationTree(root, work, 1.0, fallbackLoopTripCount);
      recomputeIssueElements(work);
      stage.workload = std::move(work);
      makePerIteration(stage);
      if (!stage.workload.isFiniteAndNonNegative())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Stage '%s' has invalid operation-derived workload",
            stage.id.c_str());
    }
  }
  return llvm::Error::success();
}

llvm::Error
StagePartitionVerifier::verify(const StagePartition &partition,
                               const StageWorkload &kernelWorkload) const {
  if (partition.phases.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "StagePartition has no Phase");
  llvm::StringSet<> phaseIds;
  llvm::StringSet<> stageIds;
  llvm::DenseSet<Operation *> ownedOperations;
  llvm::DenseSet<unsigned> ownedAnchors;
  for (const LogicalPhase &phase : partition.phases) {
    if (phase.id.empty() || !phaseIds.insert(phase.id).second)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "StagePartition has duplicate Phase id");
    if (phase.stages.empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Phase '%s' has no Stage",
                                     phase.id.c_str());
    for (const LogicalStage &stage : phase.stages) {
      if (stage.id.empty() || !stageIds.insert(stage.id).second)
        return llvm::createStringError(std::errc::invalid_argument,
                                       "StagePartition has duplicate Stage id");
      if (stage.iterationCount < 1)
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' has invalid iteration count",
                                       stage.id.c_str());
      if (stage.localSimtMaterializable &&
          partition.operationOwnershipComplete && stage.operations.empty())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "materializable Stage '%s' has no operation ownership",
            stage.id.c_str());
      if (stage.localSimtMaterializable &&
          partition.operationOwnershipComplete &&
          stage.simtAnchorIndices.empty())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "materializable Stage '%s' has no exact SIMT anchor ownership",
            stage.id.c_str());
      if (partition.operationOwnershipComplete)
        for (unsigned index : stage.simtAnchorIndices)
          if (!ownedAnchors.insert(index).second)
            return llvm::createStringError(
                std::errc::invalid_argument,
                "StagePartition SIMT anchor ownership overlaps");
      if (partition.operationOwnershipComplete)
        for (Operation *operation : stage.operations)
          if (!operation || !ownedOperations.insert(operation).second)
            return llvm::createStringError(
                std::errc::invalid_argument,
                "StagePartition operation ownership overlaps");
    }
  }
  if (partition.operationOwnershipComplete &&
      static_cast<int64_t>(ownedOperations.size()) !=
          partition.modeledOperationCount)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StagePartition operation ownership is incomplete");
  if (partition.operationOwnershipComplete)
    return llvm::Error::success();
  return StageWorkloadAnalysis().verify(partition, kernelWorkload);
}

llvm::Error
StageModeLegalityAnalysis::analyze(StagePartition &partition,
                                   int64_t maximumSuperblockFactor,
                                   bool scopeSuperblockMaterializable) const {
  const int64_t maximum = std::clamp<int64_t>(maximumSuperblockFactor, 1, 4);
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      stage.simdLegal = true;
      stage.simtLegal = true;
      stage.legalSimtFactors = {1};
      // A pure-SIMT SuperBlock factor is a whole-kernel schedule, not a
      // recurrence-only annotation.  Every SIMT Stage must therefore expose
      // the same factor candidates; KernelRouteSolver keeps the chosen factor
      // uniform.  Local mixed scopes stay restricted by localSimtFactors
      // (F1 unless Scope SuperBlock materialization is explicitly available).
      if (maximum >= 2)
        stage.legalSimtFactors.push_back(2);
      if (maximum >= 4)
        stage.legalSimtFactors.push_back(4);
      if (stage.localSimtMaterializable) {
        // A mixed F2/F4 implementation is legal only when backend integration
        // can wrap the complete logical-program body with AutoBlockify V1.
        // The local scope itself remains a single-mode Stage; factor changes
        // its implementation, not the Stage boundary.
        stage.localSimtFactors = scopeSuperblockMaterializable
                                     ? stage.legalSimtFactors
                                     : std::vector<int64_t>{1};
      }
      if (stage.localSimtMaterializable &&
          (stage.localSimtFactors.empty() ||
           llvm::any_of(stage.localSimtFactors, [&](int64_t factor) {
             return !llvm::is_contained(stage.legalSimtFactors, factor);
           })))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "local SIMT factors are invalid for Stage '%s'", stage.id.c_str());
    }
  }
  return llvm::Error::success();
}

llvm::Expected<std::optional<StagePartition>>
StagePartitioner::partition(const SimdSimtFeatureSummary &features,
                            const StagePartitionerOptions &options) const {
  auto phasePlan = PhaseBoundaryAnalysis().analyze(features, options);
  if (!phasePlan)
    return phasePlan.takeError();
  if (!*phasePlan)
    return std::optional<StagePartition>{};
  auto result = StageBoundaryAnalysis().analyze(**phasePlan, features);
  if (!result)
    return result.takeError();

  StageFeatureAnalysis featureAnalysis;
  if (llvm::Error error = featureAnalysis.analyze(*result))
    return std::move(error);
  if (llvm::Error error =
          StageKindClassifier().analyze(*result, options.tinyDotFlopsMax))
    return std::move(error);
  StageModeLegalityAnalysis legalityAnalysis;
  if (llvm::Error error =
          legalityAnalysis.analyze(*result, options.maximumSuperblockFactor,
                                   options.scopeSuperblockMaterializable))
    return std::move(error);
  if (llvm::Error error = StagePartitionVerifier().verify(
          *result, buildKernelStageWorkload(features)))
    return std::move(error);
  return std::optional<StagePartition>{std::move(*result)};
}

llvm::Expected<std::optional<StagePartition>>
StagePartitioner::partition(ModuleOp module, const SimtAnchorPlan &anchorPlan,
                            const SimdSimtFeatureSummary &features,
                            const StagePartitionerOptions &options) const {
  auto phasePlan =
      PhaseBoundaryAnalysis().analyze(module, anchorPlan, features, options);
  if (!phasePlan)
    return phasePlan.takeError();
  if (!*phasePlan)
    return std::optional<StagePartition>{};
  auto result =
      StageBoundaryAnalysis().analyze(**phasePlan, features, &anchorPlan);
  if (!result)
    return result.takeError();
  StageWorkloadAnalysis workloadAnalysis;
  if (llvm::Error error = workloadAnalysis.analyze(*result))
    return std::move(error);
  StageFeatureAnalysis featureAnalysis;
  if (llvm::Error error = featureAnalysis.analyze(*result))
    return std::move(error);
  if (llvm::Error error =
          StageKindClassifier().analyze(*result, options.tinyDotFlopsMax))
    return std::move(error);
  StageModeLegalityAnalysis legalityAnalysis;
  if (llvm::Error error =
          legalityAnalysis.analyze(*result, options.maximumSuperblockFactor,
                                   options.scopeSuperblockMaterializable))
    return std::move(error);
  if (llvm::Error error = StagePartitionVerifier().verify(
          *result, buildKernelStageWorkload(features)))
    return std::move(error);
  return std::optional<StagePartition>{std::move(*result)};
}

llvm::Expected<std::optional<StagePartition>>
StagePartitioner::partition(const SimdSimtFeatureSummary &features,
                            const StagePartitionerOptions &options,
                            const SimtAnchorPlan &anchorPlan) const {
  auto phasePlan = PhaseBoundaryAnalysis().analyze(features, options);
  if (!phasePlan)
    return phasePlan.takeError();
  if (!*phasePlan)
    return std::optional<StagePartition>{};
  auto result =
      StageBoundaryAnalysis().analyze(**phasePlan, features, &anchorPlan);
  if (!result)
    return result.takeError();
  StageFeatureAnalysis featureAnalysis;
  if (llvm::Error error = featureAnalysis.analyze(*result))
    return std::move(error);
  if (llvm::Error error =
          StageKindClassifier().analyze(*result, options.tinyDotFlopsMax))
    return std::move(error);
  StageModeLegalityAnalysis legalityAnalysis;
  if (llvm::Error error =
          legalityAnalysis.analyze(*result, options.maximumSuperblockFactor,
                                   options.scopeSuperblockMaterializable))
    return std::move(error);
  if (llvm::Error error = StagePartitionVerifier().verify(
          *result, buildKernelStageWorkload(features)))
    return std::move(error);
  return std::optional<StagePartition>{std::move(*result)};
}
