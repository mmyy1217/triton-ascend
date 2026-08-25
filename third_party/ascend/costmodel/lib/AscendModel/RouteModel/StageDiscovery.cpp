//===- StageDiscovery.cpp - Generic Stage boundary discovery ------------===//

#include "AscendModel/RouteModel/StageDiscovery.h"

#include "AscendModel/RouteModel/StagePartitioner.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <limits>
#include <set>
#include <system_error>

using namespace mlir;
using namespace mlir::ascend;

namespace {

static bool isFunctionLike(Operation *operation) {
  if (!operation)
    return false;
  llvm::StringRef name = operation->getName().getStringRef();
  return name == "tt.func" || name == "func.func";
}

static std::vector<Operation *> collectSemanticRoots(ModuleOp module) {
  std::vector<Operation *> roots;
  for (Operation &operation : module.getBody()->getOperations()) {
    if (!isFunctionLike(&operation) || operation.getNumRegions() == 0)
      continue;
    for (Block &block : operation.getRegion(0)) {
      for (Operation &nested : block.getOperations()) {
        if (nested.hasTrait<OpTrait::IsTerminator>())
          continue;
        roots.push_back(&nested);
        if (!nested.hasAttr("ta.auto_blockify_v1.loop") ||
            nested.getNumRegions() == 0)
          continue;
        for (Block &body : nested.getRegion(0))
          for (Operation &bodyOperation : body.getOperations())
            if (!bodyOperation.hasTrait<OpTrait::IsTerminator>())
              roots.push_back(&bodyOperation);
      }
    }
  }
  return roots;
}

static void mapOperationTree(Operation *root, size_t unit,
                             llvm::DenseMap<Operation *, size_t> &owners) {
  owners[root] = unit;
  if (root->hasAttr("ta.auto_blockify_v1.loop"))
    return;
  root->walk([&](Operation *nested) { owners.try_emplace(nested, unit); });
}

static bool containsName(Operation *root, llvm::StringRef expected) {
  bool found = root->getName().getStringRef() == expected;
  root->walk([&](Operation *nested) {
    found |= nested->getName().getStringRef() == expected;
  });
  return found;
}

static bool hasVectorValue(Operation *root) {
  bool found = false;
  root->walk([&](Operation *operation) {
    auto inspect = [&](Value value) {
      auto shaped = dyn_cast<ShapedType>(value.getType());
      found |= shaped && shaped.hasStaticShape() && shaped.getNumElements() > 1;
    };
    for (Value value : operation->getOperands())
      inspect(value);
    for (Value value : operation->getResults())
      inspect(value);
  });
  return found;
}

struct MemorySummary {
  bool reads = false;
  bool writes = false;
  bool unknown = false;

  bool touchesMemory() const { return reads || writes || unknown; }
};

static MemorySummary summarizeMemory(Operation *root) {
  MemorySummary summary;
  auto visit = [&](Operation *operation) {
    llvm::StringRef name = operation->getName().getStringRef();
    if (auto effects = dyn_cast<MemoryEffectOpInterface>(operation)) {
      summary.reads |= effects.hasEffect<MemoryEffects::Read>();
      summary.writes |= effects.hasEffect<MemoryEffects::Write>();
      return;
    }
    if (name == "tt.load" || name == "memref.load")
      summary.reads = true;
    else if (name == "tt.store" || name == "memref.store" ||
             name == "tt.atomic_rmw" || name == "tt.atomic_cas")
      summary.writes = true;
    else if (operation->getNumRegions() == 0 && !isMemoryEffectFree(operation))
      summary.unknown = true;
  };
  visit(root);
  root->walk([&](Operation *nested) {
    if (nested != root)
      visit(nested);
  });
  return summary;
}

static std::string classifyEvidence(Operation *root, bool anchorEvidence) {
  if (anchorEvidence)
    return "anchor";
  if (containsName(root, "tt.load") || containsName(root, "tt.store"))
    return "memory_seed";
  if (containsName(root, "scf.for") || containsName(root, "scf.while"))
    return "loop_seed";
  if (containsName(root, "scf.if"))
    return "control_seed";
  llvm::StringRef name = root->getName().getStringRef();
  if (name.starts_with("arith.") || name.starts_with("tt."))
    return "scalar_or_index_seed";
  return "semantic_unit";
}

static bool isMaterializableRoot(Operation *operation) {
  if (!operation || !operation->getBlock() ||
      operation->hasTrait<OpTrait::IsTerminator>() ||
      operation->hasTrait<OpTrait::IsIsolatedFromAbove>())
    return false;
  llvm::StringRef name = operation->getName().getStringRef();
  return name != "scope.scope" && name != "scope.return" &&
         !isFunctionLike(operation);
}

static bool isGenericLocalSimtLowerable(Operation *root) {
  bool supported = true;
  root->walk([&](Operation *operation) {
    llvm::StringRef name = operation->getName().getStringRef();
    llvm::StringRef dialect = operation->getName().getDialectNamespace();
    if (llvm::is_contained({"tt.dot", "tt.dot_scaled", "tt.reduce", "tt.scan",
                            "tt.sort", "tt.histogram"},
                           name) ||
        !llvm::is_contained({"arith", "math", "tt", "scf"}, dialect))
      supported = false;
  });
  return supported;
}

static bool isContiguousMaterializableRange(ArrayRef<Operation *> operations) {
  if (operations.empty())
    return false;
  Block *block = operations.front()->getBlock();
  if (!block)
    return false;
  for (Operation *operation : operations)
    if (operation->getBlock() != block || !isMaterializableRoot(operation))
      return false;
  auto cursor = operations.front()->getIterator();
  for (Operation *expected : operations) {
    while (cursor != block->end() && cursor->hasTrait<OpTrait::IsTerminator>())
      ++cursor;
    if (cursor == block->end() || &*cursor != expected)
      return false;
    ++cursor;
  }
  return true;
}

static void collectOwnedTree(Operation *root,
                             llvm::DenseSet<Operation *> &owned) {
  owned.insert(root);
  if (root->hasAttr("ta.auto_blockify_v1.loop"))
    return;
  root->walk([&](Operation *nested) { owned.insert(nested); });
}

static bool definedInside(Value value,
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
  int64_t elements = shaped.getNumElements();
  int64_t bits = elementType.getIntOrFloatBitWidth();
  if (elements <= 0 || bits <= 0)
    return 0;
  if (elements > (std::numeric_limits<int64_t>::max() - 7) / bits)
    return std::numeric_limits<int64_t>::max();
  return (elements * bits + 7) / 8;
}

static int64_t sumStaticTensorBytes(ArrayRef<Value> values) {
  int64_t total = 0;
  for (Value value : values) {
    int64_t bytes = staticTensorBytes(value);
    if (bytes > std::numeric_limits<int64_t>::max() - total)
      return std::numeric_limits<int64_t>::max();
    total += bytes;
  }
  return total;
}

static void deriveLiveValues(LogicalStage &stage) {
  llvm::DenseSet<Operation *> owned;
  for (Operation *root : stage.operations)
    collectOwnedTree(root, owned);
  llvm::SetVector<Value> liveIns;
  llvm::SetVector<Value> liveOuts;
  for (Operation *operation : owned) {
    for (Value operand : operation->getOperands())
      if (!definedInside(operand, owned))
        liveIns.insert(operand);
    for (Value result : operation->getResults())
      if (llvm::any_of(result.getUsers(),
                       [&](Operation *user) { return !owned.contains(user); }))
        liveOuts.insert(result);
  }
  stage.liveIns.assign(liveIns.begin(), liveIns.end());
  stage.liveOuts.assign(liveOuts.begin(), liveOuts.end());
  stage.liveInBytes = sumStaticTensorBytes(stage.liveIns);
  stage.liveOutBytes = sumStaticTensorBytes(stage.liveOuts);
}

static bool ownsAnchor(const LogicalStage &stage,
                       const SimtAnchorDescriptor &anchor) {
  auto owns = [&](Operation *operation) {
    if (!operation)
      return false;
    Operation *root = operation;
    while (root->getParentOp() && !isFunctionLike(root->getParentOp()))
      root = root->getParentOp();
    return llvm::is_contained(stage.operations, root);
  };
  if (anchor.scopeOperations.empty())
    return owns(anchor.operation);
  return llvm::all_of(anchor.scopeOperations, owns);
}

static void refineGenericKind(LogicalStage &stage) {
  bool hasMemory = false;
  bool hasVectorMemory = false;
  bool hasStore = false;
  for (Operation *root : stage.operations) {
    root->walk([&](Operation *operation) {
      llvm::StringRef name = operation->getName().getStringRef();
      if (name != "tt.load" && name != "tt.store" && name != "memref.load" &&
          name != "memref.store")
        return;
      hasMemory = true;
      hasStore |= name == "tt.store" || name == "memref.store";
      auto inspectValue = [&](Value value) {
        auto shaped = dyn_cast<ShapedType>(value.getType());
        hasVectorMemory |=
            shaped && shaped.hasStaticShape() && shaped.getNumElements() > 1;
      };
      for (Value value : operation->getOperands())
        inspectValue(value);
      for (Value value : operation->getResults())
        inspectValue(value);
    });
  }
  if (hasVectorMemory) {
    stage.costModelKind = stage.features.hasIndirectMemory
                              ? StageCostModelKind::IndirectGatherMemory
                              : StageCostModelKind::ContinuousTileMemory;
    stage.scheduleKind = StageScheduleKind::StraightLine;
    return;
  }
  if (stage.features.conditionalBranchCount > 0) {
    stage.costModelKind = StageCostModelKind::ScalarControl;
    stage.scheduleKind = StageScheduleKind::StraightLine;
  } else if (hasMemory) {
    stage.costModelKind = hasStore ? StageCostModelKind::ScalarIssue
                                   : StageCostModelKind::ContinuousShortLoad;
  } else {
    stage.costModelKind = StageCostModelKind::IndexGeneration;
  }
}

static llvm::Error analyzeCandidate(LogicalStage &stage,
                                    int64_t tinyDotFlopsMax) {
  StagePartition scratch;
  scratch.domain = "generic_stage_candidate";
  scratch.boundarySource = "stage_boundary_graph";
  scratch.operationOwnershipComplete = true;
  scratch.modeledOperationCount = static_cast<int64_t>(stage.operations.size());
  LogicalPhase phase;
  phase.id = "candidate";
  phase.description = "candidate";
  phase.stages.push_back(stage);
  scratch.phases.push_back(std::move(phase));
  if (llvm::Error error = StageWorkloadAnalysis().analyze(scratch))
    return error;
  if (llvm::Error error = StageFeatureAnalysis().analyze(scratch))
    return error;
  if (llvm::Error error =
          StageKindClassifier().analyze(scratch, tinyDotFlopsMax)) {
    stage = std::move(scratch.phases.front().stages.front());
    return error;
  }
  stage = std::move(scratch.phases.front().stages.front());
  refineGenericKind(stage);
  return llvm::Error::success();
}

static void makeConservativeFallback(LogicalStage &stage) {
  stage.costModelKind = StageCostModelKind::ScalarIssue;
  stage.scheduleKind = StageScheduleKind::StraightLine;
  stage.simdLegal = true;
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.localSimtMaterializable = false;
  stage.localSimtFactors.clear();
}

} // namespace

llvm::StringRef
mlir::ascend::stringifyStageDependenceKind(StageDependenceKind kind) {
  switch (kind) {
  case StageDependenceKind::SSA:
    return "ssa";
  case StageDependenceKind::Memory:
    return "memory";
  case StageDependenceKind::Control:
    return "control";
  case StageDependenceKind::LoopCarried:
    return "loop_carried";
  }
  llvm_unreachable("unknown StageDependenceKind");
}

llvm::json::Object StageDependenceEdge::toJSON() const {
  llvm::json::Object result;
  result["source"] = static_cast<int64_t>(source);
  result["target"] = static_cast<int64_t>(target);
  result["kind"] = stringifyStageDependenceKind(kind);
  result["detail"] = detail;
  return result;
}

llvm::json::Object StageSemanticUnit::toJSON() const {
  llvm::json::Object result;
  result["index"] = static_cast<int64_t>(index);
  result["operation"] = operation ? operation->getName().getStringRef().str()
                                  : std::string("<null>");
  result["anchor_evidence"] = anchorEvidence;
  result["seed_evidence"] = seedEvidence;
  result["evidence"] = evidence;
  return result;
}

bool StageDependenceGraph::canCut(size_t boundary) const {
  return boundary < cuttableBoundaries.size() && cuttableBoundaries[boundary];
}

llvm::json::Object StageDependenceGraph::toJSON() const {
  llvm::json::Object result;
  llvm::json::Array unitArray;
  for (const StageSemanticUnit &unit : units)
    unitArray.push_back(unit.toJSON());
  result["semantic_units"] = std::move(unitArray);
  llvm::json::Array edgeArray;
  for (const StageDependenceEdge &edge : edges)
    edgeArray.push_back(edge.toJSON());
  result["edges"] = std::move(edgeArray);
  llvm::json::Array boundaries;
  for (bool cuttable : cuttableBoundaries)
    boundaries.push_back(cuttable);
  result["cuttable_boundaries"] = std::move(boundaries);
  return result;
}

llvm::json::Object CandidateStage::toJSON() const {
  llvm::json::Object result;
  result["index"] = static_cast<int64_t>(index);
  result["begin_boundary"] = static_cast<int64_t>(beginBoundary);
  result["end_boundary"] = static_cast<int64_t>(endBoundary);
  result["id"] = stage.id;
  result["model"] = stringifyStageCostModel(stage.costModelKind);
  result["fallback"] = fallback;
  result["owned_operation_count"] =
      static_cast<int64_t>(stage.operations.size());
  result["live_in_count"] = static_cast<int64_t>(stage.liveIns.size());
  result["live_out_count"] = static_cast<int64_t>(stage.liveOuts.size());
  result["live_in_bytes"] = stage.liveInBytes;
  result["live_out_bytes"] = stage.liveOutBytes;
  result["features"] = stage.features.toJSON();
  result["workload"] = stage.workload.toJSON();
  result["simd_legal"] = stage.simdLegal;
  result["simt_legal"] = stage.simtLegal;
  result["local_simt_materializable"] = stage.localSimtMaterializable;
  if (!localSimtRejectionReason.empty())
    result["local_simt_rejection_reason"] = localSimtRejectionReason;
  llvm::json::Array evidenceArray;
  for (const std::string &item : evidence)
    evidenceArray.push_back(item);
  result["evidence"] = std::move(evidenceArray);
  return result;
}

llvm::json::Object RejectedCandidateStage::toJSON() const {
  llvm::json::Object result;
  result["begin_boundary"] = static_cast<int64_t>(beginBoundary);
  result["end_boundary"] = static_cast<int64_t>(endBoundary);
  result["reason"] = reason;
  return result;
}

llvm::json::Object StageBoundaryGraph::toJSON() const {
  llvm::json::Object result;
  result["domain"] = domain;
  result["boundary_source"] = boundarySource;
  result["boundary_count"] = static_cast<int64_t>(boundaryCount());
  result["dependence_graph"] = dependenceGraph.toJSON();
  llvm::json::Array candidateArray;
  for (const CandidateStage &candidate : candidates)
    candidateArray.push_back(candidate.toJSON());
  result["candidates"] = std::move(candidateArray);
  llvm::json::Array rejectedArray;
  for (const RejectedCandidateStage &candidate : rejectedCandidates)
    rejectedArray.push_back(candidate.toJSON());
  result["rejected_candidates"] = std::move(rejectedArray);
  return result;
}

llvm::Expected<StageDependenceGraph>
StageDependenceAnalysis::analyze(ModuleOp module,
                                 const SimtAnchorPlan &anchorPlan) const {
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "StageDependenceAnalysis requires ModuleOp");
  StageDependenceGraph graph;
  std::vector<Operation *> roots = collectSemanticRoots(module);
  if (roots.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageDependenceAnalysis found no top-level semantic unit");

  llvm::DenseMap<Operation *, size_t> owners;
  for (auto indexedRoot : llvm::enumerate(roots))
    mapOperationTree(indexedRoot.value(), indexedRoot.index(), owners);

  llvm::DenseSet<Operation *> anchorOperations;
  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    if (anchor.scopeOperations.empty())
      anchorOperations.insert(anchor.operation);
    else
      anchorOperations.insert(anchor.scopeOperations.begin(),
                              anchor.scopeOperations.end());
  }

  graph.cuttableBoundaries.assign(roots.size() + 1, true);
  for (auto indexedRoot : llvm::enumerate(roots)) {
    bool anchorEvidence = false;
    Operation *root = indexedRoot.value();
    root->walk([&](Operation *operation) {
      anchorEvidence |= anchorOperations.contains(operation);
    });
    std::string evidence = classifyEvidence(root, anchorEvidence);
    graph.units.push_back({indexedRoot.index(), root, anchorEvidence,
                           evidence != "semantic_unit", evidence});
  }

  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    llvm::SmallVector<size_t> unitIndices;
    auto appendUnit = [&](Operation *operation) {
      auto owner = owners.find(operation);
      if (owner != owners.end() &&
          !llvm::is_contained(unitIndices, owner->second))
        unitIndices.push_back(owner->second);
    };
    if (anchor.scopeOperations.empty())
      appendUnit(anchor.operation);
    else
      for (Operation *operation : anchor.scopeOperations)
        appendUnit(operation);
    if (unitIndices.size() > 1) {
      auto [minimum, maximum] =
          std::minmax_element(unitIndices.begin(), unitIndices.end());
      for (size_t boundary = *minimum + 1; boundary <= *maximum; ++boundary)
        graph.cuttableBoundaries[boundary] = false;
    }
  }

  llvm::StringSet<> seenEdges;
  auto addEdge = [&](size_t source, size_t target, StageDependenceKind kind,
                     llvm::StringRef detail) {
    std::string key = llvm::formatv("{0}:{1}:{2}", source, target,
                                    static_cast<unsigned>(kind));
    if (!seenEdges.insert(key).second)
      return;
    graph.edges.push_back({source, target, kind, detail.str()});
  };

  for (auto indexedRoot : llvm::enumerate(roots)) {
    Operation *root = indexedRoot.value();
    root->walk([&](Operation *operation) {
      for (Value operand : operation->getOperands()) {
        Operation *definition = operand.getDefiningOp();
        auto producer = owners.find(definition);
        if (producer != owners.end() && producer->second != indexedRoot.index())
          addEdge(producer->second, indexedRoot.index(),
                  StageDependenceKind::SSA, "use_def");
      }
    });
    if (containsName(root, "scf.if"))
      addEdge(indexedRoot.index(), indexedRoot.index(),
              StageDependenceKind::Control, "structured_control");
    if (containsName(root, "scf.for") || containsName(root, "scf.while"))
      addEdge(indexedRoot.index(), indexedRoot.index(),
              StageDependenceKind::LoopCarried, "structured_loop");
  }

  std::vector<MemorySummary> memory;
  memory.reserve(roots.size());
  for (Operation *root : roots)
    memory.push_back(summarizeMemory(root));
  for (size_t source = 0; source < roots.size(); ++source) {
    if (!memory[source].touchesMemory())
      continue;
    for (size_t target = source + 1; target < roots.size(); ++target) {
      if (!memory[target].touchesMemory())
        continue;
      if (memory[source].writes || memory[target].writes ||
          memory[source].unknown || memory[target].unknown)
        addEdge(source, target, StageDependenceKind::Memory,
                "ordered_memory_effect");
    }
  }
  return graph;
}

llvm::Expected<StageBoundaryGraph>
StageDiscovery::discover(ModuleOp module, const SimtAnchorPlan &anchorPlan,
                         const StageDiscoveryOptions &options) const {
  auto dependence = StageDependenceAnalysis().analyze(module, anchorPlan);
  if (!dependence)
    return dependence.takeError();
  StageBoundaryGraph graph;
  graph.dependenceGraph = std::move(*dependence);
  size_t unitCount = graph.dependenceGraph.units.size();
  if (options.maximumCandidateCount == 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "StageDiscovery candidate limit is zero");

  using CandidateInterval = std::pair<size_t, size_t>;
  std::vector<CandidateInterval> intervals;
  std::set<CandidateInterval> seenIntervals;
  std::set<CandidateInterval> fallbackIntervals;
  auto addInterval = [&](size_t begin, size_t end) {
    if (begin >= end || !graph.dependenceGraph.canCut(begin) ||
        !graph.dependenceGraph.canCut(end) ||
        !seenIntervals.insert({begin, end}).second)
      return;
    intervals.push_back({begin, end});
  };

  std::vector<size_t> cutPoints;
  for (size_t boundary = 0; boundary <= unitCount; ++boundary)
    if (graph.dependenceGraph.canCut(boundary))
      cutPoints.push_back(boundary);
  for (size_t index = 1; index < cutPoints.size(); ++index) {
    addInterval(cutPoints[index - 1], cutPoints[index]);
    fallbackIntervals.insert({cutPoints[index - 1], cutPoints[index]});
  }
  addInterval(0, unitCount);
  fallbackIntervals.insert({0, unitCount});

  std::vector<bool> vectorUnits;
  vectorUnits.reserve(unitCount);
  for (const StageSemanticUnit &unit : graph.dependenceGraph.units)
    vectorUnits.push_back(hasVectorValue(unit.operation));
  for (size_t boundary = 1; boundary < unitCount; ++boundary) {
    if (vectorUnits[boundary - 1] == vectorUnits[boundary])
      continue;
    addInterval(0, boundary);
    addInterval(boundary, unitCount);
  }

  for (size_t begin = 0;
       begin < unitCount && intervals.size() < options.maximumCandidateCount;
       ++begin)
    for (size_t end = begin + 1;
         end <= unitCount && intervals.size() < options.maximumCandidateCount;
         ++end)
      addInterval(begin, end);

  for (CandidateInterval interval : intervals) {
    size_t begin = interval.first;
    size_t end = interval.second;
    bool wholeKernel = begin == 0 && end == unitCount;
    bool minimumEdge = fallbackIntervals.count(interval) != 0;

    CandidateStage candidate;
    candidate.beginBoundary = begin;
    candidate.endBoundary = end;
    candidate.fallback = wholeKernel || minimumEdge;
    candidate.stage.id = llvm::formatv("candidate_b{0}_b{1}", begin, end).str();
    candidate.stage.description = "generic contiguous TTIR candidate";
    candidate.stage.iterationCount = 1;
    for (size_t index = begin; index < end; ++index) {
      const StageSemanticUnit &unit = graph.dependenceGraph.units[index];
      candidate.stage.operations.push_back(unit.operation);
      if (unit.seedEvidence || unit.anchorEvidence)
        candidate.evidence.push_back(unit.evidence);
    }
    llvm::sort(candidate.evidence);
    candidate.evidence.erase(
        std::unique(candidate.evidence.begin(), candidate.evidence.end()),
        candidate.evidence.end());
    deriveLiveValues(candidate.stage);

    if (!candidate.fallback && candidate.evidence.empty()) {
      graph.rejectedCandidates.push_back(
          {begin, end, "missing_anchor_or_seed"});
      continue;
    }

    bool conservativeFallback = false;
    if (llvm::Error error =
            analyzeCandidate(candidate.stage, options.tinyDotFlopsMax)) {
      std::string reason = llvm::toString(std::move(error));
      graph.rejectedCandidates.push_back({begin, end, reason});
      if (!candidate.fallback)
        continue;
      makeConservativeFallback(candidate.stage);
      conservativeFallback = true;
      candidate.evidence.push_back("conservative_simd_fallback");
    }

    candidate.stage.simdLegal = true;
    candidate.stage.simtLegal = !conservativeFallback;
    if (candidate.stage.simtLegal) {
      candidate.stage.legalSimtFactors = {1};
      if (wholeKernel && options.maximumSuperblockFactor >= 2)
        candidate.stage.legalSimtFactors.push_back(2);
      if (wholeKernel && options.maximumSuperblockFactor >= 4)
        candidate.stage.legalSimtFactors.push_back(4);
    }

    bool partial = !wholeKernel;
    bool contiguous =
        isContiguousMaterializableRange(candidate.stage.operations);
    bool lowerable =
        llvm::all_of(candidate.stage.operations, isGenericLocalSimtLowerable);
    candidate.stage.localSimtMaterializable =
        partial && contiguous && lowerable;
    if (!partial)
      candidate.localSimtRejectionReason = "whole_kernel_candidate";
    else if (!contiguous)
      candidate.localSimtRejectionReason =
          "non_contiguous_or_structurally_illegal";
    else if (!lowerable)
      candidate.localSimtRejectionReason = "unsupported_generic_simt_operation";
    if (candidate.stage.localSimtMaterializable) {
      candidate.stage.localSimtFactors = {1};
      candidate.stage.localSimtScopeCount = 1;
      candidate.stage.scopeInputTensorBytes = candidate.stage.liveInBytes;
      candidate.stage.scopeOutputTensorBytes = candidate.stage.liveOutBytes;
    }
    for (auto indexedAnchor : llvm::enumerate(anchorPlan.anchors))
      if (indexedAnchor.value().materializable &&
          ownsAnchor(candidate.stage, indexedAnchor.value()))
        candidate.stage.simtAnchorIndices.push_back(
            static_cast<unsigned>(indexedAnchor.index()));

    candidate.index = graph.candidates.size();
    graph.candidates.push_back(std::move(candidate));
  }

  bool hasWholeKernel = llvm::any_of(graph.candidates, [&](const auto &edge) {
    return edge.beginBoundary == 0 && edge.endBoundary == unitCount;
  });
  if (!hasWholeKernel)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageDiscovery failed to preserve whole-kernel fallback");
  return graph;
}

llvm::json::Object SimtStageRange::toJSON() const {
  llvm::json::Object result;
  result["begin_boundary"] = static_cast<int64_t>(beginBoundary);
  result["end_boundary"] = static_cast<int64_t>(endBoundary);
  llvm::json::Array operationsArray;
  for (Operation *operation : operations)
    operationsArray.push_back(operation->getName().getStringRef());
  result["operations"] = std::move(operationsArray);
  return result;
}

llvm::json::Object StageMaterializationPlan::toJSON() const {
  llvm::json::Object result;
  llvm::json::Array rangesArray;
  for (const SimtStageRange &range : ranges)
    rangesArray.push_back(range.toJSON());
  result["simt_scope_ranges"] = std::move(rangesArray);
  return result;
}

llvm::Expected<StageMaterializationPlan>
mlir::ascend::buildStageMaterializationPlan(
    const StageCostModelSummary &stageModel) {
  StageMaterializationPlan plan;
  const StageRoutePlan &route = stageModel.mixed;
  if (!route.legal || route.implementations.size() != route.stageIndices.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageMaterializationPlan requires a complete legal mixed route");

  for (size_t position = 0; position < route.implementations.size();
       ++position) {
    if (route.implementations[position].mode != StageMode::SIMT)
      continue;
    size_t stageIndex = route.stageIndices[position];
    if (stageIndex >= stageModel.stages.size())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "StageMaterializationPlan references an invalid candidate Stage");
    const LogicalStageCost &stage = stageModel.stages[stageIndex];
    if (!stage.localSimtMaterializable || stage.operations.empty() ||
        stage.beginBoundary < 0 || stage.endBoundary <= stage.beginBoundary)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "candidate Stage '%s' cannot become a local SIMT scope",
          stage.id.c_str());

    bool merge = !plan.ranges.empty() &&
                 plan.ranges.back().endBoundary ==
                     static_cast<size_t>(stage.beginBoundary) &&
                 plan.ranges.back().operations.back()->getBlock() ==
                     stage.operations.front()->getBlock() &&
                 plan.ranges.back().operations.back()->getNextNode() ==
                     stage.operations.front();
    if (merge) {
      plan.ranges.back().endBoundary = static_cast<size_t>(stage.endBoundary);
      llvm::append_range(plan.ranges.back().operations, stage.operations);
      continue;
    }
    SimtStageRange range;
    range.beginBoundary = static_cast<size_t>(stage.beginBoundary);
    range.endBoundary = static_cast<size_t>(stage.endBoundary);
    range.operations = stage.operations;
    plan.ranges.push_back(std::move(range));
  }
  if (plan.ranges.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mixed route selected no materializable SIMT Stage");
  return plan;
}
