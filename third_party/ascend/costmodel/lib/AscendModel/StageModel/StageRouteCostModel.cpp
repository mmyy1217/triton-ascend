//===- StageRouteCostModel.cpp - Logical-stage route solver ---------------===//

#include "AscendModel/StageModel/StageRouteCostModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <system_error>

using namespace mlir;
using namespace mlir::ascend;

namespace {

enum class RouteClass : unsigned {
  AllSIMD = 0,
  AllSIMT = 1,
  Mixed = 2,
  AllSIMTForMixed = 3
};

struct PartialRoute {
  double totalCycles = 0.0;
  StageMode exitMode = StageMode::SIMD;
  RouteClass routeClass = RouteClass::AllSIMD;
  std::vector<StageImplementation> implementations;
  std::vector<size_t> stageIndices;
  std::vector<double> entryTransitionCycles;
  std::vector<double> logicalStageCycles;
  std::vector<ScopeRunCost> scopeRuns;
  std::optional<size_t> openScopeStagePosition;
  int64_t routeSuperblockFactor = 1;
};

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

static int64_t sumTensorBytes(ArrayRef<Value> values) {
  int64_t total = 0;
  for (Value value : values) {
    int64_t bytes = staticTensorBytes(value);
    if (bytes > std::numeric_limits<int64_t>::max() - total)
      return std::numeric_limits<int64_t>::max();
    total += bytes;
  }
  return total;
}

static void collectOwnedTree(Operation *root,
                             llvm::DenseSet<Operation *> &owned) {
  if (root)
    root->walk([&](Operation *operation) { owned.insert(operation); });
}

static bool definedInside(Value value,
                          const llvm::DenseSet<Operation *> &owned) {
  if (Operation *definition = value.getDefiningOp())
    return owned.contains(definition);
  auto argument = dyn_cast<BlockArgument>(value);
  Operation *parent = argument ? argument.getOwner()->getParentOp() : nullptr;
  return parent && owned.contains(parent);
}

static bool isMaterializableRoot(Operation *operation) {
  return operation && operation->getBlock() && !isa<ModuleOp>(operation) &&
         !operation->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         !operation->hasTrait<OpTrait::IsTerminator>() &&
         operation->getName().getStringRef() != "scope.scope" &&
         operation->getName().getStringRef() != "scope.return";
}

static ScopeRunCost buildScopeRun(const PartialRoute &route,
                                  const StageCostTable &costTable,
                                  const StageTransitionCost &transition,
                                  bool conservative) {
  ScopeRunCost run;
  if (!route.openScopeStagePosition ||
      *route.openScopeStagePosition >= route.stageIndices.size()) {
    run.rejectionReason = "missing_scope_run_start";
    return run;
  }

  const size_t beginPosition = *route.openScopeStagePosition;
  llvm::SmallVector<Operation *> roots;
  bool allStagesLocal = true;
  for (size_t position = beginPosition; position < route.stageIndices.size();
       ++position) {
    size_t stageIndex = route.stageIndices[position];
    if (stageIndex >= costTable.stages.size()) {
      run.rejectionReason = "invalid_stage_index";
      return run;
    }
    const LogicalStageCost &stage = costTable.stages[stageIndex];
    if (position == beginPosition)
      run.beginBoundary = stage.beginBoundary;
    run.endBoundary = stage.endBoundary;
    run.stageIndices.push_back(stageIndex);
    allStagesLocal &= stage.localSimtMaterializable;
    llvm::append_range(roots, stage.operations);
  }
  run.superblockFactor = route.routeSuperblockFactor;
  if (!allStagesLocal) {
    run.rejectionReason = "stage_not_locally_materializable";
    return run;
  }

  if (roots.empty()) {
    const LogicalStageCost &first = costTable.stages[run.stageIndices.front()];
    const LogicalStageCost &last = costTable.stages[run.stageIndices.back()];
    run.liveInCount = first.liveInCount;
    run.liveOutCount = last.liveOutCount;
    run.liveInTensorBytes = first.scopeInputTensorBytes;
    run.liveOutTensorBytes = last.scopeOutputTensorBytes;
  } else {
    for (auto [index, operation] : llvm::enumerate(roots)) {
      if (!isMaterializableRoot(operation)) {
        run.rejectionReason = "scope_run_contains_illegal_operation";
        return run;
      }
      if (index > 0 && (roots[index - 1]->getBlock() != operation->getBlock() ||
                        roots[index - 1]->getNextNode() != operation)) {
        run.rejectionReason = "scope_run_is_not_one_contiguous_block_range";
        return run;
      }
    }
    llvm::DenseSet<Operation *> owned;
    for (Operation *root : roots)
      collectOwnedTree(root, owned);
    llvm::SetVector<Value> liveIns;
    llvm::SetVector<Value> liveOuts;
    for (Operation *operation : owned) {
      for (Value operand : operation->getOperands())
        if (!definedInside(operand, owned))
          liveIns.insert(operand);
      for (Value result : operation->getResults())
        if (llvm::any_of(result.getUsers(), [&](Operation *user) {
              return !owned.contains(user);
            }))
          liveOuts.insert(result);
    }
    run.liveInCount = static_cast<int64_t>(liveIns.size());
    run.liveOutCount = static_cast<int64_t>(liveOuts.size());
    run.liveInTensorBytes = sumTensorBytes(liveIns.getArrayRef());
    run.liveOutTensorBytes = sumTensorBytes(liveOuts.getArrayRef());
  }

  const double inputBytes = static_cast<double>(run.liveInTensorBytes);
  const double outputBytes = static_cast<double>(run.liveOutTensorBytes);
  run.nominalTransitionCycles =
      transition.get(StageMode::SIMD, StageMode::SIMT) +
      transition.get(StageMode::SIMT, StageMode::SIMD) +
      inputBytes / transition.simdUbStoreBytesPerCycle +
      inputBytes / transition.simtUbLoadBytesPerCycle +
      outputBytes / transition.simtUbStoreBytesPerCycle +
      outputBytes / transition.simdUbLoadBytesPerCycle;
  run.setupProxyCycles = transition.scopeSetupProxyCycles;
  run.chargedTransitionCycles =
      run.nominalTransitionCycles + (conservative ? run.setupProxyCycles : 0.0);
  run.materializable = true;
  return run;
}

static bool finalizeOpenScope(PartialRoute &route,
                              const StageCostTable &costTable,
                              const StageTransitionCost &transition,
                              bool conservative) {
  if (!route.openScopeStagePosition)
    return true;
  ScopeRunCost run = buildScopeRun(route, costTable, transition, conservative);
  if (!run.materializable)
    return false;
  const size_t chargePosition = *route.openScopeStagePosition;
  route.totalCycles += run.chargedTransitionCycles;
  route.entryTransitionCycles[chargePosition] += run.chargedTransitionCycles;
  route.logicalStageCycles[chargePosition] += run.chargedTransitionCycles;
  route.scopeRuns.push_back(std::move(run));
  route.openScopeStagePosition.reset();
  return true;
}

static unsigned modeIndex(StageMode mode) {
  return mode == StageMode::SIMD ? 0u : 1u;
}

static RouteClass initialClass(StageMode mode) {
  return mode == StageMode::SIMD ? RouteClass::AllSIMD : RouteClass::AllSIMT;
}

static RouteClass appendClass(RouteClass current, StageMode next) {
  if (current == RouteClass::Mixed)
    return current;
  if (current == RouteClass::AllSIMTForMixed)
    return next == StageMode::SIMT ? current : RouteClass::Mixed;
  if ((current == RouteClass::AllSIMD && next == StageMode::SIMD) ||
      (current == RouteClass::AllSIMT && next == StageMode::SIMT))
    return current;
  return RouteClass::Mixed;
}

static StageRoutePlan toPlan(const std::optional<PartialRoute> &route,
                             StageKernelRouteKind kind) {
  StageRoutePlan result;
  result.candidate = kind;
  if (!route)
    return result;
  result.legal = true;
  result.implementations = route->implementations;
  result.stageIndices = route->stageIndices;
  result.entryTransitionCycles = route->entryTransitionCycles;
  result.logicalStageCycles = route->logicalStageCycles;
  result.scopeRuns = route->scopeRuns;
  result.routeSuperblockFactor = route->routeSuperblockFactor;
  result.totalCycles = route->totalCycles;
  result.source = "stage_dynamic_programming";
  return result;
}

static llvm::Expected<StageCostModelSummary>
solveBoundaryGraphRoutes(const StageCostTable &costTable,
                         const StageTransitionCost &transition,
                         bool conservative) {
  if (costTable.boundaryCount < 2)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "Stage Boundary Graph requires at least two boundaries");
  using RouteKey = std::pair<int64_t, int64_t>;
  using FactorRoutes = std::map<RouteKey, PartialRoute>;
  using State = std::array<std::array<FactorRoutes, 4>, 2>;
  std::vector<State> states(static_cast<size_t>(costTable.boundaryCount));

  auto record = [&](State &state, PartialRoute route) {
    auto &routes = state[modeIndex(route.exitMode)]
                        [static_cast<unsigned>(route.routeClass)];
    int64_t openBoundary = -1;
    if (route.openScopeStagePosition) {
      size_t stageIndex = route.stageIndices[*route.openScopeStagePosition];
      openBoundary = costTable.stages[stageIndex].beginBoundary;
    }
    auto [slot, inserted] = routes.try_emplace(
        RouteKey{route.routeSuperblockFactor, openBoundary}, route);
    if (!inserted && route.totalCycles < slot->second.totalCycles)
      slot->second = std::move(route);
  };

  for (size_t boundary = 0;
       boundary + 1 < static_cast<size_t>(costTable.boundaryCount);
       ++boundary) {
    for (auto indexedStage : llvm::enumerate(costTable.stages)) {
      const LogicalStageCost &stage = indexedStage.value();
      const size_t stageIndex = indexedStage.index();
      if (stage.beginBoundary != static_cast<int64_t>(boundary))
        continue;
      if (stage.endBoundary <= stage.beginBoundary ||
          stage.endBoundary >= costTable.boundaryCount)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Stage '%s' has invalid boundary edge [%lld, %lld)",
            stage.id.c_str(), static_cast<long long>(stage.beginBoundary),
            static_cast<long long>(stage.endBoundary));
      if (stage.implementations.empty())
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' has no legal implementation",
                                       stage.id.c_str());
      State &target = states[static_cast<size_t>(stage.endBoundary)];

      for (const StageImplementationCost &cost : stage.implementations) {
        if (!cost.isValid())
          return llvm::createStringError(
              std::errc::invalid_argument,
              "Stage '%s' has an invalid implementation cost",
              stage.id.c_str());
        if (boundary == 0) {
          PartialRoute route;
          route.totalCycles = cost.totalCycles;
          route.exitMode = cost.implementation.mode;
          route.routeClass = initialClass(cost.implementation.mode);
          route.implementations.push_back(cost.implementation);
          route.stageIndices.push_back(stageIndex);
          route.routeSuperblockFactor = cost.implementation.superblockFactor;
          route.entryTransitionCycles.push_back(0.0);
          route.logicalStageCycles.push_back(cost.totalCycles);
          if (cost.implementation.mode == StageMode::SIMT &&
              stage.localSimtMaterializable &&
              llvm::is_contained(stage.localSimtFactors,
                                 cost.implementation.superblockFactor)) {
            PartialRoute futureMixed = route;
            futureMixed.routeClass = RouteClass::AllSIMTForMixed;
            futureMixed.openScopeStagePosition = 0;
            record(target, std::move(futureMixed));
          }
          record(target, std::move(route));
          continue;
        }

        const State &source = states[boundary];
        for (const auto &byClass : source) {
          for (const auto &factorRoutes : byClass) {
            for (const auto &factorRoute : factorRoutes) {
              const PartialRoute &previous = factorRoute.second;
              if (previous.routeClass == RouteClass::AllSIMT &&
                  cost.implementation.mode == StageMode::SIMD)
                continue;
              const bool routeAlreadyHasSimt =
                  previous.routeClass != RouteClass::AllSIMD;
              if (cost.implementation.mode == StageMode::SIMT &&
                  routeAlreadyHasSimt &&
                  cost.implementation.superblockFactor !=
                      previous.routeSuperblockFactor)
                continue;
              PartialRoute route = previous;
              route.routeClass =
                  appendClass(route.routeClass, cost.implementation.mode);
              if (cost.implementation.mode == StageMode::SIMT &&
                  route.routeClass != RouteClass::AllSIMT &&
                  (!stage.localSimtMaterializable ||
                   !llvm::is_contained(stage.localSimtFactors,
                                       cost.implementation.superblockFactor)))
                continue;
              if (previous.exitMode == StageMode::SIMT &&
                  cost.implementation.mode == StageMode::SIMD &&
                  !finalizeOpenScope(route, costTable, transition,
                                     conservative))
                continue;
              route.exitMode = cost.implementation.mode;
              route.implementations.push_back(cost.implementation);
              route.stageIndices.push_back(stageIndex);
              if (cost.implementation.mode == StageMode::SIMT &&
                  !routeAlreadyHasSimt)
                route.routeSuperblockFactor =
                    cost.implementation.superblockFactor;
              if (cost.implementation.mode == StageMode::SIMT &&
                  previous.exitMode == StageMode::SIMD)
                route.openScopeStagePosition = route.stageIndices.size() - 1;
              route.entryTransitionCycles.push_back(0.0);
              route.logicalStageCycles.push_back(cost.totalCycles);
              route.totalCycles += cost.totalCycles;
              record(target, std::move(route));
            }
          }
        }
      }
    }
  }

  const State &finalState = states.back();
  auto bestClass = [&](RouteClass routeClass) -> std::optional<PartialRoute> {
    std::optional<PartialRoute> best;
    for (const auto &byClass : finalState) {
      const auto &candidates = byClass[static_cast<unsigned>(routeClass)];
      for (const auto &factorRoute : candidates) {
        PartialRoute candidate = factorRoute.second;
        if (routeClass == RouteClass::Mixed &&
            candidate.exitMode == StageMode::SIMT &&
            !finalizeOpenScope(candidate, costTable, transition, conservative))
          continue;
        if (!best || candidate.totalCycles < best->totalCycles)
          best = std::move(candidate);
      }
    }
    return best;
  };

  StageCostModelSummary result;
  result.applied = true;
  result.boundarySource = costTable.boundarySource;
  result.operationOwnershipComplete = costTable.operationOwnershipComplete;
  result.modeledOperationCount = costTable.modeledOperationCount;
  result.profileVersion = costTable.profileVersion;
  result.stages = costTable.stages;
  result.boundaryCount = costTable.boundaryCount;
  result.discoveryJSON = costTable.discoveryJSON;
  result.transition = transition;
  result.allSimd =
      toPlan(bestClass(RouteClass::AllSIMD), StageKernelRouteKind::AllSIMD);
  result.allSimt =
      toPlan(bestClass(RouteClass::AllSIMT), StageKernelRouteKind::AllSIMT);
  result.mixed =
      toPlan(bestClass(RouteClass::Mixed), StageKernelRouteKind::Mixed);
  return result;
}

} // namespace

llvm::StringRef mlir::ascend::stringifyStageMode(StageMode mode) {
  return mode == StageMode::SIMD ? "simd" : "simt";
}

llvm::StringRef
mlir::ascend::stringifyStageKernelRoute(StageKernelRouteKind kind) {
  switch (kind) {
  case StageKernelRouteKind::AllSIMD:
    return "all_simd";
  case StageKernelRouteKind::AllSIMT:
    return "all_simt_only";
  case StageKernelRouteKind::Mixed:
    return "mixed_simd_simt";
  }
  llvm_unreachable("unknown stage kernel route kind");
}

llvm::StringRef mlir::ascend::stringifyStageSchedule(StageScheduleKind kind) {
  switch (kind) {
  case StageScheduleKind::StraightLine:
    return "straight_line";
  case StageScheduleKind::IndependentPipelined:
    return "independent_pipelined";
  case StageScheduleKind::LoopCarriedSerial:
    return "loop_carried_serial";
  case StageScheduleKind::PartiallyDependent:
    return "partially_dependent";
  }
  llvm_unreachable("unknown stage schedule kind");
}

bool StageImplementation::isValid() const {
  if (superblockFactor <= 0 || (superblockFactor & (superblockFactor - 1)) != 0)
    return false;
  return mode == StageMode::SIMT || superblockFactor == 1;
}

llvm::json::Object StageImplementation::toJSON() const {
  llvm::json::Object result;
  result["mode"] = stringifyStageMode(mode);
  result["superblock_factor"] = superblockFactor;
  return result;
}

bool StageModelFeatures::isValid() const {
  return conditionalBranchCount >= 0 && divergentBranchCount >= 0 &&
         loopBackedgeCount >= 0 && synchronizationCount >= 0 &&
         parallelRecurrenceGroupCount > 0 && std::isfinite(activeLaneRatio) &&
         activeLaneRatio >= 0.0 && activeLaneRatio <= 1.0 &&
         (!hasLoopCarriedDataDependency || hasLoop);
}

bool StageModelFeatures::permitsSimdRoofline() const {
  return !hasLoopCarriedDataDependency;
}

bool StageWorkload::isFiniteAndNonNegative() const {
  const std::array<double, 10> values = {scalarOperations,
                                         loadBytes,
                                         storeBytes,
                                         loadWarpInstructions,
                                         storeWarpInstructions,
                                         predicateElements,
                                         shuffleLaneSteps,
                                         dotFlops,
                                         issueElements,
                                         estimatedSpillTransactions};
  if (!std::all_of(values.begin(), values.end(), [](double value) {
        return std::isfinite(value) && value >= 0.0;
      }))
    return false;
  return llvm::all_of(operationElements, [](const auto &entry) {
    return std::isfinite(entry.second) && entry.second >= 0.0;
  });
}

llvm::json::Object StageWorkload::toJSON() const {
  llvm::json::Object result;
  llvm::json::Object operations;
  for (const auto &[name, elements] : operationElements)
    operations[name] = elements;
  result["operation_elements_per_iteration"] = std::move(operations);
  result["scalar_operations_per_iteration"] = scalarOperations;
  result["load_bytes_per_iteration"] = loadBytes;
  result["store_bytes_per_iteration"] = storeBytes;
  result["load_warp_instructions_per_iteration"] = loadWarpInstructions;
  result["store_warp_instructions_per_iteration"] = storeWarpInstructions;
  result["predicate_elements_per_iteration"] = predicateElements;
  result["shuffle_lane_steps_per_iteration"] = shuffleLaneSteps;
  result["dot_flops_per_iteration"] = dotFlops;
  result["issue_elements_per_iteration"] = issueElements;
  result["estimated_spill_transactions_per_iteration"] =
      estimatedSpillTransactions;
  result["pays_kernel_setup"] = paysKernelSetup;
  return result;
}

llvm::json::Object StageModelFeatures::toJSON() const {
  llvm::json::Object result;
  result["has_loop"] = hasLoop;
  result["has_loop_carried_data_dependency"] = hasLoopCarriedDataDependency;
  result["has_pointer_induction"] = hasPointerInduction;
  result["has_contiguous_memory"] = hasContiguousMemory;
  result["has_indirect_memory"] = hasIndirectMemory;
  result["has_reduction"] = hasReduction;
  result["has_dot"] = hasDot;
  result["has_conversion_pack"] = hasConversionPack;
  result["conditional_branch_count"] = conditionalBranchCount;
  result["divergent_branch_count"] = divergentBranchCount;
  result["loop_backedge_count"] = loopBackedgeCount;
  result["synchronization_count"] = synchronizationCount;
  result["parallel_recurrence_group_count"] = parallelRecurrenceGroupCount;
  result["active_lane_ratio"] = activeLaneRatio;
  result["simd_roofline_permitted"] = permitsSimdRoofline();
  result["source"] = source;
  return result;
}

bool StageResourceCycles::isFiniteAndNonNegative() const {
  const std::array<double, 17> values = {
      setup,           scalar, load,    store,        compute,       predicate,
      shuffle,         dot,    control, loopControl,  branchControl, divergence,
      synchronization, spill,  issue,   criticalPath, epilogue};
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value) && value >= 0.0;
  });
}

llvm::json::Object StageResourceCycles::toJSON() const {
  llvm::json::Object result;
  result["setup"] = setup;
  result["scalar_per_iteration"] = scalar;
  result["load_per_iteration"] = load;
  result["store_per_iteration"] = store;
  result["compute_per_iteration"] = compute;
  result["predicate_per_iteration"] = predicate;
  result["shuffle_per_iteration"] = shuffle;
  result["dot_per_iteration"] = dot;
  result["control_per_iteration"] = control;
  result["loop_control_per_iteration"] = loopControl;
  result["branch_control_per_iteration"] = branchControl;
  result["divergence_per_iteration"] = divergence;
  result["synchronization_per_iteration"] = synchronization;
  result["spill_per_iteration"] = spill;
  result["issue_per_iteration"] = issue;
  result["critical_path_per_iteration"] = criticalPath;
  result["epilogue"] = epilogue;
  return result;
}

bool StageImplementationCost::isValid() const {
  return implementation.isValid() && std::isfinite(totalCycles) &&
         totalCycles >= 0.0 && resources.isFiniteAndNonNegative() &&
         !modelName.empty() && !profileVersion.empty();
}

llvm::json::Object StageImplementationCost::toJSON() const {
  llvm::json::Object result;
  result["implementation"] = implementation.toJSON();
  result["total_system_cycles"] = totalCycles;
  result["resource_system_cycles"] = resources.toJSON();
  result["model_name"] = modelName;
  result["profile_version"] = profileVersion;
  result["source"] = source;
  return result;
}

llvm::json::Object LogicalStageCost::toJSON() const {
  llvm::json::Object result;
  result["id"] = id;
  result["description"] = description;
  result["model"] = model;
  result["schedule_kind"] = stringifyStageSchedule(schedule);
  result["iteration_count"] = iterationCount;
  result["features"] = features.toJSON();
  result["workload"] = workload.toJSON();
  result["owned_operation_count"] = ownedOperationCount;
  result["live_in_count"] = liveInCount;
  result["live_out_count"] = liveOutCount;
  result["live_in_bytes"] = liveInBytes;
  result["live_out_bytes"] = liveOutBytes;
  result["local_simt_scope_count"] = localSimtScopeCount;
  result["scope_input_tensor_bytes"] = scopeInputTensorBytes;
  result["scope_output_tensor_bytes"] = scopeOutputTensorBytes;
  llvm::json::Array anchorIndices;
  for (unsigned index : simtAnchorIndices)
    anchorIndices.push_back(static_cast<int64_t>(index));
  result["simt_anchor_indices"] = std::move(anchorIndices);
  result["local_simt_materializable"] = localSimtMaterializable;
  llvm::json::Array localFactors;
  for (int64_t factor : localSimtFactors)
    localFactors.push_back(factor);
  result["local_simt_factors"] = std::move(localFactors);
  llvm::json::Array costs;
  for (const StageImplementationCost &implementation : implementations)
    costs.push_back(implementation.toJSON());
  result["implementations"] = std::move(costs);
  if (beginBoundary >= 0 && endBoundary > beginBoundary) {
    result["begin_boundary"] = beginBoundary;
    result["end_boundary"] = endBoundary;
  }
  return result;
}

llvm::json::Object StageCostTable::toJSON() const {
  llvm::json::Object result;
  result["boundary_source"] = boundarySource;
  result["operation_ownership_complete"] = operationOwnershipComplete;
  result["modeled_operation_count"] = modeledOperationCount;
  result["profile_version"] = profileVersion;
  result["boundary_count"] = boundaryCount;
  llvm::json::Array stageCosts;
  for (const LogicalStageCost &stage : stages)
    stageCosts.push_back(stage.toJSON());
  result["stages"] = std::move(stageCosts);
  if (!discoveryJSON.empty()) {
    auto parsed = llvm::json::parse(discoveryJSON);
    if (parsed)
      result["stage_boundary_graph"] = std::move(*parsed);
  }
  return result;
}

bool StageTransitionCost::isValid() const {
  return std::isfinite(simdToSimtCycles) && std::isfinite(simtToSimdCycles) &&
         simdToSimtCycles >= 0.0 && simtToSimdCycles >= 0.0 &&
         std::isfinite(simdUbLoadBytesPerCycle) &&
         simdUbLoadBytesPerCycle > 0.0 &&
         std::isfinite(simdUbStoreBytesPerCycle) &&
         simdUbStoreBytesPerCycle > 0.0 &&
         std::isfinite(simtUbLoadBytesPerCycle) &&
         simtUbLoadBytesPerCycle > 0.0 &&
         std::isfinite(simtUbStoreBytesPerCycle) &&
         simtUbStoreBytesPerCycle > 0.0 && simtWarpSize > 0 &&
         std::isfinite(scopeSetupProxyCycles) && scopeSetupProxyCycles >= 0.0;
}

double StageTransitionCost::get(StageMode from, StageMode to) const {
  if (from == to)
    return 0.0;
  return from == StageMode::SIMD ? simdToSimtCycles : simtToSimdCycles;
}

llvm::json::Object StageTransitionCost::toJSON() const {
  llvm::json::Object result;
  result["simd_to_simt_system_cycles"] = simdToSimtCycles;
  result["simt_to_simd_system_cycles"] = simtToSimdCycles;
  result["simd_ub_load_bytes_per_system_cycle"] = simdUbLoadBytesPerCycle;
  result["simd_ub_store_bytes_per_system_cycle"] = simdUbStoreBytesPerCycle;
  result["simt_ub_load_bytes_per_system_cycle"] = simtUbLoadBytesPerCycle;
  result["simt_ub_store_bytes_per_system_cycle"] = simtUbStoreBytesPerCycle;
  result["simt_warp_size"] = simtWarpSize;
  result["scope_setup_proxy_system_cycles"] = scopeSetupProxyCycles;
  result["scope_setup_proxy_confidence"] = scopeSetupProxyConfidence;
  result["scope_setup_proxy_source"] = scopeSetupProxySource;
  result["source"] = source;
  return result;
}

llvm::json::Object ScopeRunCost::toJSON() const {
  llvm::json::Object result;
  result["begin_boundary"] = beginBoundary;
  result["end_boundary"] = endBoundary;
  result["superblock_factor"] = superblockFactor;
  llvm::json::Array stages;
  for (size_t index : stageIndices)
    stages.push_back(static_cast<int64_t>(index));
  result["stage_indices"] = std::move(stages);
  result["live_in_count"] = liveInCount;
  result["live_out_count"] = liveOutCount;
  result["live_in_tensor_bytes"] = liveInTensorBytes;
  result["live_out_tensor_bytes"] = liveOutTensorBytes;
  result["nominal_transition_system_cycles"] = nominalTransitionCycles;
  result["scope_setup_proxy_system_cycles"] = setupProxyCycles;
  result["charged_transition_system_cycles"] = chargedTransitionCycles;
  result["materializable"] = materializable;
  if (!rejectionReason.empty())
    result["rejection_reason"] = rejectionReason;
  return result;
}

llvm::json::Object StageRoutePlan::toJSON() const {
  llvm::json::Object result;
  result["candidate"] = stringifyStageKernelRoute(candidate);
  result["legal"] = legal;
  result["total_system_cycles"] = totalCycles;
  result["route_superblock_factor"] = routeSuperblockFactor;
  result["source"] = source;
  llvm::json::Array stages;
  for (size_t i = 0; i < implementations.size(); ++i) {
    llvm::json::Object stage;
    if (i < stageIndices.size())
      stage["stage_index"] = static_cast<int64_t>(stageIndices[i]);
    stage["implementation"] = implementations[i].toJSON();
    stage["entry_transition_system_cycles"] = entryTransitionCycles[i];
    stage["logical_stage_system_cycles"] = logicalStageCycles[i];
    stages.push_back(std::move(stage));
  }
  result["stages"] = std::move(stages);
  llvm::json::Array runs;
  for (const ScopeRunCost &run : scopeRuns)
    runs.push_back(run.toJSON());
  result["scope_runs"] = std::move(runs);
  return result;
}

llvm::json::Object StageCostModelSummary::toJSON() const {
  llvm::json::Object result;
  result["applied"] = applied;
  result["boundary_source"] = boundarySource;
  result["operation_ownership_complete"] = operationOwnershipComplete;
  result["modeled_operation_count"] = modeledOperationCount;
  result["profile_version"] = profileVersion;
  llvm::json::Array stageArray;
  for (const LogicalStageCost &stage : stages)
    stageArray.push_back(stage.toJSON());
  result["logical_stages"] = std::move(stageArray);
  result["boundary_count"] = boundaryCount;
  if (!discoveryJSON.empty()) {
    auto parsed = llvm::json::parse(discoveryJSON);
    if (parsed)
      result["stage_boundary_graph"] = std::move(*parsed);
  }
  result["transition_cost"] = transition.toJSON();
  llvm::json::Object routes;
  routes["all_simd"] = allSimd.toJSON();
  routes["all_simt_only"] = allSimt.toJSON();
  routes["mixed_simd_simt"] = mixed.toJSON();
  routes["mixed_simd_simt_conservative"] = conservativeMixed.toJSON();
  result["routes"] = std::move(routes);
  return result;
}

llvm::Expected<StageCostModelSummary>
mlir::ascend::solveStageRoutes(const StageCostTable &costTable,
                               const StageTransitionCost &transition) {
  if (costTable.stages.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "stage route model requires at least one "
                                   "logical stage");
  if (!transition.isValid())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "stage transition costs must be finite and "
                                   "non-negative");
  if (costTable.boundaryCount < 2)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "stage route model requires a Stage Boundary Graph");
  auto nominal = solveBoundaryGraphRoutes(costTable, transition, false);
  if (!nominal)
    return nominal.takeError();
  auto conservative = solveBoundaryGraphRoutes(costTable, transition, true);
  if (!conservative)
    return conservative.takeError();
  nominal->conservativeMixed = std::move(conservative->mixed);
  nominal->conservativeMixed.source =
      "stage_dynamic_programming_conservative_scope_setup";
  return std::move(*nominal);
}
