//===- StageRouteCostModel.cpp - Logical-stage route solver ---------------===//

#include "AscendModel/StageModel/StageRouteCostModel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <cmath>
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
  /// Counterfactual cost used if this prefix later becomes a mixed route.
  /// Local SIMT Stages retain their selected F1/F2/F4 implementation and add
  /// the exact UB-backed scope-boundary cost.
  double mixedEquivalentCycles = 0.0;
  StageMode exitMode = StageMode::SIMD;
  RouteClass routeClass = RouteClass::AllSIMD;
  bool allSimtStagesLocal = true;
  std::vector<StageImplementation> implementations;
  std::vector<size_t> stageIndices;
  std::vector<double> entryTransitionCycles;
  std::vector<double> logicalStageCycles;
  std::vector<double> mixedEquivalentStageCycles;
  int64_t routeSuperblockFactor = 1;
};

static double mixedEquivalentStageCost(const LogicalStageCost &stage,
                                       const StageImplementationCost &selected,
                                       const StageTransitionCost &transition) {
  if (selected.implementation.mode != StageMode::SIMT ||
      !stage.localSimtMaterializable)
    return selected.totalCycles;
  // Materializer currently creates one scope per primitive anchor.  The
  // route DP otherwise observes only one Stage-mode change and would charge
  // one transition pair even when the generated TTIR contains several local
  // scopes.  Charge the additional physical pairs here; all-SIMT routes do
  // not consume this mixed-only equivalent cost.
  const int64_t scopeCount = std::max<int64_t>(1, stage.localSimtScopeCount);
  const double fixedScopeTransitions =
      static_cast<double>(scopeCount) *
      (transition.get(StageMode::SIMD, StageMode::SIMT) +
       transition.get(StageMode::SIMT, StageMode::SIMD));
  const double activeThreads =
      std::max(1.0, static_cast<double>(transition.simtWarpSize) *
                        std::clamp(stage.features.activeLaneRatio, 0.0, 1.0));
  const double simtLoadBytesPerCycle =
      transition.simtUbLoadBytesPerThreadPerCycle * activeThreads;
  const double simtStoreBytesPerCycle =
      transition.simtUbStoreBytesPerThreadPerCycle * activeThreads;
  const double inputBytes = static_cast<double>(stage.scopeInputTensorBytes);
  const double outputBytes = static_cast<double>(stage.scopeOutputTensorBytes);
  // SIMD producer register -> UB -> SIMT register.
  const double inputHandoffCycles =
      inputBytes / transition.simdUbStoreBytesPerCycle +
      inputBytes / simtLoadBytesPerCycle;
  // SIMT producer register -> UB -> SIMD register.
  const double outputHandoffCycles =
      outputBytes / simtStoreBytesPerCycle +
      outputBytes / transition.simdUbLoadBytesPerCycle;
  return selected.totalCycles + fixedScopeTransitions + inputHandoffCycles +
         outputHandoffCycles;
}

static double scopeEntryCost(const LogicalStageCost &stage,
                             const StageTransitionCost &transition) {
  const double activeThreads =
      std::max(1.0, static_cast<double>(transition.simtWarpSize) *
                        std::clamp(stage.features.activeLaneRatio, 0.0, 1.0));
  const double inputBytes = static_cast<double>(stage.scopeInputTensorBytes);
  return transition.get(StageMode::SIMD, StageMode::SIMT) +
         inputBytes / transition.simdUbStoreBytesPerCycle +
         inputBytes /
             (transition.simtUbLoadBytesPerThreadPerCycle * activeThreads);
}

static double scopeExitCost(const LogicalStageCost &stage,
                            const StageTransitionCost &transition) {
  const double activeThreads =
      std::max(1.0, static_cast<double>(transition.simtWarpSize) *
                        std::clamp(stage.features.activeLaneRatio, 0.0, 1.0));
  const double outputBytes = static_cast<double>(stage.scopeOutputTensorBytes);
  return transition.get(StageMode::SIMT, StageMode::SIMD) +
         outputBytes /
             (transition.simtUbStoreBytesPerThreadPerCycle * activeThreads) +
         outputBytes / transition.simdUbLoadBytesPerCycle;
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
  result.routeSuperblockFactor = route->routeSuperblockFactor;
  result.totalCycles = route->totalCycles;
  result.source = "stage_dynamic_programming";
  return result;
}

static llvm::Expected<StageCostModelSummary>
solveBoundaryGraphRoutes(const StageCostTable &costTable,
                         const StageTransitionCost &transition) {
  if (costTable.boundaryCount < 2)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "Stage Boundary Graph requires at least two boundaries");
  using FactorRoutes = std::map<int64_t, PartialRoute>;
  using State = std::array<std::array<FactorRoutes, 4>, 2>;
  std::vector<State> states(static_cast<size_t>(costTable.boundaryCount));

  auto record = [](State &state, PartialRoute route) {
    auto &routes = state[modeIndex(route.exitMode)]
                        [static_cast<unsigned>(route.routeClass)];
    auto [slot, inserted] =
        routes.try_emplace(route.routeSuperblockFactor, route);
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
            "candidate Stage '%s' has invalid boundary edge [%lld, %lld)",
            stage.id.c_str(), static_cast<long long>(stage.beginBoundary),
            static_cast<long long>(stage.endBoundary));
      if (stage.implementations.empty())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "candidate Stage '%s' has no legal implementation",
            stage.id.c_str());
      State &target = states[static_cast<size_t>(stage.endBoundary)];

      for (const StageImplementationCost &cost : stage.implementations) {
        if (!cost.isValid())
          return llvm::createStringError(
              std::errc::invalid_argument,
              "candidate Stage '%s' has an invalid implementation cost",
              stage.id.c_str());
        if (boundary == 0) {
          PartialRoute route;
          route.totalCycles = cost.totalCycles;
          route.mixedEquivalentCycles = cost.totalCycles;
          if (cost.implementation.mode == StageMode::SIMT)
            route.mixedEquivalentCycles += scopeEntryCost(stage, transition);
          route.exitMode = cost.implementation.mode;
          route.routeClass = initialClass(cost.implementation.mode);
          route.allSimtStagesLocal =
              cost.implementation.mode != StageMode::SIMT ||
              (stage.localSimtMaterializable &&
               llvm::is_contained(stage.localSimtFactors,
                                  cost.implementation.superblockFactor));
          route.implementations.push_back(cost.implementation);
          route.stageIndices.push_back(stageIndex);
          route.routeSuperblockFactor = cost.implementation.superblockFactor;
          route.entryTransitionCycles.push_back(0.0);
          route.logicalStageCycles.push_back(cost.totalCycles);
          route.mixedEquivalentStageCycles.push_back(
              route.mixedEquivalentCycles);
          if (cost.implementation.mode == StageMode::SIMT &&
              route.allSimtStagesLocal) {
            PartialRoute futureMixed = route;
            futureMixed.routeClass = RouteClass::AllSIMTForMixed;
            futureMixed.totalCycles = futureMixed.mixedEquivalentCycles;
            futureMixed.entryTransitionCycles.front() =
                futureMixed.mixedEquivalentCycles - cost.totalCycles;
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
              route.allSimtStagesLocal =
                  route.allSimtStagesLocal &&
                  (cost.implementation.mode != StageMode::SIMT ||
                   (stage.localSimtMaterializable &&
                    llvm::is_contained(stage.localSimtFactors,
                                       cost.implementation.superblockFactor)));
              if (route.routeClass == RouteClass::Mixed &&
                  !route.allSimtStagesLocal)
                continue;
              route.exitMode = cost.implementation.mode;
              route.implementations.push_back(cost.implementation);
              route.stageIndices.push_back(stageIndex);
              if (cost.implementation.mode == StageMode::SIMT &&
                  !routeAlreadyHasSimt)
                route.routeSuperblockFactor =
                    cost.implementation.superblockFactor;
              const double logicalStageCycles = cost.totalCycles;
              double mixedLogicalStageCycles = logicalStageCycles;
              double entryTransition = 0.0;
              if (previous.exitMode == StageMode::SIMD &&
                  cost.implementation.mode == StageMode::SIMT)
                entryTransition = scopeEntryCost(stage, transition);
              if (previous.exitMode == StageMode::SIMT &&
                  cost.implementation.mode == StageMode::SIMD) {
                const LogicalStageCost &previousStage =
                    costTable.stages[previous.stageIndices.back()];
                entryTransition = scopeExitCost(previousStage, transition);
              }
              mixedLogicalStageCycles += entryTransition;
              route.entryTransitionCycles.push_back(entryTransition);
              route.mixedEquivalentCycles += mixedLogicalStageCycles;
              route.mixedEquivalentStageCycles.push_back(
                  mixedLogicalStageCycles);
              if (route.routeClass == RouteClass::Mixed) {
                route.totalCycles = route.mixedEquivalentCycles;
                route.logicalStageCycles = route.mixedEquivalentStageCycles;
              } else {
                route.logicalStageCycles.push_back(logicalStageCycles);
                route.totalCycles += logicalStageCycles;
              }
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
            candidate.exitMode == StageMode::SIMT) {
          const LogicalStageCost &lastStage =
              costTable.stages[candidate.stageIndices.back()];
          const double exitCost = scopeExitCost(lastStage, transition);
          candidate.totalCycles += exitCost;
          candidate.mixedEquivalentCycles += exitCost;
          candidate.logicalStageCycles.back() += exitCost;
          candidate.mixedEquivalentStageCycles.back() += exitCost;
        }
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
  llvm::json::Array candidates;
  for (const LogicalStageCost &stage : stages)
    candidates.push_back(stage.toJSON());
  result["candidate_stages"] = std::move(candidates);
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
         std::isfinite(simtUbLoadBytesPerThreadPerCycle) &&
         simtUbLoadBytesPerThreadPerCycle > 0.0 &&
         std::isfinite(simtUbStoreBytesPerThreadPerCycle) &&
         simtUbStoreBytesPerThreadPerCycle > 0.0 && simtWarpSize > 0;
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
  result["simt_ub_load_bytes_per_thread_per_system_cycle"] =
      simtUbLoadBytesPerThreadPerCycle;
  result["simt_ub_store_bytes_per_thread_per_system_cycle"] =
      simtUbStoreBytesPerThreadPerCycle;
  result["simt_warp_size"] = simtWarpSize;
  result["source"] = source;
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
      stage["candidate_stage_index"] = static_cast<int64_t>(stageIndices[i]);
    stage["implementation"] = implementations[i].toJSON();
    stage["entry_transition_system_cycles"] = entryTransitionCycles[i];
    stage["logical_stage_system_cycles"] = logicalStageCycles[i];
    stages.push_back(std::move(stage));
  }
  result["stages"] = std::move(stages);
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
  return solveBoundaryGraphRoutes(costTable, transition);
}
