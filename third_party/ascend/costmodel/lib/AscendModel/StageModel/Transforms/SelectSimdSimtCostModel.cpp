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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
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

static std::string hashText(llvm::StringRef text) {
  llvm::ArrayRef<uint8_t> bytes(reinterpret_cast<const uint8_t *>(text.data()),
                                text.size());
  auto digest = llvm::SHA256::hash(bytes);
  return llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
}

static std::string hashTTIR(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return hashText(text);
}

static std::string getKernelName(ModuleOp module) {
  std::string name = "kernel";
  module.walk([&](Operation *operation) {
    if (operation == module.getOperation())
      return WalkResult::advance();
    auto symbol = operation->getAttrOfType<StringAttr>("sym_name");
    if (!symbol || symbol.getValue().empty())
      return WalkResult::advance();
    name = symbol.getValue().str();
    return WalkResult::interrupt();
  });
  for (char &character : name) {
    unsigned char value = static_cast<unsigned char>(character);
    if (!std::isalnum(value) && character != '-' && character != '_')
      character = '_';
  }
  return name.empty() ? "kernel" : name;
}

static llvm::SmallString<256> childPath(llvm::StringRef parent,
                                        llvm::StringRef child) {
  llvm::SmallString<256> path(parent);
  llvm::sys::path::append(path, child);
  return path;
}

static llvm::Error writePrettyJSON(llvm::StringRef path,
                                   llvm::json::Value value) {
  llvm::SmallString<256> temporaryPath;
  int fileDescriptor = -1;
  std::error_code error = llvm::sys::fs::createUniqueFile(
      llvm::Twine(path) + ".tmp-%%%%%%", fileDescriptor, temporaryPath,
      llvm::sys::fs::OF_Text);
  if (error)
    return llvm::createStringError(error,
                                   "StageModel snapshot cannot create "
                                   "temporary file for `%s`",
                                   path.str().c_str());
  {
    llvm::raw_fd_ostream os(fileDescriptor, true);
    os << llvm::formatv("{0:2}\n", value);
    os.flush();
    if (os.has_error()) {
      os.clear_error();
      llvm::sys::fs::remove(temporaryPath);
      return llvm::createStringError(
          std::errc::io_error, "StageModel snapshot failed while writing `%s`",
          path.str().c_str());
    }
  }
  error = llvm::sys::fs::rename(temporaryPath, path);
  if (error) {
    llvm::sys::fs::remove(temporaryPath);
    return llvm::createStringError(
        error, "StageModel snapshot cannot atomically replace `%s`",
        path.str().c_str());
  }
  return llvm::Error::success();
}

static llvm::json::Array
selectableCandidates(const SimdSimtCostReport &report) {
  llvm::json::Array candidates;
  if (report.allSimdCandidateLegal)
    candidates.push_back(kAllSimd);
  if (report.allSimtOnlyCandidateLegal)
    candidates.push_back(kAllSimtOnly);
  if (report.mixedCandidateLegal)
    candidates.push_back(kMixedSimdSimt);
  return candidates;
}

static bool hasImplementation(const LogicalStageCost &stage, StageMode mode) {
  return llvm::any_of(stage.implementations,
                      [mode](const StageImplementationCost &cost) {
                        return cost.implementation.mode == mode;
                      });
}

static llvm::Error removeOldStageFiles(llvm::StringRef stagePath) {
  std::error_code error;
  llvm::sys::fs::directory_iterator iterator(stagePath, error), end;
  if (error)
    return llvm::createStringError(
        error, "StageModel snapshot cannot inspect Stage directory `%s`",
        stagePath.str().c_str());
  for (; iterator != end; iterator.increment(error)) {
    if (error)
      return llvm::createStringError(
          error, "StageModel snapshot cannot scan Stage directory `%s`",
          stagePath.str().c_str());
    if (llvm::sys::path::extension(iterator->path()) != ".json")
      continue;
    error = llvm::sys::fs::remove(iterator->path());
    if (error)
      return llvm::createStringError(
          error, "StageModel snapshot cannot remove stale Stage `%s`",
          iterator->path().c_str());
  }
  return llvm::Error::success();
}

static llvm::Error writeStageModelSnapshot(
    llvm::StringRef rootPath, ModuleOp module, llvm::StringRef ttirSha256,
    llvm::StringRef mode, const SimdSimtCostModelOptions &options,
    const SimdSimtCostReport &report, llvm::StringRef recommended,
    llvm::StringRef effective, llvm::StringRef selectionSource,
    llvm::StringRef applicationReason, bool actionSupported,
    int64_t selectedSuperblockFactor, int64_t materializedAnchorCount,
    const std::optional<StageMaterializationPlan> &materializationPlan,
    const std::optional<StageMaterializationPlan> &nominalMixedPlan) {
  if (rootPath.empty())
    return llvm::Error::success();

  std::string configKey =
      llvm::formatv(
          "{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}|{9}", mode, options.actualTarget,
          options.numWarps, options.compileOn91095,
          options.wholeKernelSuperblockMaterializable,
          options.scopeSuperblockMaterializable,
          options.logicalProgramCountHint, options.routeTransformCapabilityJSON,
          report.selectionProfileContentSha256,
          report.microbenchmarkProfileContentSha256)
          .str();
  const std::string configSha256 = hashText(configKey);
  const std::string kernelName = getKernelName(module);
  const std::string directoryName =
      llvm::formatv("{0}-{1}-{2}", kernelName, ttirSha256.take_front(12),
                    llvm::StringRef(configSha256).take_front(12))
          .str();
  llvm::SmallString<256> snapshotPath = childPath(rootPath, directoryName);
  llvm::SmallString<256> stagePath = childPath(snapshotPath, "stages");
  if (std::error_code error = llvm::sys::fs::create_directories(stagePath))
    return llvm::createStringError(
        error, "StageModel snapshot cannot create directory `%s`",
        stagePath.c_str());

  llvm::SmallString<256> manifestPath =
      childPath(snapshotPath, "manifest.json");
  llvm::sys::fs::remove(manifestPath);
  if (auto error = removeOldStageFiles(stagePath))
    return error;

  llvm::json::Object config;
  config["config_sha256"] = configSha256;
  config["mode"] = mode;
  config["profile_path"] = options.profilePath;
  config["actual_target"] = options.actualTarget;
  config["num_warps"] = static_cast<int64_t>(options.numWarps);
  config["compile_on_91095"] = options.compileOn91095;
  config["whole_kernel_superblock_materializable"] =
      options.wholeKernelSuperblockMaterializable;
  config["scope_superblock_materializable"] =
      options.scopeSuperblockMaterializable;
  config["logical_program_count_hint"] = options.logicalProgramCountHint;
  if (auto capability = llvm::json::parse(options.routeTransformCapabilityJSON))
    config["route_transform_capability"] = std::move(*capability);
  config["selection_profile_content_sha256"] =
      report.selectionProfileContentSha256;
  config["microbenchmark_profile_content_sha256"] =
      report.microbenchmarkProfileContentSha256;
  if (auto error = writePrettyJSON(childPath(snapshotPath, "config.json"),
                                   std::move(config)))
    return error;

  llvm::json::Object summary;
  summary["stage_model_snapshot_version"] = "3.0";
  summary["report_schema_version"] = report.schemaVersion;
  summary["kernel_name"] = kernelName;
  summary["ttir_sha256"] = ttirSha256;
  summary["config_sha256"] = configSha256;
  summary["model"] = report.model;
  summary["profile_version"] = report.profileVersion;
  summary["profile_target"] = report.profileTarget;
  summary["actual_target"] = report.actualTarget;
  summary["target_compatible"] = report.targetCompatible;
  summary["profile_content_sha256"] = report.profileContentSha256;
  summary["selection_profile_content_sha256"] =
      report.selectionProfileContentSha256;
  llvm::json::Object sharedEvidence;
  sharedEvidence["profile_version"] = report.microbenchmarkProfileVersion;
  sharedEvidence["target"] = report.microbenchmarkProfileTarget;
  sharedEvidence["content_sha256"] = report.microbenchmarkProfileContentSha256;
  summary["shared_microbenchmark_profile"] = std::move(sharedEvidence);
  summary["unit"] = report.scoreUnit;
  summary["score_scope"] = report.scoreScope;
  summary["candidate_costs"] = report.candidateCosts.toJSON();
  summary["conservative_candidate_costs"] =
      report.conservativeCandidateCosts.toJSON();
  summary["candidate_ratios_to_best"] = report.candidateRatiosToBest.toJSON();
  summary["best_score"] = report.bestScore;
  summary["selectable_candidates"] = selectableCandidates(report);
  llvm::json::Object candidateRoles;
  candidateRoles[kAllSimd] =
      report.allSimdCandidateLegal ? "selectable_candidate" : "inapplicable";
  candidateRoles[kAllSimtOnly] = report.allSimtOnlyCandidateLegal
                                     ? "selectable_candidate"
                                     : "inapplicable";
  candidateRoles[kMixedSimdSimt] =
      report.mixedCandidateLegal ? "selectable_candidate" : "inapplicable";
  summary["candidate_roles"] = std::move(candidateRoles);
  llvm::json::Array unsupported;
  for (const std::string &term : report.unsupported)
    unsupported.push_back(term);
  summary["unmodeled_cost_terms"] = std::move(unsupported);
  summary["recommended_decision_kind"] = recommended;
  summary["nominal_decision_kind"] =
      stringifySimdSimtCandidate(report.nominalDecision);
  summary["conservative_decision_kind"] =
      stringifySimdSimtCandidate(report.conservativeDecision);
  summary["transition_sensitive"] = report.transitionSensitive;
  summary["effective_decision_kind"] = effective;
  summary["selection_source"] = selectionSource;
  summary["application_reason"] = applicationReason;
  summary["action_supported"] = actionSupported;
  summary["selected_superblock_factor"] = selectedSuperblockFactor;
  summary["logical_program_count_hint"] = options.logicalProgramCountHint;
  summary["materialized_simt_anchor_count"] = materializedAnchorCount;
  summary["materialized_simt_stage_scope_count"] =
      materializationPlan
          ? static_cast<int64_t>(materializationPlan->ranges.size())
          : 0;
  if (auto error = writePrettyJSON(childPath(snapshotPath, "summary.json"),
                                   std::move(summary)))
    return error;

  llvm::json::Object features;
  features["applicability"] = report.applicability.toJSON();
  features["features"] = report.features.toJSON();
  if (auto error = writePrettyJSON(childPath(snapshotPath, "features.json"),
                                   std::move(features)))
    return error;

  llvm::json::Array discoveredStages;
  llvm::json::Array dependenceEdges;
  llvm::json::Array cuttableBoundaries;
  llvm::json::Array rejectedStages;
  llvm::json::Array *discoveryStages = nullptr;
  std::optional<llvm::json::Value> discovery;
  if (!report.stageModel.discoveryJSON.empty()) {
    auto parsed = llvm::json::parse(report.stageModel.discoveryJSON);
    if (!parsed)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "StageModel snapshot cannot parse Stage Boundary Graph JSON");
    discovery = std::move(*parsed);
    if (auto *graph = discovery->getAsObject()) {
      discoveryStages = graph->getArray("stages");
      if (auto *dependence = graph->getObject("dependence_graph")) {
        if (auto *stages = dependence->getArray("stages"))
          discoveredStages = std::move(*stages);
        if (auto *edges = dependence->getArray("edges"))
          dependenceEdges = std::move(*edges);
        if (auto *boundaries = dependence->getArray("cuttable_boundaries"))
          cuttableBoundaries = std::move(*boundaries);
      }
      if (auto *rejected = graph->getArray("rejected_stages"))
        rejectedStages = std::move(*rejected);
    }
  }

  llvm::json::Object stagesJSON;
  stagesJSON["count"] = static_cast<int64_t>(discoveredStages.size());
  stagesJSON["stages"] = std::move(discoveredStages);
  if (auto error = writePrettyJSON(childPath(snapshotPath, "stages.json"),
                                   std::move(stagesJSON)))
    return error;

  llvm::json::Object dependenceJSON;
  dependenceJSON["edge_count"] = static_cast<int64_t>(dependenceEdges.size());
  dependenceJSON["edges"] = std::move(dependenceEdges);
  dependenceJSON["cuttable_boundaries"] = std::move(cuttableBoundaries);
  if (auto error =
          writePrettyJSON(childPath(snapshotPath, "dependence-graph.json"),
                          std::move(dependenceJSON)))
    return error;

  llvm::json::Array stageIndex;
  llvm::json::Array stageFiles;
  for (auto indexedStage : llvm::enumerate(report.stageModel.stages)) {
    const LogicalStageCost &stage = indexedStage.value();
    std::string fileName = llvm::formatv("stage-{0}-{1}.json",
                                         stage.beginBoundary, stage.endBoundary)
                               .str();
    llvm::json::Object stageRecord;
    stageRecord["stage_index"] = static_cast<int64_t>(indexedStage.index());
    if (discoveryStages && indexedStage.index() < discoveryStages->size())
      stageRecord["discovery"] =
          std::move((*discoveryStages)[indexedStage.index()]);
    stageRecord["pricing"] = stage.toJSON();
    if (auto error = writePrettyJSON(childPath(stagePath, fileName),
                                     std::move(stageRecord)))
      return error;

    llvm::json::Object item;
    item["stage_index"] = static_cast<int64_t>(indexedStage.index());
    item["id"] = stage.id;
    item["begin_boundary"] = stage.beginBoundary;
    item["end_boundary"] = stage.endBoundary;
    item["model"] = stage.model;
    item["simd_legal"] = hasImplementation(stage, StageMode::SIMD);
    item["simt_legal"] = hasImplementation(stage, StageMode::SIMT);
    item["local_simt_materializable"] = stage.localSimtMaterializable;
    item["file"] = fileName;
    stageIndex.push_back(std::move(item));
    stageFiles.push_back((llvm::Twine("stages/") + fileName).str());
  }
  llvm::json::Object stageIndexJSON;
  stageIndexJSON["stage_count"] = static_cast<int64_t>(stageIndex.size());
  stageIndexJSON["stages"] = std::move(stageIndex);
  if (auto error = writePrettyJSON(childPath(stagePath, "index.json"),
                                   std::move(stageIndexJSON)))
    return error;

  llvm::json::Object boundaryGraph;
  boundaryGraph["applied"] = report.stageModel.applied;
  boundaryGraph["boundary_source"] = report.stageModel.boundarySource;
  boundaryGraph["boundary_count"] = report.stageModel.boundaryCount;
  boundaryGraph["operation_ownership_complete"] =
      report.stageModel.operationOwnershipComplete;
  boundaryGraph["modeled_operation_count"] =
      report.stageModel.modeledOperationCount;
  boundaryGraph["profile_version"] = report.stageModel.profileVersion;
  boundaryGraph["stage_count"] =
      static_cast<int64_t>(report.stageModel.stages.size());
  boundaryGraph["rejected_stages"] = std::move(rejectedStages);
  boundaryGraph["stages_file"] = "stages.json";
  boundaryGraph["dependence_graph_file"] = "dependence-graph.json";
  boundaryGraph["stage_index_file"] = "stages/index.json";
  if (auto error =
          writePrettyJSON(childPath(snapshotPath, "stage-boundary-graph.json"),
                          std::move(boundaryGraph)))
    return error;

  llvm::json::Object routes;
  routes["transition_cost"] = report.stageModel.transition.toJSON();
  routes["all_simd"] = report.stageModel.allSimd.toJSON();
  routes["all_simt_only"] = report.stageModel.allSimt.toJSON();
  routes["mixed_simd_simt"] = report.stageModel.mixed.toJSON();
  routes["mixed_simd_simt_conservative"] =
      report.stageModel.conservativeMixed.toJSON();
  routes["selected_route"] = recommended;
  if (auto error = writePrettyJSON(childPath(snapshotPath, "routes.json"),
                                   std::move(routes)))
    return error;

  llvm::json::Object materialization;
  materialization["effective_decision_kind"] = effective;
  materialization["plan_available"] = materializationPlan.has_value();
  materialization["applied"] =
      effective == kMixedSimdSimt && materializationPlan.has_value();
  materialization["plan"] =
      materializationPlan ? llvm::json::Value(materializationPlan->toJSON())
                          : llvm::json::Value(llvm::json::Object());
  materialization["nominal_mixed_plan_available"] =
      nominalMixedPlan.has_value();
  materialization["nominal_mixed_plan"] =
      nominalMixedPlan ? llvm::json::Value(nominalMixedPlan->toJSON())
                       : llvm::json::Value(llvm::json::Object());
  if (auto error =
          writePrettyJSON(childPath(snapshotPath, "materialization-plan.json"),
                          std::move(materialization)))
    return error;

  llvm::json::Array files({"summary.json", "config.json", "features.json",
                           "stages.json", "dependence-graph.json",
                           "stage-boundary-graph.json", "stages/index.json",
                           "routes.json", "materialization-plan.json"});
  for (llvm::json::Value &file : stageFiles)
    files.push_back(std::move(file));
  llvm::json::Object manifest;
  manifest["stage_model_snapshot_version"] = "3.0";
  manifest["complete"] = true;
  manifest["kernel_name"] = kernelName;
  manifest["ttir_sha256"] = ttirSha256;
  manifest["config_sha256"] = configSha256;
  manifest["files"] = std::move(files);
  return writePrettyJSON(manifestPath, std::move(manifest));
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
    options.logicalProgramCountHint =
        std::max<int64_t>(0, logicalProgramCountHint.getValue());
    options.routeTransformCapabilityJSON =
        routeTransformCapabilityJSON.getValue();
    auto capability = llvm::json::parse(options.routeTransformCapabilityJSON);
    if (!capability) {
      module.emitError("invalid route-transform-capability-json");
      signalPassFailure();
      return;
    }

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
    std::optional<StageMaterializationPlan> nominalMixedStagePlan;
    if (report.stageModel.boundarySource == "stage_boundary_graph" &&
        report.stageModel.mixed.legal) {
      auto plan = buildStageMaterializationPlan(report.stageModel);
      if (plan)
        nominalMixedStagePlan = std::move(*plan);
      else
        llvm::consumeError(plan.takeError());
    }
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
        if (!nominalMixedStagePlan) {
          actionSupported = false;
          applicationReason = "no_materializable_mixed_stage_plan";
        } else {
          selectedStagePlan = nominalMixedStagePlan;
          selectedMixedAnchorPlan =
              buildSelectedMixedAnchorPlan(report.stageModel, anchorPlan);
          mixedAnchors = selectedMixedAnchorPlan.materializableRoots();
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

    if (autoMode && actionSupported && report.transitionSensitive) {
      effective = kAllSimd.str();
      selectionSource = "cpp_cost_model_safe_fallback";
      applicationReason = "uncalibrated_scope_setup";
    } else if (autoMode && actionSupported) {
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
              : materializeSimtAnchorPlan(module, selectedMixedAnchorPlan,
                                          selectedSuperblockFactor);
      if (failed(materialized)) {
        signalPassFailure();
        return;
      }
    }

    llvm::json::Object reportJSON = report.toJSON();
    reportJSON["stage_model_snapshot_version"] = "3.0";
    reportJSON["ttir_sha256"] = ttirSha256;
    llvm::json::Object snapshotConfig;
    snapshotConfig["actual_target"] = options.actualTarget;
    snapshotConfig["num_warps"] = static_cast<int64_t>(options.numWarps);
    snapshotConfig["compile_on_91095"] = options.compileOn91095;
    snapshotConfig["whole_kernel_superblock_materializable"] =
        options.wholeKernelSuperblockMaterializable;
    snapshotConfig["scope_superblock_materializable"] =
        options.scopeSuperblockMaterializable;
    snapshotConfig["logical_program_count_hint"] =
        options.logicalProgramCountHint;
    snapshotConfig["route_transform_capability"] = std::move(*capability);
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
    if (nominalMixedStagePlan)
      reportJSON["nominal_mixed_stage_materialization_plan"] =
          nominalMixedStagePlan->toJSON();
    reportJSON["selected_superblock_factor"] = selectedSuperblockFactor;
    reportJSON["logical_program_count_hint"] = options.logicalProgramCountHint;
    if (options.logicalProgramCountHint > 0) {
      reportJSON["effective_runtime_factor"] = std::min<int64_t>(
          selectedSuperblockFactor, options.logicalProgramCountHint);
      reportJSON["full_group_count"] =
          options.logicalProgramCountHint / selectedSuperblockFactor;
      reportJSON["tail_count"] =
          options.logicalProgramCountHint % selectedSuperblockFactor;
    }
    std::string json =
        llvm::formatv("{0}", llvm::json::Value(std::move(reportJSON))).str();
    module->setAttr(kReportJSONAttr, builder.getStringAttr(json));

    if (auto error = writeStageModelSnapshot(
            reportFile.getValue(), module, ttirSha256, mode.getValue(), options,
            report, recommended, effective, selectionSource, applicationReason,
            actionSupported, selectedSuperblockFactor,
            static_cast<int64_t>(mixedAnchors.size()), selectedStagePlan,
            nominalMixedStagePlan))
      module.emitWarning("StageModel snapshot dump failed for `")
          << reportFile.getValue() << "`: " << llvm::toString(std::move(error));
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
