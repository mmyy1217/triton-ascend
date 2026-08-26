//===- StageDiscovery.h - Generic Stage boundary discovery ------*- C++ -*-===//

#ifndef ASCENDMODEL_STAGEMODEL_STAGEDISCOVERY_H
#define ASCENDMODEL_STAGEMODEL_STAGEDISCOVERY_H

#include "AscendModel/StageModel/SimtAnchorAnalysis.h"
#include "AscendModel/StageModel/StageCostModels.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlir::ascend {

enum class StageDependenceKind { SSA, Memory, Control, LoopCarried };

llvm::StringRef stringifyStageDependenceKind(StageDependenceKind kind);

struct StageDependenceEdge {
  size_t source = 0;
  size_t target = 0;
  StageDependenceKind kind = StageDependenceKind::SSA;
  std::string detail;

  llvm::json::Object toJSON() const;
};

struct StageSemanticUnit {
  size_t index = 0;
  Operation *operation = nullptr;
  bool anchorEvidence = false;
  bool seedEvidence = false;
  std::string evidence;

  llvm::json::Object toJSON() const;
};

struct StageDependenceGraph {
  std::vector<StageSemanticUnit> units;
  std::vector<StageDependenceEdge> edges;
  std::vector<bool> cuttableBoundaries;

  bool canCut(size_t boundary) const;
  llvm::json::Object toJSON() const;
};

struct CandidateStage {
  size_t index = 0;
  size_t beginBoundary = 0;
  size_t endBoundary = 0;
  bool fallback = false;
  std::vector<std::string> evidence;
  std::string localSimtRejectionReason;
  LogicalStage stage;

  llvm::json::Object toJSON() const;
};

struct RejectedCandidateStage {
  size_t beginBoundary = 0;
  size_t endBoundary = 0;
  std::string reason;

  llvm::json::Object toJSON() const;
};

struct StageBoundaryGraph {
  std::string boundarySource = "stage_boundary_graph";
  StageDependenceGraph dependenceGraph;
  std::vector<CandidateStage> candidates;
  std::vector<RejectedCandidateStage> rejectedCandidates;

  size_t boundaryCount() const { return dependenceGraph.units.size() + 1; }
  llvm::json::Object toJSON() const;
};

struct StageDiscoveryOptions {
  int64_t tinyDotFlopsMax = 16384;
  int64_t maximumSuperblockFactor = 1;
  size_t maximumCandidateCount = 1024;
};

class StageDependenceAnalysis {
public:
  llvm::Expected<StageDependenceGraph>
  analyze(ModuleOp module, const SimtAnchorPlan &anchorPlan) const;
};

class StageDiscovery {
public:
  llvm::Expected<StageBoundaryGraph>
  discover(ModuleOp module, const SimtAnchorPlan &anchorPlan,
           const StageDiscoveryOptions &options) const;
};

struct SimtStageRange {
  size_t beginBoundary = 0;
  size_t endBoundary = 0;
  std::vector<Operation *> operations;

  llvm::json::Object toJSON() const;
};

struct StageMaterializationPlan {
  std::vector<SimtStageRange> ranges;

  llvm::json::Object toJSON() const;
};

llvm::Expected<StageMaterializationPlan>
buildStageMaterializationPlan(const StageCostModelSummary &stageModel);

LogicalResult materializeSimtStagePlan(ModuleOp module,
                                       const StageMaterializationPlan &plan);

} // namespace mlir::ascend

#endif // ASCENDMODEL_STAGEMODEL_STAGEDISCOVERY_H
