//===- StageAnalysis.h - Operation-derived Stage analysis ------*- C++ -*-===//

#ifndef ASCENDMODEL_STAGEMODEL_STAGEANALYSIS_H
#define ASCENDMODEL_STAGEMODEL_STAGEANALYSIS_H

#include "AscendModel/StageModel/StageCostModels.h"

#include "llvm/Support/Error.h"

#include <cstdint>

namespace mlir::ascend {

class StageWorkloadAnalysis {
public:
  llvm::Error analyze(LogicalStage &stage) const;
};

class StageFeatureAnalysis {
public:
  llvm::Error analyze(LogicalStage &stage) const;
};

class StageKindClassifier {
public:
  llvm::Error analyze(LogicalStage &stage, int64_t tinyDotFlopsMax) const;
};

} // namespace mlir::ascend

#endif // ASCENDMODEL_STAGEMODEL_STAGEANALYSIS_H
