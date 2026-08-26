# The source-tree profile is a fallback for native tests and standalone C++
# users. Python production wiring passes the packaged profile path explicitly.
add_compile_definitions(
  TRITON_ASCEND_SIMD_SIMT_PROFILE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../../profiles/simd_simt/david_v100_stage_model_v1.json"
)

add_mlir_library(AscendModelStageModel
  SimtAnchorAnalysis.cpp
  StageAnalysis.cpp
  StageDiscovery.cpp
  StageCostModels.cpp
  StageRouteCostModel.cpp
  SimdSimtCostModel.cpp
  Transforms/SelectSimdSimtCostModel.cpp
  Transforms/MaterializeSimtScopes.cpp

  ADDITIONAL_HEADER_DIRS
  ${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}/include/AscendModel/StageModel

  DEPENDS
  AscendModelOpsIncGen
  AscendModelTransformsPassIncGen

  LINK_LIBS PUBLIC
  AscendModelProfile
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  LLVMSupport
)
