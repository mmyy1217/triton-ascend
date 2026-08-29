#ifndef TRITON_ADAPTER_SCOPE_PROFILE_H
#define TRITON_ADAPTER_SCOPE_PROFILE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DECL_TASCOPEPROFILE
#include "ascend/include/ScopeProfile/Passes.h.inc"

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createTAScopeProfilePass(const TAScopeProfileOptions &options = {});

} // namespace mlir::triton

#endif
