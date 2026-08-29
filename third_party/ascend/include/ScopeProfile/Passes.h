#ifndef TRITON_ADAPTER_SCOPE_PROFILE_PASSES_H
#define TRITON_ADAPTER_SCOPE_PROFILE_PASSES_H

#include "ScopeProfile.h"

namespace mlir::triton {

#define GEN_PASS_REGISTRATION
#include "ascend/include/ScopeProfile/Passes.h.inc"

} // namespace mlir::triton

#endif
