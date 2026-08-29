#ifndef TRITON_ADAPTER_SCOPE_PROFILE_PASSES
#define TRITON_ADAPTER_SCOPE_PROFILE_PASSES

include "mlir/Pass/PassBase.td"

def TAScopeProfile : Pass<"ta-scope-profile", "mlir::ModuleOp"> {
  let summary = "Enumerate or materialize one contiguous SIMT scope";
  let constructor = "mlir::triton::createTAScopeProfilePass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::scope::ScopeDialect",
    "mlir::triton::TritonDialect"
  ];
  let options = [
    Option<"planId", "plan-id", "int64_t", "-1",
           "-1 discovers plans; non-negative values materialize one plan">
  ];
}

#endif
