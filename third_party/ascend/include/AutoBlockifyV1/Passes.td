#ifndef AUTO_BLOCKIFY_V1_PASSES
#define AUTO_BLOCKIFY_V1_PASSES

include "mlir/Pass/PassBase.td"

def TASIMTAutoBlockifyV1
    : Pass<"ta-simt-auto-blockify-v1", "mlir::triton::FuncOp"> {
  let summary = "Map logical Triton programs onto a fixed physical-core wave";
  let description = [{
    This is the Triton-Adapter port of the NPUIR SIMT AutoBlockify V1 pass.
    It keeps every logical program's tile shape unchanged and wraps the
    original program body in an outer scf.for.  AutoBlockify V2 is a separate
    pass that expands tile tensor shapes and is intentionally unaffected.
  }];
  let constructor = "mlir::triton::createTASIMTAutoBlockifyV1Pass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::gpu::GPUDialect",
    "mlir::scf::SCFDialect",
    "mlir::triton::TritonDialect"
  ];
  let options = [
    Option<"physicalVectorCoreCount", "physical-vector-core-count",
           "unsigned", "0",
           "Number of physical vector cores used by the launcher">,
    Option<"superBlockFactor", "superblock-factor", "unsigned", "1",
           "Power-of-two number of logical blocks sharing a super block">
  ];
}

def TARefineSIMTAutoBlockifyV1SuperBlock
    : Pass<"ta-refine-simt-auto-blockify-v1-superblock",
           "mlir::triton::FuncOp"> {
  let summary = "Refine an existing factor-1 V1 loop into a pure-SIMT SuperBlock loop";
  let description = [{
    StageModel intentionally observes the route-neutral factor-1 V1 loop.
    After it selects a pure-SIMT route, this pass refines that existing loop to
    the selected power-of-two SuperBlock factor.  It never recreates V1 and it
    does not touch AutoBlockify V2.
  }];
  let constructor =
      "mlir::triton::createTARefineSIMTAutoBlockifyV1SuperBlockPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::gpu::GPUDialect",
    "mlir::scf::SCFDialect",
    "mlir::triton::TritonDialect"
  ];
  let options = [
    Option<"superBlockFactor", "superblock-factor", "unsigned", "1",
           "Power-of-two number of logical programs sharing a physical program">
  ];
}

#endif // AUTO_BLOCKIFY_V1_PASSES
