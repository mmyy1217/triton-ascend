add_triton_library(ScopeProfile
  ScopeProfile.cpp

  DEPENDS
  ScopeProfilePassIncGen

  LINK_LIBS PUBLIC
  BiShengIRScopeDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  TritonIR
)
