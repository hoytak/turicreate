project(unity_toolkits)

make_library(unity_coreml_model_export
  SOURCES
    mldata_exporter.cpp
    xgboost_exporter.cpp
    linear_models_exporter.cpp
    neural_net_models_exporter.cpp
    coreml_export_utils.cpp
    mlmodel_wrapper.cpp
  REQUIRES
    eigen
    protobuf
    coremltools_mlmodel
  COMPILE_FLAGS_EXTRA
    -Wno-return-type              # TODO: resolve these issues at the source level
    -Wno-sign-compare             # TODO: resolve these issues at the source level
    -Wno-unused-function          # TODO: resolve these issues at the source level
    -Wno-unused-local-typedef     # TODO: resolve these issues at the source level
    -Wno-unused-variable          # TODO: resolve these issues at the source level
  COMPILE_FLAGS_EXTRA_CLANG
    -Wno-format                   # TODO: resolve these issues at the source level
  COMPILE_FLAGS_EXTRA_GCC
    -Wno-format-contains-nul      # TODO: resolve these issues at the source level
    -Wno-maybe-uninitialized      # TODO: resolve these issues at the source level
    -Wno-unused-but-set-variable  # TODO: resolve these issues at the source level
    -Wno-unknown-pragmas          # NOTE: used for auto-generated protobuf source files
    -Wno-unused-function          # NOTE: used for auto-generated protobuf source files
  )

target_compile_definitions(protobuf PUBLIC CoreML=__tc_CoreML)

