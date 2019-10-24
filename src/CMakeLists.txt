
subdirs(
  external
  capi
  core
  deployment
  ml
  model_server
  toolkits)

if(${TC_BUILD_VISULIZATION_CLIENT})
  subdirs(visualization)
endif()


if(TC_BUILD_PYTHON)
  subdirs(python)

  include_directories(${PYTHON_INCLUDE_DIRS})
  message("Python library = ${PYTHON_LIBRARIES}")

endif()

