project(TuriCreate_ml)

# If this variable is specified, then we pull the coremltools
#from that repo to compile in.It should just work here.
file(GLOB_RECURSE _ml_src_files
  RELATIVE
  ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS
  LIST_DIRECTORIES false
  *.cpp
  )

make_library(ml
  OBJECT
  SOURCES
    ${_ml_src_files}
  EXTERNAL_VISIBILITY
  )
