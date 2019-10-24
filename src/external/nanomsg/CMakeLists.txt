project(Turi)

set(NN_STATIC_LIB 1)
set(NN_TESTS 0)
set(NN_TOOLS 0)
set(NN_ENABLE_DOC 0)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/nanomsg-1.1.5/cmake")


subdirs(nanomsg-1.1.5)


