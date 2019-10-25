project(Turi)

set(NN_STATIC_LIB 1 CACHE BOOL "nanomsg static lib")
set(NN_TESTS 0 CACHE BOOL "nanomsg tests.")
set(NN_TOOLS 0 CACHE BOOL "nanomsg tools.")
set(NN_ENABLE_DOC 0 CACHE BOOL "nanomsg docs.")
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/nanomsg-1.1.5/cmake")


subdirs(nanomsg-1.1.5)


