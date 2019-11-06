if(NOT ${TC_BUILD_REMOTEFS})
  make_empty_library(openssl)
  return()
endif()

# WARNING:  OPENSSL DOES NOT SUPPORT PARALLEL BUILDS. make -j1 MUST BE USED!!!!!


if(APPLE)
  # SSL seems to link fine even when compiled using the default compiler
  # The alternative to get openssl to use gcc on mac requires a patch to
  # the ./Configure script
ExternalProject_Add(ex_libssl
  PREFIX ${CMAKE_SOURCE_DIR}/deps/build/libssl
  URL ${CMAKE_SOURCE_DIR}/deps/src/openssl-1.0.2t 
  INSTALL_DIR ${CMAKE_SOURCE_DIR}/deps/local
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND env SDKROOT=${CMAKE_OSX_SYSROOT} CC="${CMAKE_C_COMPILER}" CFLAGS="${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} -Wno-everything" ./Configure darwin64-x86_64-cc no-rc5 -fPIC --prefix=<INSTALL_DIR>
  BUILD_COMMAND bash -c "SDKROOT=${CMAKE_OSX_SYSROOT} make -j1"
  INSTALL_COMMAND bash -c "SDKROOT=${CMAKE_OSX_SYSROOT} make -j1 install && cp ./libcrypto.a <INSTALL_DIR>/ssl && cp ./libssl.a <INSTALL_DIR>/ssl"
  BUILD_BYPRODUCTS ${CMAKE_SOURCE_DIR}/deps/local/lib/libssl.a ${CMAKE_SOURCE_DIR}/deps/local/lib/libcrypto.a
  )
elseif(WIN32)
ExternalProject_Add(ex_libssl
  PREFIX ${CMAKE_SOURCE_DIR}/deps/build/libssl
  URL ${CMAKE_SOURCE_DIR}/deps/src/openssl-1.0.2t 
  INSTALL_DIR ${CMAKE_SOURCE_DIR}/deps/local
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ./Configure mingw64 no-idea no-mdc2 no-rc5 --prefix=<INSTALL_DIR>
  BUILD_COMMAND make depend && make -j1
  INSTALL_COMMAND make -j1 install_sw
  )
else()
ExternalProject_Add(ex_libssl
  PREFIX ${CMAKE_SOURCE_DIR}/deps/build/libssl
  URL ${CMAKE_SOURCE_DIR}/deps/src/openssl-1.0.2t
  INSTALL_DIR ${CMAKE_SOURCE_DIR}/deps/local
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND env CC=${CMAKE_C_COMPILER} CFLAGS="${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} -Wno-everything" ./config no-rc5 -fPIC --prefix=<INSTALL_DIR>
  BUILD_COMMAND make -j1
  INSTALL_COMMAND make -j1 install_sw
  BUILD_BYPRODUCTS ${CMAKE_SOURCE_DIR}/deps/local/lib/libssl.a ${CMAKE_SOURCE_DIR}/deps/local/lib/libcrypto.a
  )
endif()

add_library(libssla STATIC IMPORTED)
set_property(TARGET libssla PROPERTY IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/deps/local/lib/libssl.a)

add_library(libcryptoa STATIC IMPORTED)
set_property(TARGET libcryptoa PROPERTY IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/deps/local/lib/libcrypto.a)

add_library(openssl INTERFACE )
target_link_libraries(openssl INTERFACE libssla libcryptoa)
if(NOT WIN32)
  target_link_libraries(openssl INTERFACE dl)
endif()

target_compile_definitions(openssl INTERFACE HAS_OPENSSL)

add_dependencies(openssl libssla ex_libssl)
add_dependencies(libssla ex_libssl)
add_dependencies(libcryptoa ex_libssl)
set(HAS_OPENSSL TRUE CACHE BOOL "")
