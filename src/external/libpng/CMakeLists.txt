project(Turi)

set(libpng_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/libpng-1.6.37/")

make_library(libpng
  SOURCES
${libpng_src_dir}/png.c
${libpng_src_dir}/pngerror.c
${libpng_src_dir}/pngget.c
${libpng_src_dir}/pngmem.c
${libpng_src_dir}/pngpread.c
${libpng_src_dir}/pngread.c
${libpng_src_dir}/pngrio.c
${libpng_src_dir}/pngrtran.c
${libpng_src_dir}/pngrutil.c
${libpng_src_dir}/pngset.c
${libpng_src_dir}/pngtest.c
${libpng_src_dir}/pngtrans.c
${libpng_src_dir}/pngwio.c
${libpng_src_dir}/pngwrite.c
${libpng_src_dir}/pngwtran.c
${libpng_src_dir}/pngwutil.c
REQUIRES
  z
  )

target_include_directories(libpng SYSTEM PUBLIC ${libpng_src_dir})

