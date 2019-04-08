project(TuriCreateObjC)

if(APPLE AND HAS_MLCUSTOMMODEL)
  find_library(FOUNDATION NAMES Foundation)
  message("Foundation found at ${FOUNDATION}.")

  find_library(CORE_ML NAMES CoreML)
  message("CoreML found at ${CORE_ML}.")

  make_library(Recommender
    SOURCES
      TuriCreateObjC.m
      recommender_initialization.cpp
    REQUIRES
      unity_recsys
      capi
      ${FOUNDATION}
      ${CORE_ML}
    SHARED
    SHARED_ALL_DEFINED
    EXPORT_OSX_MAP_FILE ${CMAKE_CURRENT_SOURCE_DIR}/symbol_exports.ver
    DEAD_STRIP
  )

  make_library(AudioPreprocessing
    SOURCES
      AudioPreprocessing.m
    REQUIRES
      ${ACCELERATE}
      ${_TC_COMMON_REQUIREMENTS}
    SHARED
    SHARED_ALL_DEFINED
    EXPORT_OSX_MAP_FILE ${CMAKE_CURRENT_SOURCE_DIR}/symbol_exports.ver
    DEAD_STRIP
  )

  # Adding sources here allows us to unit test these custom models in python.
  make_library(objcapi
    SOURCES
      AudioPreprocessing.m
      TuriCreateObjC.m
    REQUIRES
      TuriCreate_static
      ${FOUNDATION}
      ${CORE_ML}
  )

  target_compile_options(objcapi PUBLIC "-fobjc-arc")
  target_compile_options(Recommender PUBLIC "-fobjc-arc")
  target_compile_options(AudioPreprocessing PUBLIC "-fobjc-arc")

  if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_custom_command(TARGET Recommender
      POST_BUILD
      COMMAND strip -Sx "$<TARGET_FILE:Recommender>"
      COMMENT "Stripping Objective-C Recommender library"
    )
    add_custom_command(TARGET AudioPreprocessing
      POST_BUILD
      COMMAND strip -Sx "$<TARGET_FILE:AudioPreprocessing>"
      COMMENT "Stripping Objective-C AudioPreprocessing library"
    )
  endif()

endif()
