project(Turi)

if(APPLE AND HAS_COREML_CUSTOM_MODEL)
  message(STATUS "Enabling Recommender and AudioPreprocessing targets.")

  make_library(Recommender
    SOURCES
      TuriCreateObjC.mm
      recommender_initialization.cpp
      ${_TC_COMMON_OBJECTS}
      $<TARGET_OBJECTS:capi>
    REQUIRES
      ${_TC_COMMON_REQUIREMENTS}
    SHARED
    SHARED_ALL_DEFINED
    EXPORT_OSX_MAP_FILE "${CMAKE_CURRENT_SOURCE_DIR}/symbol_exports.ver"
    DEAD_STRIP
  )

  make_library(AudioPreprocessing
    SOURCES
      AudioPreprocessing.mm
      $<TARGET_OBJECTS:capi>
    REQUIRES
      ${ACCELERATE}
      ${_TC_COMMON_REQUIREMENTS}
    SHARED
    SHARED_ALL_DEFINED
    EXPORT_OSX_MAP_FILE ${CMAKE_CURRENT_SOURCE_DIR}/symbol_exports.ver
    DEAD_STRIP
  )

  # Adding sources here allows us to unit test these custom models in python.
  make_library(objcapi OBJECT
    SOURCES
      AudioPreprocessing.mm
      TuriCreateObjC.mm
      $<TARGET_OBJECTS:capi>
    REQUIRES
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
else()
  message(STATUS "Skipping Recommender and AudioPreprocessing targets.")
endif()
