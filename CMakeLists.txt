cmake_minimum_required(VERSION 3.16)
project(GBO VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- User-facing build options ----------------------------------------------
option(GBO_BUILD_APP       "Build the standalone CLI tool (gbo_app)"   ON)
option(GBO_BUILD_BENCH     "Build the test-bench binary (gbo_bench)"   OFF)
option(GBO_BUILD_SHARED    "Build a shared library (libgbo.so/.dylib)" OFF)
option(GBO_BUILD_STATIC    "Build a static library (libgbo.a)"         OFF)

# ---- Dependencies -----------------------------------------------------------
find_package(OpenCV REQUIRED)

# PyTorch (optional) — auto-detected in standard paths + /tmp/libtorch
if(EXISTS "/tmp/libtorch")
    list(APPEND CMAKE_PREFIX_PATH "/tmp/libtorch/share/cmake/Torch")
endif()

find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "PyTorch found - classifier integration enabled")
    set(TORCH_AVAILABLE TRUE)
    add_definitions(-DTORCH_AVAILABLE)
else()
    message(STATUS "PyTorch not found - classifier integration disabled (optional)")
    set(TORCH_AVAILABLE FALSE)
endif()

# ---- Source lists -----------------------------------------------------------
set(SRC_DIR GRADIENT_BASED_OPTIMIZER/src)

set(GBO_CORE_SOURCES
    ${SRC_DIR}/gbo.cpp
    ${SRC_DIR}/population.cpp
    ${SRC_DIR}/attacks.cpp
    ${SRC_DIR}/block_metrics.cpp
    ${SRC_DIR}/image_metrics.cpp
    ${SRC_DIR}/image_processing_custom.cpp
    ${SRC_DIR}/jpeg/compression.cpp
    ${SRC_DIR}/random_utils.cpp
    ${SRC_DIR}/embedding_schemes.cpp
    ${SRC_DIR}/gbo_api.cpp
)

set(GBO_BENCH_SOURCES
    ${SRC_DIR}/launch.cpp
    ${SRC_DIR}/dataset_generation.cpp
)

if(TORCH_AVAILABLE)
    list(APPEND GBO_CORE_SOURCES
        ${SRC_DIR}/ensemble_classifier.cpp
        ${SRC_DIR}/single_classifier.cpp
        ${SRC_DIR}/embedding_with_classifier.cpp
    )
    list(APPEND GBO_BENCH_SOURCES
        ${SRC_DIR}/quadrant_classifier.cpp
        ${SRC_DIR}/quadrant_embedding.cpp
        ${SRC_DIR}/attack_type_classifier.cpp
        ${SRC_DIR}/attack_type_embedding.cpp
    )
endif()

set(GBO_PUBLIC_HEADERS
    ${SRC_DIR}/gbo_api.h
)

# Collect all internal headers for IDE convenience
file(GLOB GBO_INTERNAL_HEADERS ${SRC_DIR}/*.h ${SRC_DIR}/jpeg/*.h)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${SRC_DIR})

# ---- Helper: link the right libs to a target --------------------------------
function(gbo_link_libs target)
    if(TORCH_AVAILABLE)
        target_link_libraries(${target} ${OpenCV_LIBS} ${TORCH_LIBRARIES})
    else()
        target_link_libraries(${target} ${OpenCV_LIBS})
    endif()
endfunction()

# ---- Shared library ---------------------------------------------------------
if(GBO_BUILD_SHARED)
    add_library(gbo_shared SHARED ${GBO_CORE_SOURCES} ${GBO_INTERNAL_HEADERS})
    set_target_properties(gbo_shared PROPERTIES
        OUTPUT_NAME gbo
        VERSION     ${PROJECT_VERSION}
        SOVERSION   1
    )
    target_compile_definitions(gbo_shared PRIVATE GBO_BUILDING_LIBRARY)
    gbo_link_libs(gbo_shared)
    message(STATUS "Target: libgbo.so (shared library)")
endif()

# ---- Static library ---------------------------------------------------------
if(GBO_BUILD_STATIC)
    add_library(gbo_static STATIC ${GBO_CORE_SOURCES} ${GBO_INTERNAL_HEADERS})
    set_target_properties(gbo_static PROPERTIES
        OUTPUT_NAME gbo
    )
    gbo_link_libs(gbo_static)
    message(STATUS "Target: libgbo.a (static library)")
endif()

# ---- Standalone CLI app -----------------------------------------------------
if(GBO_BUILD_APP)
    add_executable(gbo_app
        GRADIENT_BASED_OPTIMIZER/gbo_app.cpp
        ${GBO_CORE_SOURCES}
        ${GBO_INTERNAL_HEADERS}
    )
    gbo_link_libs(gbo_app)
    message(STATUS "Target: gbo_app (standalone CLI)")
endif()

# ---- Test bench (full experiment runner) ------------------------------------
if(GBO_BUILD_BENCH)
    add_executable(gbo_bench
        GRADIENT_BASED_OPTIMIZER/GRADIENT_BASED_OPTIMIZER.cpp
        ${GBO_CORE_SOURCES}
        ${GBO_BENCH_SOURCES}
        ${GBO_INTERNAL_HEADERS}
    )
    gbo_link_libs(gbo_bench)
    message(STATUS "Target: gbo_bench (test bench)")

    # Classifier example binaries (only with PyTorch)
    if(TORCH_AVAILABLE)
        add_executable(classifier_example
            ${SRC_DIR}/example_classifier_integration.cpp
            ${GBO_CORE_SOURCES}
        )
        gbo_link_libs(classifier_example)

        add_executable(single_classifier_example
            ${SRC_DIR}/example_single_classifier_integration.cpp
            ${GBO_CORE_SOURCES}
        )
        gbo_link_libs(single_classifier_example)
        message(STATUS "Target: classifier_example, single_classifier_example")
    endif()
endif()

# ---- Install ----------------------------------------------------------------
include(GNUInstallDirs)

# Always install public header
install(FILES ${GBO_PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gbo
)

# Install libraries
if(GBO_BUILD_SHARED)
    install(TARGETS gbo_shared
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()
if(GBO_BUILD_STATIC)
    install(TARGETS gbo_static
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

# Install executables
if(GBO_BUILD_APP)
    install(TARGETS gbo_app RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
if(GBO_BUILD_BENCH)
    install(TARGETS gbo_bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# Install scheme config
install(FILES embedding_schemes.json
    DESTINATION ${CMAKE_INSTALL_DATADIR}/gbo
)

# ---- Build-type default -----------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
