cmake_minimum_required(VERSION 3.16)
project(GRADIENT_BASED_OPTIMIZER)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# Add PyTorch path if exists
if(EXISTS "/tmp/libtorch")
    list(APPEND CMAKE_PREFIX_PATH "/tmp/libtorch/share/cmake/Torch")
    message(STATUS "🔧 Added PyTorch path: /tmp/libtorch")
endif()

# Try to find PyTorch - optional for classifier integration
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "✅ PyTorch found - classifier integration enabled")
    set(TORCH_AVAILABLE TRUE)
    add_definitions(-DTORCH_AVAILABLE)
else()
    message(WARNING "⚠️  PyTorch not found - classifier integration disabled")
    message(WARNING "    To enable classifier: install PyTorch C++ (libtorch)")
    message(WARNING "    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip")
    set(TORCH_AVAILABLE FALSE)
endif()

set(SOURCES
    GRADIENT_BASED_OPTIMIZER/GRADIENT_BASED_OPTIMIZER.cpp
    GRADIENT_BASED_OPTIMIZER/src/attacks.cpp
    GRADIENT_BASED_OPTIMIZER/src/block_metrics.cpp
    GRADIENT_BASED_OPTIMIZER/src/gbo.cpp
    GRADIENT_BASED_OPTIMIZER/src/image_metrics.cpp
    GRADIENT_BASED_OPTIMIZER/src/image_processing_custom.cpp
    GRADIENT_BASED_OPTIMIZER/src/jpeg/compression.cpp
    GRADIENT_BASED_OPTIMIZER/src/launch.cpp
    GRADIENT_BASED_OPTIMIZER/src/population.cpp
    GRADIENT_BASED_OPTIMIZER/src/random_utils.cpp
    GRADIENT_BASED_OPTIMIZER/src/embedding_schemes.cpp
    GRADIENT_BASED_OPTIMIZER/src/dataset_generation.cpp
)

# Add classifier sources only if PyTorch is available
if(TORCH_AVAILABLE)
    list(APPEND SOURCES
        GRADIENT_BASED_OPTIMIZER/src/ensemble_classifier.cpp
        GRADIENT_BASED_OPTIMIZER/src/single_classifier.cpp
        GRADIENT_BASED_OPTIMIZER/src/embedding_with_classifier.cpp
        GRADIENT_BASED_OPTIMIZER/src/quadrant_classifier.cpp
        GRADIENT_BASED_OPTIMIZER/src/quadrant_embedding.cpp
        GRADIENT_BASED_OPTIMIZER/src/attack_type_classifier.cpp
        GRADIENT_BASED_OPTIMIZER/src/attack_type_embedding.cpp
    )
endif()

set(HEADERS
    GRADIENT_BASED_OPTIMIZER/src/attacks.h
    GRADIENT_BASED_OPTIMIZER/src/block_metrics.h
    GRADIENT_BASED_OPTIMIZER/src/config.h
    GRADIENT_BASED_OPTIMIZER/src/gbo.h
    GRADIENT_BASED_OPTIMIZER/src/image_metrics.h
    GRADIENT_BASED_OPTIMIZER/src/image_processing_custom.h
    GRADIENT_BASED_OPTIMIZER/src/jpeg/compression.h
    GRADIENT_BASED_OPTIMIZER/src/jpeg/quantization_tables.h
    GRADIENT_BASED_OPTIMIZER/src/launch.h
    GRADIENT_BASED_OPTIMIZER/src/population.h
    GRADIENT_BASED_OPTIMIZER/src/random_utils.h
    GRADIENT_BASED_OPTIMIZER/src/embedding_schemes.h
    GRADIENT_BASED_OPTIMIZER/src/dataset_generation.h
)

# Add classifier headers only if PyTorch is available
if(TORCH_AVAILABLE)
    list(APPEND HEADERS
        GRADIENT_BASED_OPTIMIZER/src/ensemble_classifier.h
        GRADIENT_BASED_OPTIMIZER/src/single_classifier.h
        GRADIENT_BASED_OPTIMIZER/src/embedding_with_classifier.h
        GRADIENT_BASED_OPTIMIZER/src/quadrant_classifier.h
        GRADIENT_BASED_OPTIMIZER/src/quadrant_embedding.h
        GRADIENT_BASED_OPTIMIZER/src/attack_type_classifier.h
        GRADIENT_BASED_OPTIMIZER/src/attack_type_embedding.h
    )
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(GRADIENT_BASED_OPTIMIZER/src)

add_executable(gradient_based_optimizer ${SOURCES} ${HEADERS})

# Link libraries
if(TORCH_AVAILABLE)
    target_link_libraries(gradient_based_optimizer ${OpenCV_LIBS} ${TORCH_LIBRARIES})
    set_target_properties(gradient_based_optimizer PROPERTIES CXX_STANDARD 14)
    
    # Add example executable only if PyTorch is available
    add_executable(classifier_example
        GRADIENT_BASED_OPTIMIZER/src/example_classifier_integration.cpp
        GRADIENT_BASED_OPTIMIZER/src/ensemble_classifier.cpp
        GRADIENT_BASED_OPTIMIZER/src/single_classifier.cpp
        GRADIENT_BASED_OPTIMIZER/src/embedding_with_classifier.cpp
        GRADIENT_BASED_OPTIMIZER/src/embedding_schemes.cpp
        GRADIENT_BASED_OPTIMIZER/src/image_processing_custom.cpp
        GRADIENT_BASED_OPTIMIZER/src/block_metrics.cpp
        GRADIENT_BASED_OPTIMIZER/src/gbo.cpp
        GRADIENT_BASED_OPTIMIZER/src/population.cpp
        GRADIENT_BASED_OPTIMIZER/src/random_utils.cpp
        GRADIENT_BASED_OPTIMIZER/src/attacks.cpp
    )

    target_link_libraries(classifier_example ${OpenCV_LIBS} ${TORCH_LIBRARIES})
    set_target_properties(classifier_example PROPERTIES CXX_STANDARD 14)

    # Add single classifier example executable
    add_executable(single_classifier_example
        GRADIENT_BASED_OPTIMIZER/src/example_single_classifier_integration.cpp
        GRADIENT_BASED_OPTIMIZER/src/ensemble_classifier.cpp
        GRADIENT_BASED_OPTIMIZER/src/single_classifier.cpp
        GRADIENT_BASED_OPTIMIZER/src/embedding_with_classifier.cpp
        GRADIENT_BASED_OPTIMIZER/src/embedding_schemes.cpp
        GRADIENT_BASED_OPTIMIZER/src/image_processing_custom.cpp
        GRADIENT_BASED_OPTIMIZER/src/block_metrics.cpp
        GRADIENT_BASED_OPTIMIZER/src/gbo.cpp
        GRADIENT_BASED_OPTIMIZER/src/population.cpp
        GRADIENT_BASED_OPTIMIZER/src/random_utils.cpp
        GRADIENT_BASED_OPTIMIZER/src/attacks.cpp
    )

    target_link_libraries(single_classifier_example ${OpenCV_LIBS} ${TORCH_LIBRARIES})
    set_target_properties(single_classifier_example PROPERTIES CXX_STANDARD 14)
    
    message(STATUS "🎯 Classifier example executables will be built")
else()
    target_link_libraries(gradient_based_optimizer ${OpenCV_LIBS})
    message(STATUS "🔧 Building without classifier integration")
endif()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")