
cmake_minimum_required(VERSION 3.18 REQUIRED)
project(ensemble_classifier)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(ensemble_classifier 
    ensemble_classifier.cpp
    main.cpp
)

target_link_libraries(ensemble_classifier 
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

set_property(TARGET ensemble_classifier PROPERTY CXX_STANDARD 14)

# Копирование моделей в папку сборки
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/best_scheme_classifier_torchscript.pt"
               "${CMAKE_CURRENT_BINARY_DIR}/best_scheme_classifier_torchscript.pt" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ensemble_model_1_torchscript.pt" 
               "${CMAKE_CURRENT_BINARY_DIR}/ensemble_model_1_torchscript.pt" COPYONLY)
