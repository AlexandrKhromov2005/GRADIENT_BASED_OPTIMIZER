cmake_minimum_required(VERSION 3.10)
project(GRADIENT_BASED_OPTIMIZER)

set(CMAKE_CXX_STANDARD 20)

# Находим пакеты OpenCV и TBB
find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)

# Указываем директории включения для OpenCV и TBB
include_directories(${OpenCV_INCLUDE_DIRS} include)

# Добавляем исходные файлы
file(GLOB SOURCES "src/*.cpp")

# Создаем исполняемый файл
add_executable(${PROJECT_NAME} ${SOURCES})

# Связываем библиотеки OpenCV и TBB с исполняемым файлом
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} TBB::tbb)