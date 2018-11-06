cmake_minimum_required(VERSION 3.6)

set(target_name rclst)

file(GLOB_RECURSE SOURCES *.cpp *.h)

add_executable(${target_name} ${SOURCES})
target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/src)

set_target_properties(
        ${target_name} PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
)

target_link_libraries(${target_name} openblas dlib)

install(TARGETS ${target_name} RUNTIME DESTINATION bin)