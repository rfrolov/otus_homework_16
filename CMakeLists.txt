cmake_minimum_required(VERSION 3.6)

if (NOT DEFINED ENV{TRAVIS_BUILD_NUMBER})
    set(ENV{TRAVIS_BUILD_NUMBER} 1)
endif ()

project(realty VERSION 1.0.$ENV{TRAVIS_BUILD_NUMBER})

add_subdirectory(dlib EXCLUDE_FROM_ALL)
add_subdirectory(rclss)
add_subdirectory(rclst)

set(CPACK_GENERATOR DEB)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VENDOR "rfrolov")
set(CPACK_PACKAGE_CONTACT rfrolov@engineer.com)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Otus homework 16")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_BUILD $ENV{TRAVIS_BUILD_NUMBER})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})

include(CPack)