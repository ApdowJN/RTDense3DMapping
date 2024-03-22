cmake_minimum_required(VERSION 3.10)
project(IROS)
SET(CMAKE_BUILD_TYPE Release)
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_STANDARD 11)

message("Build type: " ${CMAKE_BUILD_TYPE})

if( CMAKE_BUILD_TYPE MATCHES "Debug" )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -pthread -O3 -g -Wall -fmessage-length=0 -msse4 -fopenmp -ftree-vectorize -std=c++11 -fPIC -MMD -MP")
else( CMAKE_BUILD_TYPE MATCHES "Release" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O3 -g3 -Wall -fmessage-length=0 -msse4 -fopenmp -ftree-vectorize -std=c++11 -fPIC -MMD -MP")
endif( CMAKE_BUILD_TYPE MATCHES "Debug" )


#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fmessage-length=0 -mavx2 -fopenmp -ftree-vectorize -fopt-info -std=c++11 -fPIC -MMD -MP")
#set(CMAKE_C_FLAGS_DEBUG  "-pthread -O3 -g3 -Wall -fmessage-length=0 -msse4 -fopenmp -ftree-vectorize -fopt-info -std=c++11 -fPIC -MMD -MP")
#set(CMAKE_CXX_FLAGS_DEBUG  "-pthread -O3 -g3 -Wall -fmessage-length=0 -msse4 -fopenmp -ftree-vectorize -std=c++11 -fPIC -MMD -MP") #-fopt-info



LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(OpenCV REQUIRED)

find_package(Eigen3 3.1.0 REQUIRED)
#find_package(Pangolin REQUIRED)
find_package(OpenMP)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()
# pcl
find_package(PCL REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/sgm
        ${EIGEN3_INCLUDE_DIR}
        #${Pangolin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

# For real-time stereo module
#include_directories(Real-Time-Stereo/libs)
#include_directories(Real-Time-Stereo/CPU/sgm)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
        src/util.cpp
        src/Frame.cpp
        src/stereo.cpp
		src/fusion.cpp
        src/pointCloudBuilder.cpp
        src/sgm/FastFilters.cpp
        src/sgm/StereoBMHelper.cpp
        src/planesweep.cpp
        )

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${PCL_LIBRARIES}
        OpenMP::OpenMP_CXX
        )

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/)
# add_executable(depth_fusion src/main.cpp)
# target_link_libraries(depth_fusion ${PROJECT_NAME})

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/)
# add_executable(depth_fusion2 src/main2.cpp)
# target_link_libraries(depth_fusion2 ${PROJECT_NAME})

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/)
#add_executable(accelerated_depth_fusion src/accelerated_version_main.cpp)
#target_link_libraries(accelerated_depth_fusion ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/)
add_executable(plane_sweep_depth_fusion src/plane_sweep_main.cpp)
target_link_libraries(plane_sweep_depth_fusion ${PROJECT_NAME})




