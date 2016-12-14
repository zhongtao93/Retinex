cmake_minimum_required(VERSION 2.8.8)
PROJECT(hello) 
set(OPENCV_HOME /usr/local/opencv2.4.13)
include_directories(
	${OPENCV_HOME}/include
)
add_executable(retinex retinex.cpp)
target_link_libraries(retinex 
	${OPENCV_HOME}/lib/libopencv_core.so
	${OPENCV_HOME}/lib/libopencv_highgui.so
	${OPENCV_HOME}/lib/libopencv_imgproc.so
	${OPENCV_HOME}/lib/libopencv_ml.so
	${OPENCV_HOME}/lib/libopencv_objdetect.so

)

