CMAKE_MINIMUM_REQUIRED(VERSION 3.5.0 FATAL_ERROR)

PROJECT(bitsery-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(bitsery
	GIT_REPOSITORY https://github.com/fraillt/bitsery
	GIT_TAG master
	SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/bitsery"
	BINARY_DIR "${DNN_DEPENDENCIES_BINARY_DIR}/bitsery"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	TEST_COMMAND ""
)
