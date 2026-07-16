find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Found Python3: ${Python3_EXECUTABLE} (version ${Python3_VERSION})")
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")
