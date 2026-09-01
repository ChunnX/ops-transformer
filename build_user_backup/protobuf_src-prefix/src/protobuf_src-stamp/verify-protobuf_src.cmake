# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if("/home/developer/codes/ops-transformer/third_party/protobuf/protobuf-25.1.tar.gz" STREQUAL "")
  message(FATAL_ERROR "LOCAL can't be empty")
endif()

if(NOT EXISTS "/home/developer/codes/ops-transformer/third_party/protobuf/protobuf-25.1.tar.gz")
  message(FATAL_ERROR "File not found: /home/developer/codes/ops-transformer/third_party/protobuf/protobuf-25.1.tar.gz")
endif()

if("SHA256" STREQUAL "")
  message(WARNING "File will not be verified since no URL_HASH specified")
  return()
endif()

if("9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42" STREQUAL "")
  message(FATAL_ERROR "EXPECT_VALUE can't be empty")
endif()

message(STATUS "verifying file...
     file='/home/developer/codes/ops-transformer/third_party/protobuf/protobuf-25.1.tar.gz'")

file("SHA256" "/home/developer/codes/ops-transformer/third_party/protobuf/protobuf-25.1.tar.gz" actual_value)

if(NOT "${actual_value}" STREQUAL "9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42")
  message(FATAL_ERROR "error: SHA256 hash of
  /home/developer/codes/ops-transformer/third_party/protobuf/protobuf-25.1.tar.gz
does not match expected value
  expected: '9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42'
    actual: '${actual_value}'
")
endif()

message(STATUS "verifying file... done")
