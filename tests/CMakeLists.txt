include_directories(../include)

add_executable(tests test.cpp)

target_link_libraries(
  tests
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(tests)