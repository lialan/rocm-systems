# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# rj_isa_diff core, region executor: run a recorded instruction sequence on a standalone
# wave from a given machine state and capture the resulting state.
add_library(
    rj_isa_diff_core
    STATIC
    effects.cpp
    json.cpp
    location.cpp
    machine_state.cpp
    region_runner.cpp
    target.cpp
)
target_include_directories(rj_isa_diff_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# The execution-capable ISA registry (via RJ_TOOL_OBJECT_LIBS) is required: the
# model-only registry leaves Instruction::execute null.
target_link_libraries(
    rj_isa_diff_core
    PUBLIC ${RJ_TOOL_OBJECT_LIBS} util Threads::Threads
)
rj_configure_target(rj_isa_diff_core PRIVATE_INCLUDES WARNINGS)

if(BUILD_TESTING)
    include(GoogleTest)
    add_executable(
        rj_isa_diff_tests
        ${PROJECT_SOURCE_DIR}/tests/isa_diff/region_executor_test.cpp
    )
    target_include_directories(
        rj_isa_diff_tests
        PRIVATE ${PROJECT_SOURCE_DIR}/tests/isa_diff
    )
    target_link_libraries(
        rj_isa_diff_tests
        PRIVATE rj_isa_diff_core GTest::gtest_main
    )
    rj_configure_target(rj_isa_diff_tests TEST)
    rj_add_gtest_tests(rj_isa_diff_tests TEST_PREFIX "isa_diff.")
endif()
