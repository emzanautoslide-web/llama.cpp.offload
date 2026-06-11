set(TARGET llama-moe-bench)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_MOE_OFFLOAD)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_MOE_OFFLOAD)
endif()

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE psapi)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()