set(TARGET llama-moe-bench)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()