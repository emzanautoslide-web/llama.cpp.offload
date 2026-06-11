if (NOT DEFINED ORACLE_EXE OR NOT DEFINED ORACLE_MODEL)
    message(FATAL_ERROR "ORACLE_EXE and ORACLE_MODEL are required")
endif()

execute_process(
    COMMAND "${ORACLE_EXE}"
        --model "${ORACLE_MODEL}"
        --moe-oracle
        -p test
        -n 1
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)

set(combined "${out}\n${err}")

if (rc EQUAL 0)
    message(FATAL_ERROR "--moe-oracle unexpectedly succeeded")
endif()

if (NOT combined MATCHES "moe-oracle is deferred to post-MVP")
    message(FATAL_ERROR "--moe-oracle failed without the expected post-MVP message:\n${combined}")
endif()

message(STATUS "--moe-oracle failed fast as expected")
