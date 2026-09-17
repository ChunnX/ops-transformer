# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Called only by the MTE build; also executable with cmake -P for CPU checks.
if(NOT EXISTS "${CATCCOS_MTE_SHMEM_LIBRARY}")
    message(FATAL_ERROR "MTE SHMEM export check: library not found: ${CATCCOS_MTE_SHMEM_LIBRARY}")
endif()
find_program(CATCCOS_MTE_NM NAMES llvm-nm nm)
if(NOT CATCCOS_MTE_NM)
    message(FATAL_ERROR "MTE requires llvm-nm or nm to verify the pinned SHMEM bootstrap export")
endif()
execute_process(
    COMMAND "${CATCCOS_MTE_NM}" -D --defined-only "${CATCCOS_MTE_SHMEM_LIBRARY}"
    RESULT_VARIABLE CATCCOS_MTE_NM_STATUS
    OUTPUT_VARIABLE CATCCOS_MTE_SHMEM_EXPORTS
    ERROR_VARIABLE CATCCOS_MTE_NM_ERROR
)
if(NOT CATCCOS_MTE_NM_STATUS EQUAL 0)
    message(FATAL_ERROR
        "MTE could not inspect SHMEM dynamic exports: ${CATCCOS_MTE_NM_ERROR}")
endif()
if(NOT CATCCOS_MTE_SHMEM_EXPORTS MATCHES
        "(^|\n)[^\n]*[ \t][BDGSRV][ \t]+g_boot_handle(@[^ \t\r\n]+)?[ \t]*(\r?\n|$)")
    message(FATAL_ERROR
        "MTE requires the pinned SHMEM library to export g_boot_handle. "
        "Rebuild the pinned dependency without hidden bootstrap symbols; no backend fallback is allowed.")
endif()
