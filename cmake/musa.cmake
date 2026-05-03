IF (TENGINE_ENABLE_MUSA)
    SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/modules)

    find_path(MUSA_TOOLKIT_ROOT
        NAMES include/musa.h
        PATHS ENV MUSA_HOME /usr/local/musa
        DOC "MUSA SDK root directory"
    )

    if(MUSA_TOOLKIT_ROOT)
        SET (MUSA_FOUND TRUE)
        SET (MUSA_INCLUDE_DIRS "${MUSA_TOOLKIT_ROOT}/include")

        find_program(MUSA_MCC_EXECUTABLE NAMES mcc
            PATHS "${MUSA_TOOLKIT_ROOT}/bin" DOC "MUSA mcc compiler")

        find_library(MUSA_RUNTIME_LIBRARY NAMES musart
            PATHS "${MUSA_TOOLKIT_ROOT}/lib" DOC "MUSA runtime library")
        find_library(MUSA_DRIVER_LIBRARY NAMES musa
            PATHS "${MUSA_TOOLKIT_ROOT}/lib" DOC "MUSA driver library")
        find_library(MUBLAS_LIBRARY NAMES mublas
            PATHS "${MUSA_TOOLKIT_ROOT}/lib" DOC "muBLAS library")
        find_library(MUDNN_LIBRARY NAMES mudnn
            PATHS "${MUSA_TOOLKIT_ROOT}/lib" DOC "muDNN library")

        SET (MUSA_LIBRARIES ${MUSA_RUNTIME_LIBRARY} ${MUSA_DRIVER_LIBRARY})

        MESSAGE (STATUS "MUSA SDK: ${MUSA_TOOLKIT_ROOT}, mcc: ${MUSA_MCC_EXECUTABLE}")
    else()
        SET (MUSA_FOUND FALSE)
        MESSAGE (FATAL_ERROR "Tengine: MUSA SDK not found. Set MUSA_HOME.")
    endif()
ENDIF()
