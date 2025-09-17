# Copyright (c) 2025-present The Bitcoin Core developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or https://opensource.org/license/mit/.

if(MSVC)
  option(WITH_SCCACHE "Use sccache for MSVC builds" ON)

  if(WITH_SCCACHE)
    find_program(SCCACHE_EXECUTABLE sccache)
    if(SCCACHE_EXECUTABLE)
      list(APPEND CMAKE_C_COMPILER_LAUNCHER ${SCCACHE_EXECUTABLE})
      list(APPEND CMAKE_CXX_COMPILER_LAUNCHER ${SCCACHE_EXECUTABLE})

      # Handle debug information format for sccache compatibility
      if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
        set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)
        cmake_policy(SET CMP0141 NEW)
      else()
        # For CMake < 3.25, use /Z7 flag replacement
        foreach(config DEBUG RELEASE RELWITHDEBINFO)
          string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_${config} "${CMAKE_CXX_FLAGS_${config}}")
          string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_${config} "${CMAKE_C_FLAGS_${config}}")
        endforeach()
      endif()

      message(STATUS "Using sccache: ${SCCACHE_EXECUTABLE}")
    else()
      set(WITH_SCCACHE OFF)
      message(STATUS "sccache not found, disabling")
    endif()
  endif()
else()
  set(WITH_SCCACHE OFF)
endif()

mark_as_advanced(SCCACHE_EXECUTABLE)
