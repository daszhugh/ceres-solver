# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2023 Google Inc. All rights reserved.
# http://ceres-solver.org/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Google Inc. nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: alexs.mac@gmail.com (Alex Stewart)
#

# Extract abseil version from ${CERES_SOURCE_ROOT}/third_party/abseil-cpp/CMakeLists.txt
macro(read_abseil_version_from_source CERES_SOURCE_ROOT)
  set(CERES_ABSEIL_VERSION_FILE ${CERES_SOURCE_ROOT}/third_party/abseil-cpp/CMakeLists.txt)
  if (NOT EXISTS ${CERES_ABSEIL_VERSION_FILE})
    message(FATAL_ERROR "Cannot find abseil CMakeLists.txt file in specified "
      "Ceres source directory: ${CERES_SOURCE_ROOT}, it is not here: "
      "${CERES_ABSEIL_VERSION_FILE}")
  endif()

  file(READ ${CERES_ABSEIL_VERSION_FILE} CERES_ABSEIL_VERSION_FILE_CONTENTS)

  string(REGEX MATCH "project\\(absl LANGUAGES CXX VERSION [0-9]+\\)"
  CERES_ABSEIL_VERSION_LINE "${CERES_ABSEIL_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "project\\(absl LANGUAGES CXX VERSION ([0-9]+)\\)" "\\1" CERES_ABSEIL_VERSION ${CERES_ABSEIL_VERSION_LINE})
  # NOTE: if (VAR) is FALSE if VAR is numeric and <= 0, as such we cannot use
  #       it for testing version numbers, which might well be zero, at least
  #       for the patch version, hence check for empty string explicitly.
  if ("${CERES_ABSEIL_VERSION}" STREQUAL "")
    message(FATAL_ERROR "Failed to extract abseil version from "
      "${CERES_ABSEIL_VERSION_FILE}")
  endif()

  message(STATUS "Detected abseil version: ${CERES_ABSEIL_VERSION} from "
    "${CERES_ABSEIL_VERSION_FILE}")
endmacro()
