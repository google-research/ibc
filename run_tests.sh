#!/usr/bin/env bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script to run python tests in git repo.

echo "Running run_tests.sh in directory $(pwd)"

RUN_TEST_SAFE () {
  echo "$1"
  eval "$1" || { echo "ERROR: '$1' failed!" ; exit 1 ; }
}

# set -o xtrace

readonly ibc_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
readonly outer_dir="${ibc_dir}/.."
readonly current_python_path="${PYTHONPATH:-}"  # "${...:-}" is bash parameter expansion
readonly updated_python_path="${current_python_path}:${outer_dir}"
readonly test_files=$(find "${ibc_dir}" -name "*_test.py")

echo "Running tests: ${test_files}"

# Run the python unit tests.
for test_file in ${test_files}; do
  echo "***********************************************************************"
  echo "Running test ${test_file}"
  echo "***********************************************************************"
  # PYTHONPATH must include outer-level directory.
  cd "${outer_dir}"
  RUN_TEST_SAFE "PYTHONPATH=${updated_python_path} python3 ${test_file} --alsologtostderr"
done

echo ""
echo "Ran the tests:"
echo "${test_files}"
echo ""
echo "All tests passed!"
