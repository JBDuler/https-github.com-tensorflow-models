# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys

# Hack code until a more formal solution is decided.
_ROOT = os.path.abspath(__file__).split("official/utils")[0]
if _ROOT not in sys.path: sys.path.append(_ROOT)

import official.utils.arg_parsers

class TestParser(official.utils.arg_parsers.BaseParser):
  def __init__(self):
    super().__init__()
    self._add_device_args(allow_cpu=True, allow_gpu=True, allow_multi_gpu=True)
    self._add_supervised_args()

  def _add_learning_rate(self):
    pass

if __name__ == "__main__":
  parser = TestParser()
  flags = parser.parse_args()
  print(flags)

