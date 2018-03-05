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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tempfile import mkstemp


import tensorflow as tf

import hooks

tf.logging.set_verbosity(tf.logging.ERROR)

_HOOK_NAMES = ['LoggingTensorHook', 'ProfilerHook', 'ExamplesPerSecondHook']

class BaseTest(tf.test.TestCase):

  def test_raise_in_empty_names(self):
    with self.assertRaises(ValueError):
      hooks.get_train_hooks(None)

  def test_raise_in_non_string_names(self):
    with self.assertRaises(ValueError):
      hooks.get_train_hooks(['LoggingTensorHook', 'ProfilerHook'])

  def test_raise_in_invalid_names(self):
    invalid_names = 'StepCounterHook, StopAtStepHook'
    with self.assertRaises(ValueError):
      hooks.get_train_hooks(invalid_names)

  def get_train_hooks_valid_names_helper(self, hook_names, hook_counts):
    returned_hooks = hooks.get_train_hooks(hook_names)
    self.assertEqual(len(returned_hooks),hook_counts)
    for returned_hook in returned_hooks:
      self.assertIsInstance(returned_hook, tf.train.SessionRunHook)

  def test_get_train_hooks_one_valid_names(self):
    valid_names = 'LoggingTensorHook'
    self.get_train_hooks_valid_names_helper(valid_names, 1)

  def test_get_train_hooks_three_valid_names(self):
    valid_names = 'LoggingTensorHook, ProfilerHook, ExamplesPerSecondHook'
    self.get_train_hooks_valid_names_helper(valid_names, 3)

if __name__ == '__main__':
  tf.test.main()





