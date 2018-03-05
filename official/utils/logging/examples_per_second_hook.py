from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class ExamplesPerSecondHook(tf.train.SessionRunHook):
  """Hook to print out examples per second.
    Total time is tracked and then divided by the total number of steps
    to get the average step time and then batch_size is used to determine
    the running average of examples per second. The examples per second for the
    most recent interval is also logged.
  """

  def __init__(self,
               batch_size,
               every_n_steps=100,
               every_n_secs=None,
               warm_steps=0):
    """Initializer for ExamplesPerSecondHook.
    Args:
      batch_size: Total batch size used to calculate examples/second from
        global time.
      every_n_steps: Log stats every n steps.
      every_n_secs: Log stats every n seconds.
      warm_steps: skip this number of steps before logging and running
        average.
    Raises:
      ValueError: if neither `every_n_steps` or `every_n_secs` is set.
    """

    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError('exactly one of every_n_steps'
                       ' and every_n_secs should be provided.')

    self._timer = tf.train.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)

    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size
    self._warm_steps = warm_steps

  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          'Global step should be created to use StepCounterHook.')

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return tf.train.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    _ = run_context

    global_step = run_values.results

    if self._timer.should_trigger_for_step(
        global_step) and global_step > self._warm_steps:
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        self._step_train_time += elapsed_time
        self._total_steps += elapsed_steps

        average_examples_per_sec = self._batch_size * (
            self._total_steps / self._step_train_time)
        current_examples_per_sec = self._batch_size * (
            elapsed_steps / elapsed_time)
        # Average examples/sec followed by current examples/sec
        tf.logging.info('Batch [%g] %g exp/sec (%g)', self._total_steps,
                     current_examples_per_sec, average_examples_per_sec)

