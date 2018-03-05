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

import argparse
import re


class NearlyRawTextHelpFormatter(argparse.HelpFormatter):
  """
    This formatter allows explicit newlines and indentation, but handles
  wrapping text where appropriate.
  """
  def _split_lines(self, text, width):
    output = []
    for line in text.splitlines():
      output.extend(self._split_for_length(line, width=width))
    return output

  @staticmethod
  def _split_for_length(text, width):
    out_lines = [[]]
    segments = [i + " " for i in text.split(" ")]
    segments[-1] = segments[-1][:-1]
    current_len = 0
    for segment in segments:
      if not current_len or current_len + len(segment) <= width:
        current_len += len(segment)
        out_lines[-1].append(segment)
      else:
        current_len = 0
        out_lines.append([segment])
    return ["".join(i) for i in out_lines]



class BaseParser(argparse.ArgumentParser):
  """
    This class is intended to house convenience functions, enforcement code, and
  universal or nearly universal arguments.
  """

  def __init__(self):
    super().__init__(
        formatter_class=NearlyRawTextHelpFormatter,
        allow_abbrev=False,  # abbreviations are handled explictly.
    )
    self._add_generic_args()

  @staticmethod
  def _add_checks(default, required):
    if required and default is not None:
      raise ValueError("Required flags do not need defaults.")

  @staticmethod
  def _stringify_choices(choices):
    if choices is None:
      return choices
    output = []
    for i in choices:
      if isinstance(i, (str, int)):
        output.append(str(i))
      else:
        raise ValueError("Could not stringify choices.")
    return output

  @staticmethod
  def _pretty_help(help, default, choices, required, var_type):
    prestring = ""
    if choices is not None:
      prestring = "{" + ", ".join([str(var_type(i)) for i in choices]) + "}    "

    if required:
      prestring += "Required."
    elif default is not None:
      prestring += "Default: %(default)s"

    if len(prestring):
      prestring += "\n"
    return prestring + help

  def add_int(self, name, short_name=None, default=None, choices=None,
              required=False, help=""):
    return self._add_generic_type(
      str, name=name, short_name=short_name, nargs=1, default=default,
      choices=choices, required=required, help=help
    )

  def add_float(self, name, short_name=None, default=None, choices=None,
                required=False, help=""):
    return self._add_generic_type(
      float, name=name, short_name=short_name, nargs=1, default=default,
      choices=choices, required=required, help=help
    )

  def add_str(self, name, short_name=None, nargs=None, default=None,
              choices=None, required=False, help=""):
    return self._add_generic_type(
      str, name=name, short_name=short_name, nargs=nargs, default=default,
      choices=choices, required=required, help=help
    )

  def _add_generic_type(self, var_type, name, short_name=None, nargs=None,
                        default=None, choices=None, required=False, help=""):
    self._add_checks(default=default, required=required)

    names = ["--" + name]
    if short_name is not None:
      names = ["-" + short_name] + names

    self.add_argument(
      *names,
      nargs=nargs,
      default=default,
      type=var_type,
      choices=choices,
      required=required,
      help=self._pretty_help(help=help, default=default, choices=choices,
                             required=required, var_type=var_type),
      metavar=name.upper(),
      dest=name,
    )

  def add_bool(self, name, short_name=None, help=""):
    names = ["--" + name]
    if short_name is not None:
      names = ["-" + short_name] + names

    self.add_argument(
       *names,
       action="store_true",
       help=help,
       dest=name,
    )

  #============================================================================
  # Add Generic Args
  #============================================================================
  """
    Each arg addition is wrapped in a function so that if a subclass doesn't
  want the arg it can override the defining function with a no-op.
  """
  def _add_generic_args(self):
    self._add_tmp_dir()
    self._add_data_dir()
    self._add_model_dir()

  def _add_tmp_dir(self):
    self.add_str("tmp_dir", "td", default="/tmp",
                 help="A directory to place temporary files.")

  def _add_data_dir(self):
    self.add_str("data_dir", "dd", default="/tmp",
                 help="The directory where the input data is stored.")

  def _add_model_dir(self):
    self.add_str("model_dir", "md", default="/tmp",
                 help="The directory where model specific files (event files, "
                      "snapshots, etc.) are stored."
    )

  #=============================================================================
  # Add Common Supervised Learning Args
  #=============================================================================
  def _add_supervised_args(self):
    self._add_train_epochs()
    self._add_epochs_per_eval()
    self._add_learning_rate()
    self._add_batch_size()

  def _add_train_epochs(self):
    self.add_int("train_epochs", "te", default=1,
                 help="The number of epochs to use for training.",
    )

  def _add_epochs_per_eval(self):
    self.add_int("epochs_per_eval", "epe", default=1,
               help="The number of training epochs to run between evaluations.",
    )

  def _add_learning_rate(self):
    self.add_float("learning_rate", "lr", default=1.,
                   help="The learning rate to be used during training.",
    )

  def _add_batch_size(self):
    self.add_int("batch_size", "bs", default=32,
                 help="Batch size for training and evaluation.",
    )

  #=============================================================================
  # Add Args for Specifying Devices
  #=============================================================================
  def _add_device_args(self, allow_cpu=False, allow_gpu=False, allow_tpu=False,
                       allow_multi_gpu=False):
    """
      This method should be called in the __init__ of the child class. The exact
    pattern for this section has yet to be finalized.
    """
    allow_gpu = allow_gpu or allow_multi_gpu  # multi_gpu implies gpu=True

    device_types = []
    if allow_cpu: device_types.append("cpu")
    if allow_gpu: device_types.append("gpu")
    if allow_tpu:
      device_types.append("tpu")
      raise ValueError("tpu args are not ready yet.")

    if not len(device_types):
      raise ValueError("No legal devices specified.")

    self._add_set_device_arg(device_types=device_types)
    if allow_multi_gpu:
      self.add_bool("multi_gpu",
                    help="If set, run across all available GPUs. Note that "
                         "this is superseded by the --num_gpus flag."
      )

  def _add_set_device_arg(self, device_types):
    if len(device_types) == 1:
      return  # no need for the user to specify the device

    self.add_str("device", "d", default="auto",
                 choices=["auto"] + device_types,
                 help="Primary device for neural network computations. Other "
                      "tasks such as dataset managenent may occur on other "
                      "devices. (Generally the CPU.)"
    )










