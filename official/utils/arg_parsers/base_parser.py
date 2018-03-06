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


class NearlyRawTextHelpFormatter(argparse.HelpFormatter):
  """Formatter for unified arg parser.

    This formatter allows explicit newlines and indentation but handles wrapping
  text where appropriate.
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
  """Parent parser class for official models.

    This class is intended to house convenience functions, enforcement code, and
  universal or nearly universal arguments.
  """

  def __init__(self):
    super(BaseParser, self).__init__(
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
  def _pretty_help(help_text, default, choices, required, var_type):
    prestring = ""
    if choices is not None:
      prestring = "{" + ", ".join([str(var_type(i)) for i in choices]) + "}    "

    if required:
      prestring += "Required."
    elif default is not None:
      prestring += "Default: %(default)s"

    if prestring:
      prestring += "\n"
    return prestring + help_text

  def add_int(self, name, short_name=None, default=None, choices=None,
              required=False, help_text=""):
    return self._add_generic_type(
        str, name=name, short_name=short_name, nargs=1, default=default,
        choices=choices, required=required, help_text=help_text
    )

  def add_float(self, name, short_name=None, default=None, choices=None,
                required=False, help_text=""):
    return self._add_generic_type(
        float, name=name, short_name=short_name, nargs=1, default=default,
        choices=choices, required=required, help_text=help_text
    )

  def add_str(self, name, short_name=None, nargs=None, default=None,
              choices=None, required=False, help_text=""):
    return self._add_generic_type(
        str, name=name, short_name=short_name, nargs=nargs, default=default,
        choices=choices, required=required, help_text=help_text
    )

  def _add_generic_type(self, var_type, name, short_name=None, nargs=None,
                        default=None, choices=None, required=False,
                        help_text=""):
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
        help=self._pretty_help(help_text=help_text, default=default,
                               choices=choices, required=required,
                               var_type=var_type),
        metavar=name.upper(),
        dest=name
    )

  def add_bool(self, name, short_name=None, help_text=""):
    names = ["--" + name]
    if short_name is not None:
      names = ["-" + short_name] + names

    self.add_argument(
        *names,
        action="store_true",
        help=help_text,
        dest=name
    )

  #============================================================================
  # Add Generic Args
  #============================================================================
  #   Each arg addition is wrapped in a function so that if a subclass doesn't
  # want the arg it can override the defining function with a no-op.

  def _add_generic_args(self):
    self._add_tmp_dir()
    self._add_data_dir()
    self._add_model_dir()

  def _add_tmp_dir(self):
    self.add_str("tmp_dir", "td", default="/tmp",
                 help_text="A directory to place temporary files.")

  def _add_data_dir(self):
    self.add_str("data_dir", "dd", default="/tmp",
                 help_text="The directory where the input data is stored.")

  def _add_model_dir(self):
    self.add_str(
        "model_dir", "md", default="/tmp",
        help_text="The directory where model specific files (event files, "
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
    self.add_int(
        "train_epochs", "te", default=1,
        help_text="The number of epochs to use for training."
    )

  def _add_epochs_per_eval(self):
    self.add_int(
        "epochs_per_eval", "epe", default=1,
        help_text="The number of training epochs to run between evaluations."
    )

  def _add_learning_rate(self):
    self.add_float(
        "learning_rate", "lr", default=1.,
        help_text="The learning rate to be used during training."
    )

  def _add_batch_size(self):
    self.add_int(
        "batch_size", "bs", default=32,
        help_text="Batch size for training and evaluation."
    )

  #=============================================================================
  # Add Args for Specifying Devices
  #=============================================================================
  def _add_device_args(self, allow_cpu=False, allow_gpu=False, allow_tpu=False,
                       allow_multi_gpu=False):
    """Function for determining which device type args are relevant.

      This method should be called in the __init__ of the child class. The exact
    pattern for this section has yet to be finalized.

    Args:
      allow_cpu: The model can be set to run on CPU.
      allow_gpu: The model can be set to run on GPU.
      allow_tpu: The model can be set to run on TPU.
      allow_multi_gpu: The model allows multi GPU training as an option.
    """
    allow_gpu = allow_gpu or allow_multi_gpu  # multi_gpu implies gpu=True

    device_types = []
    if allow_cpu: device_types.append("cpu")
    if allow_gpu: device_types.append("gpu")
    if allow_tpu:
      device_types.append("tpu")
      raise ValueError("tpu args are not ready yet.")

    if not device_types:
      raise ValueError("No legal devices specified.")

    self._add_set_device_arg(device_types=device_types)
    if allow_multi_gpu:
      self.add_bool("multi_gpu",
                    help_text="If set, run across all available GPUs.")

  def _add_set_device_arg(self, device_types):
    if len(device_types) == 1:
      return  # no need for the user to specify the device

    self.add_str(
        "device", "d", default="auto", choices=["auto"] + device_types,
        help_text="Primary device for neural network computations. Other tasks "
                  "may occur on other devices. (Generally the CPU.)"
    )










