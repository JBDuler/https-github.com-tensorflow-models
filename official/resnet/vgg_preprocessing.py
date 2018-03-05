# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Provides utilities to preprocess images.

The preprocessing steps for VGG were introduced in the following technical
report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def _get_h_w(image):
  """Convenience for grabbing the height and width of an image.
  """
  shape = tf.shape(image)
  return shape[0], shape[1]


def _random_crop_and_flip(image, bbox, crop_height, crop_width):
  """Crops the given image to a random part of the image, and randomly flips.

  Args:
    image: a 3-D image tensor
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    crop_height: the new height.
    crop_width: the new width.

  Returns:
    3-D tensor with cropped image.

  """

  # A large fraction of image datasets contain a human-annotated bounding box
  # delineating the region of the image containing the object of interest.  We
  # choose to create a new bounding box for the object which is a randomly
  # distorted version of the human-annotated bounding box that obeys an
  # allowed range of aspect ratios, sizes and overlap with the human-annotated
  # bounding box. If no box is supplied, then we assume the bounding box is
  # the entire image.
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      tf.shape(image),
      bounding_boxes=bbox,
      min_object_covered=0.1,
      aspect_ratio_range=[0.75, 1.33],
      area_range=[0.05, 1.0],
      max_attempts=100,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  cropped = tf.slice(image, bbox_begin, bbox_size)

  cropped = tf.image.random_flip_left_right(cropped)
  return cropped


def _central_crop(image, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image: a 3-D image tensor
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    3-D tensor with cropped image.
  """
  height, width = _get_h_w(image)

  total_crop_height = (height - crop_height)
  crop_top = total_crop_height // 2
  total_crop_width = (width - crop_width)
  crop_left = total_crop_width // 2
  return tf.slice(
      image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  # We have a 1-D tensor of means; convert to 3-D.
  means = tf.expand_dims(tf.expand_dims(means, 0), 0)

  return image - means


def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: an int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.cast(smallest_side, tf.float32)

  height = tf.cast(height, tf.float32)
  width = tf.cast(width, tf.float32)

  smaller_dim = tf.minimum(height, width)
  scale_ratio = smallest_side / smaller_dim
  new_height = tf.cast(height * scale_ratio, tf.int32)
  new_width = tf.cast(width * scale_ratio, tf.int32)

  return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height, width = _get_h_w(image)
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)

  resized_image = tf.image.resize_images(
      image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR,
      align_corners=False)
  return resized_image


def preprocess_image(image, bbox, output_height, output_width,
                     is_training=False,
                     resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, then this value
      is used for rescaling.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, this value is
      ignored. Otherwise, the resize side is sampled from
        [resize_size_min, resize_size_max].

  Returns:
    A preprocessed image.
  """
  if is_training:
    # For training, we want to randomize some of the distortions.
    image = _random_crop_and_flip(image, bbox, output_height, output_width)
    image = tf.image.resize_images(
        image, [output_height, output_width],
        method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
  else:
    image = _aspect_preserving_resize(image, resize_side_min)
    image = _central_crop(image, output_height, output_width)

  num_channels = image.get_shape().as_list()[-1]
  image.set_shape([output_height, output_width, num_channels])

  image = tf.cast(image, tf.float32)
  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
