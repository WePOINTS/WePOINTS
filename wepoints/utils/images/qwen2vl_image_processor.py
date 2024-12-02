import math
from typing import List, Optional, Tuple, Union

import numpy as np
from transformers import Qwen2VLImageProcessor
from transformers.image_transforms import (convert_to_rgb, resize,
                                           to_channel_dimension_format)
from transformers.image_utils import (ChannelDimension, ImageInput,
                                      PILImageResampling, VideoInput,
                                      get_image_size,
                                      infer_channel_dimension_format,
                                      is_scaled_image, make_list_of_images,
                                      to_numpy_array)
from transformers.utils import logging

logger = logging.get_logger(__name__)


def smart_resize(height: int,
                 width: int,
                 factor: int = 28,
                 min_pixels: int = 56 * 56,
                 max_pixels: int = 14 * 14 * 4 * 1280) -> Tuple[int, int]:
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range
        ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    Copied from https://github.com/huggingface/transformers/blob/f41d5d8f747f48849005d18dd1c04d5889f31c1b/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L99 # noqa

    Do not raise exception if height or width is less than factor
    """
    if height < factor or width < factor:
        pass
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f'absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}'  # noqa
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen2ImageProcessorForPOINTSV15(Qwen2VLImageProcessor):
    """Copied from Qwen2ImageProcessor.

    Modified to support small images.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: bool = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                'It looks like you are trying to rescale already '
                'rescaled images. If the input'
                ' images have pixel values between 0 and 1, '
                'set `do_rescale=False` to avoid rescaling them again.')
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0],
                                       channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                image = resize(image,
                               size=(resized_height, resized_width),
                               resample=resample,
                               input_data_format=input_data_format)

            if do_rescale:
                image = self.rescale(image,
                                     scale=rescale_factor,
                                     input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(image=image,
                                       mean=image_mean,
                                       std=image_std,
                                       input_data_format=input_data_format)

            image = to_channel_dimension_format(
                image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size  # noqa
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size *
            self.patch_size * self.patch_size)

        return flatten_patches, (grid_t, grid_h, grid_w)
