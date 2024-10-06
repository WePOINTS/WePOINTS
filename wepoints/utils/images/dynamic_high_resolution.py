from typing import List

from PIL import Image


def factorize_number(num: int) -> list:
    """Factorize a number into its prime factors.

    Args:
        num (int): The number to factorize.

    Returns:
        list: A list of prime factors of the number.
    """
    factors = []
    for i in range(1, int(num) + 1):
        if num % i == 0:
            factors.append([i, num // i])
    return factors


def construct_mapping_dict(max_splits: int = 8, image_size: int = 336) -> dict:
    """Construct a mapping dictionary for image size reduction.

    Args:
        max_splits (int, optional): The maximum number of splits for each
            dimension. Defaults to 8.
        image_size (int, optional): The original image size.
            Defaults to 336.

    Returns:
        dict: A dictionary containing the mapping of image sizes to
            the corresponding factors.
    """
    mapping_dict = {}
    for i in range(1, max_splits + 1):
        factor_list = factorize_number(i)
        for factor in factor_list:
            ratio = factor[0] / factor[1]
            if ratio not in mapping_dict:
                mapping_dict[ratio] = [[
                    factor[0] * image_size, factor[1] * image_size
                ]]
            else:
                mapping_dict[ratio].append(
                    [factor[0] * image_size, factor[1] * image_size])
    return mapping_dict


def find_best_image_size(cur_image_size: list,
                         max_splits: int = 8,
                         image_size: int = 336) -> list:
    """Find the best image size for a given image size.

    Args:
        cur_image_size (list): The current image size.
        max_splits (int, optional): The maximum number of splits for each
            dimension. Defaults to 8.
        image_size (int, optional): The original image size.
            Defaults to 336.

    Returns:
        list: The best image size for the given image size.
    """

    mapping_dict = construct_mapping_dict(max_splits, image_size)
    ratio = cur_image_size[0] / cur_image_size[1]
    # find the value which key is the closest to the ratio
    best_ratio = min(mapping_dict.keys(), key=lambda x: abs(x - ratio))
    # best_image_sizes is a list of image sizes
    best_image_sizes = mapping_dict[best_ratio]
    # find the image_size whose area is closest to the current image size
    best_image_size = min(
        best_image_sizes,
        key=lambda x: abs(x[0] * x[1] - cur_image_size[0] * cur_image_size[1]))
    return best_image_size


def split_image(pil_image: Image.Image,
                image_size: int = 336,
                max_splits: int = 8) -> List[Image.Image]:
    """Split an image into sub-image.

    Similar to that used in InternVL2。

    Args:
        pil_image (Image.Image): The input image.
        image_size (int, optional): The size of the image.
            Defaults to 336.
        max_splits (int, optional): The maximum number of splits for each
            dimension. Defaults to 8.

    Returns:
        List[Image.Image]: A list of cropped images.
    """
    whole_sub_image = pil_image.resize((image_size, image_size), resample=2)
    best_size = find_best_image_size(pil_image.size,
                                     max_splits=max_splits,
                                     image_size=image_size)
    pil_image = pil_image.resize(best_size, resample=2)
    num_sub_images = ((best_size[0] // image_size),
                      (best_size[1] // image_size))
    # crop pil_image to sub_images
    sub_images = []
    for i in range(num_sub_images[1]):
        for j in range(num_sub_images[0]):
            sub_image = pil_image.crop(
                (j * image_size, i * image_size, (j + 1) * image_size,
                 (i + 1) * image_size))
            sub_images.append(sub_image)
    sub_images.append(whole_sub_image)
    return sub_images
