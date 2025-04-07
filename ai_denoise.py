#!/usr/bin/env python3

import argparse
import time
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from tqdm import tqdm

# Size of patches used for processing
PATCH_SIZE = 256

def process_patches_with_tflite(
    interpreter: Interpreter,
    patches: np.ndarray,
    show_progress: bool = False
) -> np.ndarray:
    """
    Process patches through a TFLite interpreter.

    Args:
        interpreter: The TFLite interpreter
        patches: Array of patches to process, shape (n_patches, PATCH_SIZE, PATCH_SIZE, 3)
        show_progress: Whether to show a progress bar (default: False)

    Returns:
        Processed patches array with same shape as input
    """
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check if model is quantized (INT8)
    is_quantized = input_details[0]['dtype'] == np.int8

    # Process patches one by one for TFLite
    processed_patches = []
    patch_iter = tqdm(patches, desc="Denoising patches") if show_progress else patches
    for patch in patch_iter:
        # Scale input for INT8 models
        if is_quantized:
            input_scale = input_details[0]['quantization'][0]
            input_zero_point = input_details[0]['quantization'][1]
            patch = patch / input_scale + input_zero_point
            patch = np.array(patch).astype(np.int8)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(patch, 0))

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output = interpreter.get_tensor(output_details[0]['index'])

        # De-scale output for INT8 models
        if is_quantized:
            output_scale = output_details[0]['quantization'][0]
            output_zero_point = output_details[0]['quantization'][1]
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        processed_patches.append(output[0])

    return np.array(processed_patches)

def calculate_num_patches_and_padding(
    shape: tuple[int, int],
    overlap_pixels: int
) -> tuple[int, int, int, int]:
    """
    Calculate the number of patches and padding required to split an image into overlapping patches.

    Args:
        shape: Tuple of (height, width, channels) of the input image
        overlap_pixels: Number of pixels to overlap between patches

    Returns:
        Tuple of (num_patches_w, num_patches_h, width_padded, height_padded) where:
        - num_patches_w: Number of patches in the width dimension
        - num_patches_h: Number of patches in the height dimension
        - width_padded: Width of the padded image needed to make complete patches
        - height_padded: Height of the padded image needed to make complete patches

    Note:
        The function ensures that the image can be split into complete PATCH_SIZE x PATCH_SIZE patches
        by calculating the necessary padding. The stride between patches is PATCH_SIZE - overlap_pixels.
    """
    # Get dimensions
    height, width, _ = shape

    # Patch-to-patch stride (PATCH_SIZE - overlap)
    stride = PATCH_SIZE - overlap_pixels

    # Calculate the number of patches in each dimension, allowing an extra possibly imcomplete patch
    # at the end of each dimension.
    num_patches_w = (width - PATCH_SIZE + stride - 1) // stride + 1
    num_patches_h = (height - PATCH_SIZE + stride - 1) // stride + 1

    # Calculate the padded dimensions of the image to allow for those incomplete patches.
    width_padded = (num_patches_w - 1) * stride + PATCH_SIZE
    height_padded = (num_patches_h - 1) * stride + PATCH_SIZE

    return num_patches_w, num_patches_h, width_padded, height_padded

def split_into_patches(
    image: np.ndarray,
    overlap_pixels: int
) -> np.ndarray:
    """
    Split an image into overlapping PATCH_SIZE x PATCH_SIZE patches with reflection padding.

    Args:
        image: Input image as a numpy array of shape (height, width, channels)
        overlap_pixels: Number of pixels to overlap between patches

    Returns:
        Array of patches with shape (num_patches, PATCH_SIZE, PATCH_SIZE, channels) where:
        - num_patches = num_patches_w * num_patches_h
        - Each patch is PATCH_SIZE x PATCH_SIZE pixels
        - channels is preserved from input image

    Note:
        The function pads the input image using reflection padding to ensure
        complete patches at the edges. The stride between patches is PATCH_SIZE - overlap_pixels.
    """
    # Get image dimensions
    height, width, _ = image.shape

    # Patch-to-patch stride (PATCH_SIZE - overlap)
    stride = PATCH_SIZE - overlap_pixels

    # Figure out how many patches we need, and how to pad the image to make complete patches.
    num_patches_w, num_patches_h, width_padded, height_padded = calculate_num_patches_and_padding(
        image.shape,
        overlap_pixels
    )

    # Create padded image with reflection padding
    img_padded = np.pad(
        image,
        ((0, height_padded - height), (0, width_padded - width), (0, 0)),
        mode='reflect'
    )

    # Extract patches
    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            h_start = i * stride
            w_start = j * stride
            patch = img_padded[h_start:h_start + PATCH_SIZE, w_start:w_start + PATCH_SIZE]
            patches.append(patch)

    return np.array(patches)

def resassemble_patches(
    shape: tuple[int, int, int],
    overlap_pixels: int,
    processed_patches: np.ndarray
) -> np.ndarray:
    """
    Reassemble processed patches into a single image using linear blending in overlap regions.

    Args:
        shape: Tuple of (height, width, channels) of the original image
        overlap_pixels: Number of pixels that overlap between patches
        processed_patches: Array of processed patches with shape (num_patches, PATCH_SIZE, PATCH_SIZE, channels)

    Returns:
        Reassembled image as a numpy array with shape (height, width, channels)

    Note:
        The function uses linear blending in the overlap regions to create smooth transitions
        between patches. The blending weights are generated using a linear ramp from 0 to 1
        across the overlap region. The final image is cropped to the original dimensions
        by removing the padding added during patch extraction.
    """
    # Patch-to-patch stride (PATCH_SIZE - overlap)
    stride = PATCH_SIZE - overlap_pixels

    # Figure out how many patches we need, and how to pad the image to make complete patches.
    num_patches_w, num_patches_h, width_padded, height_padded = calculate_num_patches_and_padding(
        shape,
        overlap_pixels
    )

    # Create output image accumulator
    output_img = np.zeros((height_padded, width_padded, 3), dtype=np.float32)

    # Generate weights for linear blending in the overlap regions.
    weights = np.array([1.0] * PATCH_SIZE)
    weights[:overlap_pixels] = np.linspace(0.0, 1.0, overlap_pixels)
    weights_left_overlap, weights_top_overlap = np.meshgrid(weights, weights)
    weights_right_overlap, weights_bottom_overlap = np.meshgrid(weights[::-1], weights[::-1])
    weights_left_overlap = weights_left_overlap[..., np.newaxis]
    weights_top_overlap = weights_top_overlap[..., np.newaxis]
    weights_right_overlap = weights_right_overlap[..., np.newaxis]
    weights_bottom_overlap = weights_bottom_overlap[..., np.newaxis]

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch_idx = i * num_patches_w + j
            h_start = i * stride
            w_start = j * stride

            patch = processed_patches[patch_idx]
            if i != 0:
                patch = patch * weights_top_overlap
            if i != num_patches_h - 1:
               patch = patch * weights_bottom_overlap
            if j != 0:
                patch = patch * weights_left_overlap
            if j != num_patches_w - 1:
                patch = patch * weights_right_overlap

            output_img[h_start:h_start + PATCH_SIZE, w_start:w_start + PATCH_SIZE] += patch

    # Remove padding.
    output_img = output_img[:shape[0], :shape[1]]

    return output_img

def denoise_image(
    image: np.ndarray,
    interpreter: Interpreter,
    overlap_pixels: int = 16,
    show_progress: bool = False
) -> np.ndarray:
    """
    Process an image by breaking it into patches, running them through the model,
    and reassembling the processed patches into a complete image with linear blending
    in overlap regions.

    Args:
        interpreter: The TFLite interpreter
        image: Input image as a numpy array
        overlap_pixels: Number of pixels to overlap between patches (default: 16)
        show_progress: Whether to show a progress bar (default: False)

    Returns:
        Processed image as a numpy array

    Raises:
        ValueError: If image width or height is less than PATCH_SIZE pixels
    """
    # Check minimum dimensions
    height, width = image.shape[:2]
    if height < PATCH_SIZE or width < PATCH_SIZE:
        raise ValueError(f"Image dimensions ({width}x{height}) must be at least {PATCH_SIZE}x{PATCH_SIZE} pixels")

    # Ensure image is float32 and in range [0, 1]
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    # Split the image into overlapping patches.
    patches = split_into_patches(image, overlap_pixels)

    # Process patches through the model
    processed_patches = process_patches_with_tflite(interpreter, patches, show_progress)

    # Resassemble the patches into a single image.
    output_img = resassemble_patches(image.shape, overlap_pixels, processed_patches)

    # Scale back to 0-255 range
    output_img = (output_img * 255 + .5).clip(0, 255).astype(np.uint8)

    return output_img

def load_model(model_path: str) -> Interpreter:
    """
    Load a TFLite model from a file.

    Args:
        model_path: Path to the TFLite model file

    Returns:
        A TFLite Interpreter

    Raises:
        ValueError: If the file extension is not .tflite
    """
    if not model_path.endswith('.tflite'):
        raise ValueError("Model file must end in .tflite")

    interpreter = Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    return interpreter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI-based image denoising tool')
    parser.add_argument('-i', '--input',
                        required=True,
                        help='Input image file to denoise')
    parser.add_argument('-o', '--output',
                        required=True,
                        help='Output file path for the denoised image')
    parser.add_argument('-m', '--model',
                        default='nafnet.tflite',
                        help='Path to the TFLite model file (default: nafnet.tflite)')
    args = parser.parse_args()

    print(f"Input file: {args.input}")
    print(f"Model file: {args.model}")

    # Load the input image and convert to numpy array
    input_image = Image.open(args.input)
    image_array = np.array(input_image)

    # Load the model
    interpreter = load_model(args.model)

    # Denoise the image with timing
    start_time = time.monotonic()
    denoised_array = denoise_image(image_array, interpreter, show_progress=True)
    print(f"Time to denoise image: {time.monotonic() - start_time:.3f} seconds")

    # Convert back to PIL Image and save
    denoised_image = Image.fromarray(denoised_array)
    denoised_image.save(args.output)
    print(f"Denoised image saved to {args.output}")
