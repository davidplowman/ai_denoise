#!/usr/bin/env python3

import argparse
import time

from picamera2 import Picamera2
import libcamera

def create_preview_config(picam2):
    """Create a preview configuration with minimal noise reduction and regular sharpening."""
    controls = {'NoiseReductionMode': libcamera.controls.draft.NoiseReductionModeEnum.Minimal, 'Sharpness': 1}
    return picam2.create_preview_configuration({'format': 'YUV420'},controls=controls)

def create_capture_config(picam2):
    """Create a capture configuration with no noise reduction and no sharpening."""
    half_res = (picam2.sensor_resolution[0] // 2, picam2.sensor_resolution[1] // 2)
    controls = {'NoiseReductionMode': libcamera.controls.draft.NoiseReductionModeEnum.Off, 'Sharpness': 0}
    return picam2.create_still_configuration(sensor={'output_size': half_res}, controls=controls)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Capture a full resolution image with no noise reduction or sharpening')
    parser.add_argument('-o', '--output',
                        required=True,
                        help='Output file path for the captured image')
    parser.add_argument('-c', '--camera',
                        type=int,
                        default=0,
                        help='Camera number to use (default: 0)')
    parser.add_argument('-g', '--gain',
                        type=float,
                        default=0.0,
                        help='Analogue gain to apply (default: use auto gain)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if not args.output.lower().endswith('.png'):
        print("Warning: Output should be to a PNG file for best results")

    print(f"Capturing noisy image to {args.output} using camera {args.camera}")

    with Picamera2(camera_num=args.camera) as picam2:
        preview_config = create_preview_config(picam2)
        capture_config = create_capture_config(picam2)

        # Start in preview mode and do a capture once we're sure the camera has settled.
        picam2.configure(preview_config)
        picam2.set_controls({'AnalogueGain': args.gain})
        picam2.start()
        time.sleep(2)
        print("Capturing image...")
        picam2.switch_mode_and_capture_file(capture_config, args.output)
        print("Image captured successfully")
