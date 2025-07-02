import os
import cv2
import numpy as np
import re
import shutil
from PIL import Image # Ensure PIL is imported if you use Image.fromarray

def process_frame_channels_in_subfolders(root_folder: str):
    """
    Processes 'frame', 'output_channel_1', and 'output_channel_2' images from subfolders of a given root folder,
    combines channel images, and saves the combined image.
    Deletes 'demeaned' and 'average_subtracted' images within each subfolder.

    Args:
        root_folder (str): The root directory containing subfolders with processed data.
    """
    if not os.path.exists(root_folder):
        print(f"Root folder not found: {root_folder}")
        return

    for subfolder_name in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder_name)

        if os.path.isdir(subfolder_path):
            print(f"Processing subfolder: {subfolder_path}")

            # Delete old processed images (demeaned, average_subtracted)
            for filename in os.listdir(subfolder_path):
                if "demeaned" in filename or "average_subtracted" in filename:
                    file_path = os.path.join(subfolder_path, filename)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except OSError as e:
                        print(f"Error deleting {file_path}: {e}")

            files = os.listdir(subfolder_path)
            channel1_image = None
            channel2_image = None

            for filename in files:
                img_path = os.path.join(subfolder_path, filename)
                if os.path.isfile(img_path):
                    if "output_channel_1" in filename:
                        channel1_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if channel1_image is None: print(f"Error reading image: {img_path}")
                    elif "output_channel_2" in filename:
                        channel2_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if channel2_image is None: print(f"Error reading image: {img_path}")

            # Combine channel images if available
            if channel1_image is not None and channel2_image is not None:
                height, width = channel1_image.shape
                # Ensure dimensions match before combining
                if channel2_image.shape != (height, width):
                    channel2_image = cv2.resize(channel2_image, (width, height))
                try:
                    # Combined channels (2-channel image)
                    combined_channels = cv2.merge([channel1_image, channel2_image])

                    # Create a 3-channel image where the third channel is black (or a copy of one of the others)
                    # This is often needed for models expecting 3 channels (RGB).
                    three_channel_bathy = np.zeros((height, width, 3), dtype=np.uint8)
                    three_channel_bathy[:, :, 0] = combined_channels[:, :, 0] # Channel 1 (Red)
                    three_channel_bathy[:, :, 1] = combined_channels[:, :, 1] # Channel 2 (Green)
                    # three_channel_bathy[:, :, 2] = ... # Blue channel can be 0 or a copy if needed

                    combined_channels_path = os.path.join(subfolder_path, "combined_channels.png")
                    cv2.imwrite(combined_channels_path, three_channel_bathy)
                    print(f"Combined channels image saved to: {combined_channels_path}")
                except cv2.Error as e:
                    print(f"Error combining channel images in {subfolder_path}: {e}")
            else:
                print(f"Not both channel images (output_channel_1/2) found in {subfolder_path}. Skipping combination.")
