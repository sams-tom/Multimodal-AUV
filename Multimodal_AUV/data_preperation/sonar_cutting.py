###############################################################################################################################################################################
## This extracts the colocated geotiff sonars into images of the same dimension as the images and puts them in a folder as a stacked tensor o be used for CNN
####################################################################################################################################################################
import rasterio # For reading and writing geospatial raster datasets
from rasterio.windows import Window # For defining windows for extracting raster data
from rasterio.transform import from_origin # For creating affine transformation from pixel and map coordinates
import os # For interacting with the operating system (e.g., file operations)
import csv # For reading and writing CSV files
from PIL import Image # For image processing tasks
import pandas as pd # For data manipulation and analysis
import shutil # For high-level file operations (e.g., moving files)
from rasterio.plot import show # For visualizing raster data
from rasterio.merge import merge # For merging raster datasets
import cv2 # For computer vision tasks (e.g., image processing)
import torch # For tensor computations
import numpy as np # For numerical computations
import matplotlib.pyplot as plt # For plotting and visualization

#this function returns the value of pixels resolution (IE the return is what the pixel equals in actual on the ground measurement (IE 0.2 is 0.2 of a meter))
def get_pixel_resolution(geotiff_file):
    """
    Retrieves the pixel resolution (x and y) from a GeoTIFF file.

    Args:
        geotiff_file (str): Path to the GeoTIFF file.

    Returns:
        tuple: A tuple containing the x and y resolutions.
    """
    try:
        with rasterio.open(os.path.join(folder_path, geotiff_file)) as dataset:  # Corrected path
            transform = dataset.transform
            x_resolution = transform[0]
            y_resolution = transform[4]
            return x_resolution, y_resolution
    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening {geotiff_file}: {e}")
        return None, None  # Return None in case of error

# Output folder
# window_size in meters, this will give a square x*x meters
window_size = 20

# Folder containing GeoTIFF files
folder_path = 'D:/Dataset_AUV_IRELAND/Irish sonar/'

# Folder containing images
image_folder = 'E:/strangford/Images/Processed/'

# CSV containing image information
csv_file_path = "E:/strangford/Images/Processed/combined_csv.csv"

# Get list of files in the folder
files = os.listdir(folder_path)
def is_geotiff(file):
    """Checks if a file is a GeoTIFF."""
    return file.lower().endswith(('.tif', '.tiff'))
# Filter GeoTIFF files
geotiff_files = [file for file in files if is_geotiff(file)]

for file in geotiff_files:
    x, y = get_pixel_resolution(file)
    if x is not None and y is not None:
        print(f"File: {file}, X Resolution: {x}, Y Resolution: {y}")
#This took the folder of labelled images and then get the csv of the depths and locations etc. to be used later
def filter_csv_by_image_names(csv_file_path, image_folder_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Filter rows where Image_Name is in the image folder
    filtered_df = df[df['Image_Name'].apply(lambda x: x in os.listdir(image_folder_path))]

    return filtered_df

#Takes a geotiff and checks if the image is located there, if it is it will extract a patch, then move these, the image and mask into a folder
def extract_grid(geotiff_path, easting, northing, image_name_o, output_folder, window_size, image_folder, label, row, image_path):
    try:
        os.makedirs(output_folder, exist_ok=True)

        with rasterio.open(geotiff_path) as src:
            bounds = src.bounds
            print(f"GeoTIFF Boundaries for {geotiff_path}:")
            print(f"Left: {bounds.left}, Bottom: {bounds.bottom}, Right: {bounds.right}, Top: {bounds.top}")
            width_pixels = src.width
            height_pixels = src.height

            pixel_size_x = (bounds.right - bounds.left) / width_pixels
            pixel_size_y = (bounds.top - bounds.bottom) / height_pixels
            print(f"Width (pixels): {width_pixels}, Height (pixels): {height_pixels}")
            print(f"Pixel size: {pixel_size_x:.2f} meters (X), {pixel_size_y:.2f} meters (Y)")

            row_pixel, col_pixel = src.index(easting, northing)

            window_size_pixels_horizontal = int(window_size / pixel_size_x)
            window_size_pixels_vertical = int(window_size / pixel_size_y)

            window = Window(
                col_pixel - (window_size_pixels_horizontal // 2),
                row_pixel - (window_size_pixels_vertical // 2),
                window_size_pixels_horizontal,
                window_size_pixels_vertical
            )

            print(f"Window defined: {window}")
            print(f"Window dimensions (pixels): {window_size_pixels_horizontal} x {window_size_pixels_vertical}")

            data = src.read(window=window)

            if not data.any():
                return
            else:
                print(f"found data")

            filename = os.path.basename(geotiff_path)
            filename = os.path.splitext(filename)[0]

            filename_parts = filename.split("_")
            final_three_parts = "_".join(filename_parts[-3:])

            image_name = f"grid_{easting}_{northing}_{final_three_parts}.jpg"

            output_folder_1 = os.path.join(output_folder, os.path.splitext(image_name_o)[0])
            output_path = os.path.join(output_folder_1, image_name)

            os.makedirs(output_folder_1, exist_ok=True)

            if filename_parts[-1] == 'Bathy':
                channels_to_save = [1, 2]
                for i in channels_to_save:
                    image = Image.fromarray(data[i])
                    output_path_channel = os.path.join(output_folder_1, f"output_channel_{i}.jpg")
                    print(output_path_channel)
                    image.save(output_path_channel)
            else:
                image = Image.fromarray(data[0])
                image.save(output_path)

            image_name_full_path =image_path
            destination_folder = output_folder_1
            print(f"{image_name_full_path}, destination folder {destination_folder} sdfasdfasdfsdf")
            shutil.copy(image_name_full_path, destination_folder)

            # Save the row as a CSV in the new image folder
            csv_output_path = os.path.join(output_folder_1, 'row_data.csv')
            with open(csv_output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                header = list(row.keys())[3:]  # Get all keys except the first three
                writer.writerow(header)
                row_values = [row[key] for key in header]
                writer.writerow(row_values)

            label_file_path = os.path.join(output_folder_1, f"{label}.txt")
            with open(label_file_path, 'w') as label_file:
                label_file.write(label)
            print(f"Label saved as {label_file_path}")

    except Exception as e:
        pass
#This gets all the geotiff files 
def is_geotiff(filename):
    return filename.lower().endswith('.tif') or filename.lower().endswith('.tiff')



def process_data(csv_file_path, geotiff_files, folder_path, output_folder, window_size, image_folder):
    with open(csv_file_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            easting = float(row['easting'])
            northing = float(row['northing'])
            image_name = row['Image_Name']
            image_path = row['path']
            label = row.get('label', "unlabelled")  # Get label, default to "unlabelled" if missing

            print(image_name)
            print(easting, northing)

            for geotiff_path in geotiff_files:
                file_path = os.path.join(folder_path, geotiff_path)
                extract_grid(file_path, easting, northing, image_name, output_folder, window_size, image_folder, label, row, image_path)  # Pass the entire row


def process_images(folder_path):
    """Processes images within a single folder, including average subtraction."""

    image = None
    chosen_sss = None
    bathy_images = []
    image_name = None
    min_zeros = float('inf') # Initialize min_zeros to positive infinity

import os
import cv2
import numpy as np


import os
import cv2
import re
import shutil

def process_frame_channels_in_subfolders(root_folder):
    """
    Processes frame, output_channel_1, and output_channel_2 images from subfolders of a given root folder,
    and saves the combined images within each subfolder. Copies frame images grouped by their numerical prefix
    (e.g., frame000001, frame000005) into new directories. Deletes demeaned and average_subtracted images first.

    Args:
        root_folder (str): The root directory containing subfolders.
        output_directory (str): The directory where frame image groups will be copied.
    """

    if not os.path.exists(root_folder):
        print(f"Root folder not found: {root_folder}")
        return

    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    for subfolder_name in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder_name)

        if os.path.isdir(subfolder_path):  # Check if it's a directory
            print(f"Processing subfolder: {subfolder_path}")

            # Delete demeaned and average_subtracted images
            for filename in os.listdir(subfolder_path):
                if "demeaned" in filename or "average_subtracted" in filename:
                    file_path = os.path.join(subfolder_path, filename)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except OSError as e:
                        print(f"Error deleting {file_path}: {e}")

            files = os.listdir(subfolder_path)
            print(f"Files in {subfolder_path}: {files}")

            frame_groups = {}
            channel1_image = None
            channel2_image = None

            for filename in files:
                img_path = os.path.join(subfolder_path, filename)
                if os.path.isfile(img_path):
                    if "output_channel_1" in filename:
                        channel1_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if channel1_image is not None:
                            print(f"Channel 1 found in {subfolder_path}")
                        else:
                            print(f"Error reading image: {img_path}")
                    elif "output_channel_2" in filename:
                        channel2_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if channel2_image is not None:
                            print(f"Channel 2 found in {subfolder_path}")
                        else:
                            print(f"Error reading image: {img_path}")
                    elif filename.startswith("frame") and "mask" not in filename:
                        match = re.match(r"frame(\d+)_", filename)
                        if match:
                            frame_num = match.group(1)
                            if frame_num not in frame_groups:
                                frame_groups[frame_num] = []
                            frame_groups[frame_num].append((filename, img_path))  # Store filename and path
                        else:
                            print(f"Frame filename does not match expected pattern: {filename}")

            ## Copy frame images grouped by numerical prefix
            #for frame_num, frame_data in frame_groups.items():
            #    group_output_dir = os.path.join(root_folder, f"frame_{frame_num}")
            #    if not os.path.exists(group_output_dir):
            #        os.makedirs(group_output_dir)

            #    for filename, img_path in frame_data:
            #        dest_path = os.path.join(group_output_dir, filename)
            #        try:
            #            shutil.copy2(img_path, dest_path)
            #            print(f"Copied {filename} to {dest_path}")
            #        except Exception as e:
            #            print(f"Error copying {filename}: {e}")

            # Combine channel images if available
            if channel1_image is not None and channel2_image is not None:
                # Ensure all images have the same dimensions
                height, width = channel1_image.shape
                if channel2_image.shape != (height, width):
                    channel2_image = cv2.resize(channel2_image, (width, height))
                try:
                    combined_channels = cv2.merge([channel1_image, channel2_image])
                    print(f"Combined channels shape: {combined_channels.shape}")
                    print(f"Combined channels dtype: {combined_channels.dtype}")
                    if combined_channels.dtype != np.uint8:
                        combined_channels = combined_channels.astype(np.uint8)
                    # Create a 3 channel image
                    three_channel_bathy = np.zeros((height, width, 3), dtype=np.uint8)
                    three_channel_bathy[:, :, 0] = combined_channels[:, :, 0]
                    three_channel_bathy[:, :, 1] = combined_channels[:, :, 1]
                    combined_channels_path = os.path.join(subfolder_path, "combined_channels.png")
                    cv2.imwrite(combined_channels_path, three_channel_bathy)
                    print(f"Combined channels image saved to: {combined_channels_path}")
                except cv2.error as e:
                    print(f"Error combining channel images: {e}")
            else:
                print(f"Not both channel images found in {subfolder_path}")
def save_data_as_image(folder_path, image_name, chosen_sss, bathy_images):
    """Saves processed images to the same folder they were found in."""

    if chosen_sss is not None or bathy_images: #Check if there is anything to save.
        if chosen_sss is not None:
            chosen_sss_img = Image.fromarray(np.uint8(chosen_sss))
            chosen_sss_filename = f"{image_name}_sss.png"
            chosen_sss_img.save(os.path.join(folder_path, chosen_sss_filename))
            print(f"Saved {chosen_sss_filename} in {folder_path}")

        if bathy_images:
            combined_bathy = np.concatenate(bathy_images, axis=1)
            combined_bathy_img = Image.fromarray(np.uint8(combined_bathy))
            combined_bathy_filename = f"{image_name}_combined_bathy.png"
            combined_bathy_img.save(os.path.join(folder_path, combined_bathy_filename))
            print(f"Saved {combined_bathy_filename} in {folder_path}")
    else:
        print(f"Nothing to save for {image_name}")

def process_main_directory(main_directory):
    """Processes all subfolders within the main directory, saving in each subfolder."""

    for folder_name in os.listdir(main_directory):
        folder_path = os.path.join(main_directory, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            image, chosen_sss, bathy_images, image_name = process_images(folder_path)
            if image is not None and chosen_sss is not None and bathy_images is not None:
                save_data_as_image(folder_path, image_name, chosen_sss, bathy_images)


def update_csv_path(csv_file_path, old_prefix, new_prefix):
    """
    Reads a CSV file, updates the 'path' column by replacing old_prefix with new_prefix,
    and saves the modified CSV back to the same file.

    Args:
        csv_file_path (str): The path to the CSV file.
        old_prefix (str): The string to be replaced.
        new_prefix (str): The replacement string.
    """

    rows = []  # Store the modified rows
    header = None  # Store the header row

    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Read the header row
            for row in reader:
                if len(row) > 0: # Check the row is not empty
                    path_index = header.index('path')
                    row[path_index] = row[path_index].replace(old_prefix, new_prefix)
                rows.append(row)

        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # Write the header row
            writer.writerows(rows)  # Write the modified rows

        print(f"CSV file '{csv_file_path}' updated successfully.")

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
    except ValueError:
        print(f"Error: 'path' column not found in CSV header.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def average_subtract_frame_folders_move(root_directory):
    """
    For all folders in the root directory that have the format frame_xxxxxx,
    calculates the average of the images in the folder, subtracts this average
    from each image, and saves the resulting images into a new folder 
    named after the original image, located in the root directory,
    with "average_subtracted" appended to the filename. Removes the extension from the folder name.

    Args:
        root_directory (str): The root directory containing the frame_xxxxxx folders.
    """

    if not os.path.exists(root_directory):
        print(f"Root directory not found: {root_directory}")
        return

    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)

        if os.path.isdir(folder_path) and folder_name.startswith("frame_"):
            print(f"Processing folder: {folder_path}")

            image_paths = []
            image_names = []
            images = []

            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                if os.path.isfile(img_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')): #added image type check
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
                        image_names.append(filename)
                    else:
                        print(f"Error reading image: {img_path}")

            if images:
                images_float = [img.astype(np.float32) for img in images]
                average_image = np.mean(np.array(images_float), axis=0)

                for i, img in enumerate(images):
                    average_subtracted_image = img.astype(np.float32) - average_image
                    average_subtracted_image = np.clip(average_subtracted_image, 0, 255).astype(np.uint8)

                    base_name, ext = os.path.splitext(image_names[i])
                    new_filename = f"{base_name}_average_subtracted{ext}"

                    # Remove extension from folder name
                    output_folder = os.path.join(root_directory, base_name)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    output_path = os.path.join(output_folder, new_filename)
                    cv2.imwrite(output_path, average_subtracted_image)
                    print(f"Average subtracted image saved to: {output_path}")

                    # Optionally delete the original image

            else:
                print(f"No images found in folder: {folder_path}")



csv_file_path = 'E:/strangford/Images/Processed/combined_csv.csv'  # Replace with your CSV file path
old_prefix = 'C:/Users/phd01tm/OneDrive - SAMS/Strangford loch AUV data/AUV data/Images/'
new_prefix = 'E:/strangford/Images/'

update_csv_path(csv_file_path, old_prefix, new_prefix)

# Output folder
#window_size in meters, this will give a square x*x meters
window_size=20
# Folder containing GeoTIFF files
folder_path = 'D:/Dataset_AUV_IRELAND/Irish sonar/'
#Folder contaiting images
image_folder = 'E:/strangford/Images/Processed/'

#CSV containing image information
csv_file_path = "E:/strangford/Images/Processed/combined_csv.csv"
# Get list of files in the folder
files = os.listdir(folder_path)
# Filter GeoTIFF files
geotiff_files = [file for file in files if is_geotiff(file)]
for file in geotiff_files:
    x,y =get_pixel_resolution(file)
    print(x,y)
#To keep track of image were on
i=0

output_folder= 'D:/all_mulroy_images_and_sonar/'

#Cuttign up sonar and creating folders of image, mask, and sonar
#process_data(csv_file_path=csv_file_path, geotiff_files=geotiff_files, output_folder=output_folder, folder_path=folder_path, window_size=window_size, image_folder=image_folder)
# Process folders to create a stacked tensor of images and sonar
process_frame_channels_in_subfolders(output_folder)

#average_subtract_frame_folders_move(output_folder)