import os
import csv
import shutil
from PIL import Image # For saving images
import numpy as np # For array operations
import pandas as pd # For filter_csv_by_image_names

# Import functions from your new modules
from data_preparation.utilities import is_geotiff, filter_csv_by_image_names, update_csv_path
from data_preparation.geospatial import get_pixel_resolution, extract_grid_patch
from data_preparation.image_processing import process_frame_channels_in_subfolders
# from data_pipeline.image_processing import process_images, save_data_as_image, process_main_directory # Uncomment if needed later


def process_and_save_data(csv_file_path: str, geotiff_files_paths: list[str],
                          output_root_folder: str, window_size_meters: float,
                          original_images_folder: str):
    """
    Orchestrates the extraction of sonar grids, copying original images, and saving
    associated metadata and labels into structured output folders.

    Args:
        csv_file_path (str): Path to the CSV file containing image metadata (easting, northing, image_name, path, label).
        geotiff_files_paths (list[str]): List of full paths to GeoTIFF files to process.
        output_root_folder (str): The root directory where all processed data folders will be created.
        window_size_meters (float): The side length of the square patch to extract from GeoTIFFs, in meters.
        original_images_folder (str): Path to the folder containing the original images to copy.
    """
    if not os.path.exists(output_root_folder):
        os.makedirs(output_root_folder)
        print(f"Created output root folder: {output_root_folder}")

    # Read the CSV once
    try:
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader) # Load all rows into memory
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}. Aborting data processing.")
        return
    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}. Aborting data processing.")
        return


    for row in rows:
        easting = float(row['easting'])
        northing = float(row['northing'])
        image_name_original = row['Image_Name'] # Original image filename from CSV
        original_image_full_path = row['path'] # Full path to the original image
        label = row.get('label', "unlabelled")

        print(f"\nProcessing entry for image: {image_name_original} at ({easting}, {northing})")

        # Create a specific output subfolder for this image entry
        # Using the original image name as the subfolder name
        output_folder_for_entry = os.path.join(output_root_folder, os.path.splitext(image_name_original)[0])
        os.makedirs(output_folder_for_entry, exist_ok=True)
        print(f"Created/ensured output subfolder: {output_folder_for_entry}")

        # Copy the original image into its dedicated subfolder
        try:
            shutil.copy(original_image_full_path, output_folder_for_entry)
            print(f"Copied original image '{image_name_original}' to {output_folder_for_entry}")
        except FileNotFoundError:
            print(f"Warning: Original image not found at '{original_image_full_path}'. Skipping copy.")
        except Exception as e:
            print(f"Error copying original image '{image_name_original}': {e}")


        # Save the row's data (excluding the first three columns if desired) as a CSV
        csv_output_path = os.path.join(output_folder_for_entry, 'row_data.csv')
        try:
            # Assuming 'Image_Name', 'easting', 'northing' are the first three columns to exclude
            # Adjust if your CSV structure is different
            header = list(row.keys())
            values = list(row.values())

            # Filter out Image_Name, easting, northing if desired
            relevant_header = header[3:] # Adjust indices if needed
            relevant_values = values[3:]

            with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(relevant_header)
                writer.writerow(relevant_values)
            print(f"Row data saved to {csv_output_path}")
        except Exception as e:
            print(f"Error saving row data to CSV: {e}")

        # Save the label to a text file
        label_file_path = os.path.join(output_folder_for_entry, f"{label}.txt")
        try:
            with open(label_file_path, 'w', encoding='utf-8') as f:
                f.write(label)
            print(f"Label saved as {label_file_path}")
        except Exception as e:
            print(f"Error saving label file: {e}")

        # Process each GeoTIFF for this image entry
        for geotiff_path in geotiff_files_paths:
            extracted_patch_info = extract_grid_patch(geotiff_path, easting, northing, window_size_meters)

            if extracted_patch_info:
                data = extracted_patch_info['data']
                geotiff_filename_base = extracted_patch_info['geotiff_filename_base']
                geotiff_type = extracted_patch_info['geotiff_type']

                # Determine output filename based on GeoTIFF type
                filename_parts = geotiff_filename_base.split("_")
                final_three_parts = "_".join(filename_parts[-3:]) # e.g., "date_time_Bathy" or "date_time_SSS"

                output_image_name = f"grid_{easting:.2f}_{northing:.2f}_{final_three_parts}.png" # Use .png for lossless
                output_image_path = os.path.join(output_folder_for_entry, output_image_name)

                try:
                    if geotiff_type.lower() == 'bathy':
                        # For Bathy, save channel 1 and 2 separately as per your original logic
                        # (Adjust if you prefer to save a combined image directly here)
                        # NOTE: your original code saves channels 1 and 2, but then later combines
                        # them in process_frame_channels_in_subfolders.
                        # You might want to save them as 'output_channel_1.png' and 'output_channel_2.png'
                        # if that's what process_frame_channels_in_subfolders expects.
                        if data.shape[0] >= 2: # Ensure there are at least 2 channels
                            img_ch1 = Image.fromarray(data[0]) # Assuming channel 1 is index 0
                            img_ch2 = Image.fromarray(data[1]) # Assuming channel 2 is index 1

                            img_ch1.save(os.path.join(output_folder_for_entry, "output_channel_1.png"))
                            img_ch2.save(os.path.join(output_folder_for_entry, "output_channel_2.png"))
                            print(f"Saved output_channel_1.png and output_channel_2.png from Bathy to {output_folder_for_entry}")
                        else:
                            print(f"Warning: Bathy GeoTIFF {geotiff_filename_base} has less than 2 channels. Skipping channel save.")
                    else: # Assuming SSS or other single-channel data
                        img = Image.fromarray(data[0]) # Assuming single channel at index 0
                        img.save(output_image_path)
                        print(f"Saved {output_image_name} from {geotiff_type} to {output_folder_for_entry}")

                except Exception as e:
                    print(f"Error saving image patch from {geotiff_path} to {output_folder_for_entry}: {e}")
            else:
                print(f"Skipping patch extraction for {geotiff_path} due to previous warnings/errors.")



import os
import argparse
# Assuming these functions are defined elsewhere in your project
# from your_module import update_csv_path, is_geotiff, get_pixel_resolution, process_and_save_data, process_frame_channels_in_subfolders

# Placeholder functions (replace with your actual implementations)
def update_csv_path(csv_file_path, old_prefix, new_prefix):
    """
    Placeholder for the function that updates CSV paths.
    You should replace this with your actual implementation.
    """
    print(f"DEBUG: update_csv_path called with {csv_file_path}, {old_prefix}, {new_prefix}")
    # Example: In a real scenario, this would read the CSV,
    # modify paths, and save the CSV back.
    pass

def is_geotiff(filename):
    """
    Placeholder for the function that checks if a file is a GeoTIFF.
    You should replace this with your actual implementation.
    """
    return filename.lower().endswith(('.tif', '.tiff'))

def get_pixel_resolution(geotiff_path):
    """
    Placeholder for the function that gets GeoTIFF pixel resolution.
    You should replace this with your actual implementation.
    """
    # In a real scenario, you'd use a library like rasterio or GDAL here.
    return 1.0, 1.0 # Dummy values

def process_and_save_data(csv_file_path, geotiff_files_paths, output_root_folder, window_size_meters, original_images_folder):
    """
    Placeholder for the main data processing function.
    You should replace this with your actual implementation.
    """
    print(f"DEBUG: process_and_save_data called with:\n"
          f"  csv_file_path: {csv_file_path}\n"
          f"  geotiff_files_paths: {geotiff_files_paths}\n"
          f"  output_root_folder: {output_root_folder}\n"
          f"  window_size_meters: {window_size_meters}\n"
          f"  original_images_folder: {original_images_folder}")
    # This function would contain the core logic for extracting grids, copying images, etc.
    pass

def process_frame_channels_in_subfolders(output_root_folder):
    """
    Placeholder for the post-processing function.
    You should replace this with your actual implementation.
    """
    print(f"DEBUG: process_frame_channels_in_subfolders called with {output_root_folder}")
    # This function would handle combining bathymetry channels.
    pass


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AUV image and sonar data.")

    # Required arguments
    parser.add_argument('--csv_file_path', type=str, required=True,
                        help='Path to the main CSV file containing image metadata (e.g., E:/strangford/Images/Processed/combined_csv.csv).')
    parser.add_argument('--geotiff_folder_path', type=str, required=True,
                        help='Path to the folder containing GeoTIFF files (e.g., D:/Dataset_AUV_IRELAND/Irish sonar/).')
    parser.add_argument('--original_images_base_folder', type=str, required=True,
                        help='Base folder where the original image files are actually located (e.g., E:/strangford/Images/Processed/).')
    parser.add_argument('--output_root_folder', type=str, required=True,
                        help='Dedicated root folder for all processed output (e.g., D:/all_mulroy_images_and_sonar_processed/).')

    # Optional arguments with default values
    parser.add_argument('--window_size_meters', type=int, default=20,
                        help='Size of the square patch window in meters (e.g., 20 for 20x20m). Default is 20.')
    parser.add_argument('--old_csv_path_prefix', type=str,
                        default='C:/Users/phd01tm/OneDrive - SAMS/Strangford loch AUV data/AUV data/Images/',
                        help='Old path prefix to be replaced in the CSV file (if paths need updating).')
    parser.add_argument('--update_csv_paths_flag', action='store_true',
                        help='Set this flag to update the CSV paths (replaces old_csv_path_prefix with original_images_base_folder).')

    args = parser.parse_args()

    # --- 1. Update CSV Paths (if needed) ---
    if args.update_csv_paths_flag:
        print("\n--- Updating CSV paths ---")
        update_csv_path(args.csv_file_path, args.old_csv_path_prefix, args.original_images_base_folder)
    else:
        print("\n--- Skipping CSV path update (use --update_csv_paths_flag to enable) ---")


    # --- 2. Identify GeoTIFF files ---
    print("\n--- Identifying GeoTIFF files ---")
    if not os.path.isdir(args.geotiff_folder_path):
        print(f"Error: GeoTIFF folder path does not exist: {args.geotiff_folder_path}")
        exit()

    all_files_in_geotiff_folder = os.listdir(args.geotiff_folder_path)
    geotiff_filenames = [f for f in all_files_in_geotiff_folder if is_geotiff(f)]
    geotiff_full_paths = [os.path.join(args.geotiff_folder_path, f) for f in geotiff_filenames]

    if not geotiff_full_paths:
        print(f"Warning: No GeoTIFF files found in {args.geotiff_folder_path}")

    for f_path in geotiff_full_paths:
        x_res, y_res = get_pixel_resolution(f_path)
        print(f"GeoTIFF: {os.path.basename(f_path)}, X Resolution: {x_res:.2f}m, Y Resolution: {y_res:.2f}m")


    # --- 3. Process data: Extracting grids, copying images, saving metadata ---
    print("\n--- Starting main data processing (extracting grids, copying, saving metadata) ---")
    process_and_save_data(
        csv_file_path=args.csv_file_path,
        geotiff_files_paths=geotiff_full_paths,
        output_root_folder=args.output_root_folder,
        window_size_meters=args.window_size_meters,
        original_images_folder=args.original_images_base_folder
    )
    print("\n--- Main data processing completed. ---")


    # --- 4. Post-processing: Combine Bathymetry Channels ---
    print("\n--- Starting post-processing (combining bathymetry channels) ---")
    if not os.path.isdir(args.output_root_folder):
        print(f"Error: Output root folder does not exist for post-processing: {args.output_root_folder}")
        exit()
    process_frame_channels_in_subfolders(args.output_root_folder)
    print("\n--- Post-processing completed. ---")
