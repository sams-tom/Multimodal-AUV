import os
import csv
import shutil
from PIL import Image  # Required for image manipulation and saving (e.g., .png files)
import numpy as np     # Used for array operations, particularly with image data
import pandas as pd    # For DataFrame operations, specifically in filter_csv_by_image_names
import argparse        # For parsing command-line arguments
import sys

# Import functions from your custom data_pipeline modules.
# These modules are assumed to be located in a 'data_pipeline/' directory
from  Multimodal_AUV.data_preparation.utilities import is_geotiff, filter_csv_by_image_names, update_csv_path
from Multimodal_AUV.data_preparation.geospatial import get_pixel_resolution, extract_grid_patch
from Multimodal_AUV.data_preparation.image_processing import process_frame_channels_in_subfolders


# Optional imports for other functions that are commented out in the main block.
# Uncomment them if you intend to use these functions later.
# from data_pipeline.image_processing import process_images, save_data_as_image, process_main_directory


def process_and_save_data(csv_file_path: str, geotiff_files_paths: list[str],
                          output_root_folder: str, window_size_meters: float,
                          original_images_folder: str):
    """
    Orchestrates the extraction of sonar grids, copying original optical images, and saving
    associated metadata and classification labels into a structured output directory.

    For each entry (representing an optical image and its metadata) in the input CSV,
    this function performs the following steps:
    1. Creates a dedicated subfolder within the `output_root_folder`. The subfolder name
       is derived from the original optical image's filename.
    2. Copies the original optical image into this newly created subfolder.
    3. Saves relevant metadata from the CSV row (excluding 'Image_Name', 'easting', 'northing')
       into a `row_data.csv` file within the subfolder.
    4. Saves the classification `label` (e.g., "reef", "sand") into a text file
       (e.g., `reef.txt`) within the subfolder.
    5. Iterates through a list of GeoTIFF files (e.g., Bathymetry, Side-Scan Sonar).
       For each GeoTIFF, it extracts a square patch of `window_size_meters` centered
       at the optical image's `easting` and `northing` coordinates.
    6. Saves the extracted GeoTIFF patches as PNG images within the subfolder.
       For Bathymetry data, individual channels (if available) are saved separately
       (`output_channel_1.png`, `output_channel_2.png`) for subsequent post-processing.

    Args:
        csv_file_path (str): Full path to the CSV file containing image metadata.
                             Expected columns include 'Image_Name', 'easting', 'northing',
                             'path' (full path to original image), and optionally 'label'.
        geotiff_files_paths (list[str]): A list of full paths to all GeoTIFF files
                                         (e.g., bathymetry, side-scan sonar) that need
                                         to be processed for patch extraction.
        output_root_folder (str): The primary directory where all processed data
                                  (image-specific subfolders) will be created.
        window_size_meters (float): The desired side length (in meters) of the square
                                    patch to be extracted from the GeoTIFF files.
        original_images_folder (str): The base directory where the original optical
                                      image files (referenced in the CSV's 'path' column)
                                      are located. While the CSV's 'path' is used for
                                      direct copying, this parameter serves as general
                                      context or for path validation/correction.
    """
    # Ensure the main output directory exists
    if not os.path.exists(output_root_folder):
        os.makedirs(output_root_folder)
        print(f"Created output root folder: {output_root_folder}")

    # Read the entire CSV file into memory once to avoid repeated file I/O
    try:
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)  # Load all rows into a list of dictionaries
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_file_path}'. Aborting data processing.")
        return
    except Exception as e:
        print(f"Error reading CSV file '{csv_file_path}': {e}. Aborting data processing.")
        return

    # Process each row (image entry) from the CSV
    for row in rows:
        try:
            # Extract core metadata for the current image entry
            easting = float(row['easting'])
            northing = float(row['northing'])
            image_name_original = row['Image_Name']  # Original optical image filename
            original_image_full_path = row['path']    # Full path to the original optical image
            label = row.get('label', "unlabelled")    # Get label, default to "unlabelled" if column is missing

            print(f"\n--- Processing entry for image: '{image_name_original}' at ({easting:.2f}, {northing:.2f}) ---")

            # Create a dedicated output subfolder for this image entry.
            # The subfolder name is the original image filename without its extension.
            output_folder_for_entry = os.path.join(output_root_folder, os.path.splitext(image_name_original)[0])
            os.makedirs(output_folder_for_entry, exist_ok=True)
            print(f"Created/ensured output subfolder: '{output_folder_for_entry}'")

            # Copy the original optical image into its dedicated subfolder
            try:
                shutil.copy(original_image_full_path, output_folder_for_entry)
                print(f"Copied original image '{image_name_original}' to '{output_folder_for_entry}'")
            except FileNotFoundError:
                print(f"Warning: Original optical image not found at '{original_image_full_path}'. Skipping copy.")
            except Exception as e:
                print(f"Error copying original image '{image_name_original}': {e}")

            # Save the current row's data (excluding image-specific paths/names) as a CSV file.
            # This retains other potentially useful metadata.
            csv_output_path = os.path.join(output_folder_for_entry, 'row_data.csv')
            try:
                header = list(row.keys())
                
                # Dynamically filter out columns that are redundant in `row_data.csv`.
                # This makes the code more robust to CSV column order changes.
                columns_to_exclude = ['Image_Name', 'easting', 'northing', 'path']
                relevant_header = [h for h in header if h not in columns_to_exclude]
                relevant_values = [row[h] for h in relevant_header] # Retrieve values based on filtered headers

                with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(relevant_header) # Write header row
                    writer.writerow(relevant_values)  # Write data row
                print(f"Row data saved to '{csv_output_path}'")
            except Exception as e:
                print(f"Error saving row data to CSV for '{image_name_original}': {e}")

            # Save the classification label to a simple text file
            label_file_path = os.path.join(output_folder_for_entry, f"{label}.txt")
            try:
                with open(label_file_path, 'w', encoding='utf-8') as f:
                    f.write(label)
                print(f"Label saved as '{label_file_path}'")
            except Exception as e:
                print(f"Error saving label file for '{image_name_original}': {e}")

            # Process each GeoTIFF file: extract a grid patch and save it
            for geotiff_path in geotiff_files_paths:
                # Calls `extract_grid_patch` from `data_pipeline.geospatial` module.
                extracted_patch_info = extract_grid_patch(geotiff_path, easting, northing, window_size_meters)

                if extracted_patch_info:
                    data = extracted_patch_info['data']  # The extracted NumPy array data
                    geotiff_filename_base = extracted_patch_info['geotiff_filename_base'] # e.g., '2023_06_15_10_30_Bathy'
                    geotiff_type = extracted_patch_info['geotiff_type'] # e.g., 'Bathy', 'SSS'

                    # Construct a descriptive output filename for the extracted patch.
                    # This uses parts of the original GeoTIFF filename for identification.
                    filename_parts = geotiff_filename_base.split("_")
                    # Join the last three parts (e.g., "date_time_Bathy" or "date_time_SSS")
                    final_three_parts = "_".join(filename_parts[-3:]) 

                    # Use .png for lossless image saving, ensuring data integrity.
                    output_image_name = f"grid_{easting:.2f}_{northing:.2f}_{final_three_parts}.png"
                    output_image_path = os.path.join(output_folder_for_entry, output_image_name)

                    try:
                        # Special handling for 'Bathy' GeoTIFFs: save channels separately.
                        # This prepares the data for `process_frame_channels_in_subfolders`.
                        if geotiff_type.lower() == 'bathy':
                            if data.ndim == 3 and data.shape[0] >= 2: # Ensure it's multi-channel data
                                # Assuming channel 1 is at index 0 and channel 2 is at index 1
                                # Convert each channel (NumPy array) to a PIL Image and save.
                                img_ch1 = Image.fromarray(data[0]) 
                                img_ch2 = Image.fromarray(data[1]) 

                                img_ch1.save(os.path.join(output_folder_for_entry, "output_channel_1.png"))
                                img_ch2.save(os.path.join(output_folder_for_entry, "output_channel_2.png"))
                                print(f"Saved 'output_channel_1.png' and 'output_channel_2.png' from Bathy to '{output_folder_for_entry}'")
                            else:
                                print(f"Warning: Bathy GeoTIFF '{geotiff_filename_base}' has insufficient (expected >=2) or incorrect dimensions ({data.ndim}). Skipping channel save.")
                        else: # Assume Side-Scan Sonar (SSS) or other single-channel GeoTIFF data
                            # Assuming single channel data is at index 0 or directly 2D.
                            if data.ndim == 3: # If it's (channels, H, W)
                                img = Image.fromarray(data[0]) 
                            else: # If it's already (H, W)
                                img = Image.fromarray(data)
                            img.save(output_image_path)
                            print(f"Saved '{output_image_name}' from {geotiff_type} to '{output_folder_for_entry}'")

                    except Exception as e:
                        print(f"Error saving image patch from '{geotiff_path}' to '{output_folder_for_entry}': {e}")
                else:
                    print(f"Skipping patch extraction for '{os.path.basename(geotiff_path)}' due to previous warnings/errors in `extract_grid_patch`.")
        except Exception as e:
            # Catch any unexpected errors during the processing of a single CSV row
            print(f"Critical Error processing row for image '{row.get('Image_Name', 'Unknown')}': {e}. Skipping this row.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess AUV sonar and optical image data for machine learning tasks. "
                    "Extracts sonar grid patches, copies original images, and organizes metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )

    # Required Arguments
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Full path to the main CSV file containing image metadata (easting, northing, image_name, path, label)."
    )
    parser.add_argument(
        "--geotiff_folder",
        type=str,
        required=True,
        help="Path to the folder containing all GeoTIFF files (e.g., Bathymetry, Side-Scan Sonar)."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="The root directory where all processed and organized output data will be saved."
             "Each optical image will have its own subfolder here."
    )
    parser.add_argument(
        "--original_images_base_folder",
        type=str,
        required=True,
        help="The base directory where your *original optical image files* are stored."
             "This is used in conjunction with --csv_old_path_prefix for path correction."
    )

    # Optional Arguments with Default Values
    parser.add_argument(
        "--window_size_meters",
        type=float,
        default=20.0,
        help="The desired side length (in meters) for the square patches extracted from GeoTIFFs."
    )
    
    parser.add_argument(
        "--skip_bathy_combine",
        action="store_true",
        help="If set, the post-processing step to combine bathymetry channels will be skipped."
    )

    args = parser.parse_args()
    


    # --- 2. Identify GeoTIFF files and report resolutions ---
    print("\n--- Step 2: Identifying GeoTIFF files and reporting resolutions ---")
    all_files_in_geotiff_folder = os.listdir(args.geotiff_folder)
    geotiff_filenames = [f for f in all_files_in_geotiff_folder if is_geotiff(f)]
    geotiff_full_paths = [os.path.join(args.geotiff_folder, f) for f in geotiff_filenames]

    if not geotiff_full_paths:
        print(f"Warning: No GeoTIFF files found in '{args.geotiff_folder}'. Sonar data will not be processed.")
    else:
        for f_path in geotiff_full_paths:
            x_res, y_res = get_pixel_resolution(f_path)
            print(f"  Found GeoTIFF: '{os.path.basename(f_path)}', X Resolution: {x_res:.2f}m, Y Resolution: {y_res:.2f}m")
    print("GeoTIFF identification completed.")

    # --- 3. Main Data Processing: Extracting grids, copying images, saving metadata ---
    print("\n--- Step 3: Starting main data processing (extracting grids, copying, saving metadata) ---")
    process_and_save_data(
        csv_file_path=args.csv_path,
        geotiff_files_paths=geotiff_full_paths,
        output_root_folder=args.output_folder,
        window_size_meters=args.window_size_meters,
        original_images_folder=args.original_images_base_folder
    )
    print("\n--- Main data processing completed. ---")

    # --- 4. Post-processing: Combine Bathymetry Channels ---
    if not args.skip_bathy_combine:
        print("\n--- Step 4: Starting post-processing (combining bathymetry channels) ---")
        process_frame_channels_in_subfolders(args.output_folder)
        print("\n--- Post-processing completed. ---")
    else:
        print("\n--- Step 4: Skipping bathymetry channel combination as requested. ---")

