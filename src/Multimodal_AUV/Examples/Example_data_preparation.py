import os
import csv
import shutil
from PIL import Image 
import numpy as np 
import pandas as pd 
import argparse 
import sys
import subprocess
import glob
import skimage
from skimage import io, exposure 
from scipy import ndimage 
import matplotlib.pyplot as plt 
import re
import platform
import json
import pyproj
import utm

#Import module specific functions
from Multimodal_AUV.data_preparation.utilities import is_geotiff, filter_csv_by_image_names, update_csv_path
from Multimodal_AUV.data_preparation.geospatial import get_pixel_resolution, extract_grid_patch
from Multimodal_AUV.data_preparation.image_processing import process_frame_channels_in_subfolders



def preprocess_optical_images(raw_images_path: str, processed_images_save_folder: str,
                              exiftool_executable_path: str, image_enhancement_method: str = 'AverageSubtraction') -> pd.DataFrame:
    """
    Processes JPG images from AUV Grasshopper camera. Corrects illumination, extracts metadata,
    and converts coordinates to decimal.

    Args:
        raw_images_path (str): Path to the folder containing raw JPG images (can contain subfolders).
        processed_images_save_folder (str): Path to the folder where processed images and
                                            the metadata CSV will be saved.
        exiftool_executable_path (str): Path to the directory containing exiftool.exe.
        image_enhancement_method (str): Image Enhancement method, 'CLAHE' or 'AverageSubtraction'.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted and processed metadata.
                      Columns include 'Image_Name', 'path', 'easting', 'northing', 'altitude', 'depth',
                      'heading', 'lat' (decimal), 'lon' (decimal), 'pitch', 'roll',
                      'surge', 'sway', 'label'.
    """
    print(f"\n--- Starting optical image pre-processing from '{raw_images_path}' ---")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    print(f"Input raw_images_path: '{raw_images_path}'")
    print(f"Input processed_images_save_folder: '{processed_images_save_folder}'")
    print(f"Input exiftool_executable_path: '{exiftool_executable_path}'")
    print(f"Input image_enhancement_method: '{image_enhancement_method}'")

    # Validate raw_images_path existence and type
    if not os.path.exists(raw_images_path):
        print(f"Error: The specified raw images path '{raw_images_path}' does NOT exist. Please check the path and permissions.")
        return pd.DataFrame()
    if not os.path.isdir(raw_images_path):
        print(f"Error: The specified raw images path '{raw_images_path}' is NOT a directory. Please provide a valid folder path.")
        return pd.DataFrame()
    print(f"Validated: raw_images_path '{raw_images_path}' exists and is a directory.")

    # Ensure save folder exists
    try:
        os.makedirs(processed_images_save_folder, exist_ok=True)
        print(f"Ensured processed images save folder: '{processed_images_save_folder}'")
        if not os.path.exists(processed_images_save_folder) or not os.path.isdir(processed_images_save_folder):
             print(f"Error: Processed images save folder '{processed_images_save_folder}' could not be created or is not accessible after creation attempt.")
             sys.exit(1)
    except OSError as error:
        print(f"Error creating save folder '{processed_images_save_folder}': {error}")
        sys.exit(1)

    # List files of JPGs recursively
    search_pattern = os.path.join(raw_images_path, '**', '*.jpg')
    print(f"Searching for JPG images using pattern: '{search_pattern}' (recursive=True)")
    files = glob.glob(search_pattern, recursive=True)

    print(f"Found {len(files)} JPG images in '{raw_images_path}' and its subfolders.")
    if not files:
        print(f"Warning: No JPG images found. Skipping optical image pre-processing.")
        return pd.DataFrame()
    else:
        print(f"First 5 found image paths (if any):")
        for i, f in enumerate(files[:5]):
            print(f"  [{i+1}] {f}")
        print(f"Last 5 found image paths (if more than 5):")
        for i, f in enumerate(files[-5:]):
            if len(files) > 5 and i < (len(files) - 5): continue # Skip if already printed
            print(f"  [{i+1}] {f}")

    # Determine image dimensions from the first image
    h, w, d = 0, 0, 0
    if files:
        try:
            first_image_path = files[0]
            print(f"Attempting to read first image '{first_image_path}' to determine dimensions.")
            with Image.open(first_image_path) as img:
                h, w = img.height, img.width
                d = 3 # Assuming RGB images
            print(f"Determined image dimensions: Height={h}, Width={w}, Channels={d}")
        except IndexError:
            print("Error: No images in 'files' list, cannot determine dimensions.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error reading first image '{first_image_path}' to determine dimensions: {e}")
            return pd.DataFrame()
    else:
        print("No files found, skipping image dimension determination.")
        return pd.DataFrame()

    #Set the number of files and initialise an empty array of the image dimensions
    N = len(files)
    arr = np.zeros((h, w, d), float)
    print(f"Initialized average image array of shape {arr.shape} for AverageSubtraction.")

    #Go through each folder and get the average image intensity
    folder_average_images = {} # To store average image for each folder
    if image_enhancement_method == "AverageSubtraction":
        print("Calculating folder-specific average images for AverageSubtraction...")
        
        # Group files by their parent directory
        files_by_folder = {}
        for file_path in files:
            folder_path = os.path.dirname(file_path)
            if folder_path not in files_by_folder:
                files_by_folder[folder_path] = []
            files_by_folder[folder_path].append(file_path)
        
        for folder_path, folder_files in files_by_folder.items():
            print(f"Processing folder: '{folder_path}' with {len(folder_files)} images.")
            
            # Initialize average array for the current folder
            folder_arr = np.zeros((h, w, d), float)
            processed_for_avg_count = 0

            #Open the image, store its values in the array
            for im_file in folder_files:
                try:
                    with Image.open(im_file).convert('RGB') as img_pil:
                        imarr = np.array(img_pil, dtype=float)
                    if imarr.shape[:2] == (h, w):
                        folder_arr += imarr
                        processed_for_avg_count += 1
                    else:
                        print(f"Warning: Image '{im_file}' has inconsistent dimensions ({imarr.shape[:2]} vs expected {h, w}). Skipping for average calculation in its folder.")
                except Exception as e:
                    print(f"Warning: Could not read image '{im_file}' for average calculation: {e}. Skipping this image.")
            #Then divide by this 
            if processed_for_avg_count > 0:
                folder_average_images[folder_path] = folder_arr / processed_for_avg_count
                print(f"Successfully processed {processed_for_avg_count} images for average calculation in folder '{folder_path}'.")
                
                # Save the folder-specific average image
                avg_image_uint8 = np.array(np.round(folder_average_images[folder_path]), dtype=np.uint8)
                average_image_path = os.path.join(processed_images_save_folder, os.path.basename(folder_path) + "_Average.png")
                try:
                    out = Image.fromarray(avg_image_uint8, mode="RGB")
                    out.save(average_image_path)
                    print(f"Saved folder-specific average image to '{average_image_path}'")
                except Exception as e:
                    print(f"Error saving folder-specific average image to '{average_image_path}': {e}")
            else:
                print(f"No images processed for average calculation in folder '{folder_path}'. Average subtraction will be skipped for images in this folder.")
    


    # Create a dataframe to save metadata
    df = pd.DataFrame(columns=['Image_Name', 'path', 'easting', 'northing', 'altitude', 'depth', 'heading', 'lat', 'lon', 'pitch', 'roll', 'surge', 'sway', 'label'])
    print(f"Initialized metadata DataFrame with columns: {df.columns.tolist()}")

    

    # Construct the full path to the exiftool executable
    if platform.system() == "Windows":
        # On Windows, the executable typically has a .exe extension
        # If exiftool_executable_path is provided, use it. Otherwise, assume 'exiftool.exe' is in PATH.
        exiftool_command_name = exiftool_executable_path if exiftool_executable_path else "exiftool.exe"
    elif platform.system() == "Linux" or platform.system() == "Darwin": # Darwin is macOS
        # On Linux/macOS, it's typically just 'exiftool' and found in PATH
        # If exiftool_executable_path is provided, use it. Otherwise, assume 'exiftool' is in PATH.
        exiftool_command_name = "exiftool"

    else:
        print(f"Warning: Unsupported operating system '{platform.system()}'. Cannot determine ExifTool command.")
        sys.exit(1)
    print(f"Attempting to use ExifTool command: '{exiftool_command_name}'")



    print(f"Extracting metadata using ExifTool for {len(files)} files...")
    all_raw_metadata = []

    #Try to get the metadata using exiftool in cmd line structure
    try:
        command = [exiftool_command_name, '-G0', '-j', '-File:Comment']
        command.extend(files)

       

        process = subprocess.run(command, capture_output=True, text=True, check=True, shell=(platform.system() == "Windows") ) # ONLY shell=True on Windows 

        all_raw_metadata = json.loads(process.stdout)

        #Get the number of metadata found
        print(f"ExifTool returned {len(all_raw_metadata)} metadata entries.")
        if len(all_raw_metadata) != len(files):
            print(f"Warning: Number of metadata entries ({len(all_raw_metadata)}) does not match number of files ({len(files)}). Some files may have failed metadata extraction.")

    #Error handling
    except FileNotFoundError:
        # This occurs if the exiftool_command_name (e.g., "exiftool.exe" or "exiftool")
        # is not found in the system's PATH.
        print(f"Error: ExifTool command '{exiftool_command_name}' not found.")
        if platform.system() == "Windows":
            print("Please ensure ExifTool is installed and 'exiftool.exe' is in your system's PATH, or provide the full path to 'exiftool.exe' via the 'exiftool_executable_path' argument.")
        elif platform.system() == "Linux" or platform.system() == "Darwin":
            print("Please ensure ExifTool is installed via your package manager (e.g., `sudo apt install libimage-exiftool-perl` on Ubuntu/Debian) and is in your system's PATH, or provide the full path to the 'exiftool' executable via the 'exiftool_executable_path' argument.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running ExifTool command: {e}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        print("This usually indicates an issue with ExifTool's execution or the arguments provided.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding ExifTool JSON output: {e}")
        print(f"Raw ExifTool output (first 500 chars): {process.stdout[:500] if 'process' in locals() else 'N/A'}")
        print("This might mean ExifTool didn't output valid JSON, possibly due to an error.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during ExifTool execution: {e}")
        sys.exit(1)
   

    # Filter for successful extractions and align with `files` list
    processed_files_with_metadata = []
    metadata_dicts = []
    for i, file_path in enumerate(files):
        found_meta = None
        for meta_entry in all_raw_metadata:
            # ExifTool's JSON output for 'SourceFile' typically matches the input path directly.
            # On Windows, path separators might be '\' in input files and '/' in ExifTool output,
            # or vice-versa, depending on how `glob` returns them and how ExifTool handles paths.
            # Using os.path.normpath to normalize for comparison.
            if os.path.normpath(meta_entry.get('SourceFile', '')) == os.path.normpath(file_path):
                found_meta = meta_entry
                break

        if found_meta and 'File:Comment' in found_meta: # Ensure comment exists
            processed_files_with_metadata.append(file_path)
            metadata_dicts.append(found_meta)
        else:
            print(f"Warning: No valid 'File:Comment' metadata entry found for '{file_path}'. Skipping this image for further processing.")

    if not processed_files_with_metadata:
        print("No valid 'File:Comment' metadata extracted for any images after filtering. Returning empty DataFrame.")
        return pd.DataFrame()
    else:
        print(f"Proceeding with {len(processed_files_with_metadata)} images that have 'File:Comment' metadata.")


    # Loop over all photos for which metadata was successfully extracted
    for i, file_path in enumerate(processed_files_with_metadata):
        current_metadata = metadata_dicts[i]
        image_basename = os.path.basename(file_path)
        
        # Initialize variables with default NaN/empty string
        altitude, depth, heading, latCor, lonCor, pitch, roll, surge, sway = [np.nan] * 9

        #Extract the metadata
        comment = current_metadata.get('File:Comment', '')
        if comment:
            try:
                # searches the comment to look for something in between '<altitude>(.*)</altitude>' etc. and saves it
                # Using .group(1) will raise an AttributeError if no match is found, so we add checks.
                altitude_match = re.search('<altitude>(.*)</altitude>', comment)
                altitude = float(altitude_match.group(1)) if altitude_match else np.nan

                depth_match = re.search('<depth>(.*)</depth>', comment)
                depth = float(depth_match.group(1)) if depth_match else np.nan

                heading_match = re.search('<heading>(.*)</heading>', comment)
                heading = float(heading_match.group(1)) if heading_match else np.nan

                pitch_match = re.search('<pitch>(.*)</pitch>', comment)
                pitch = float(pitch_match.group(1)) if pitch_match else np.nan

                roll_match = re.search('<roll>(.*)</roll>', comment)
                roll = float(roll_match.group(1)) if roll_match else np.nan

                surge_match = re.search('<surge>(.*)</surge>', comment)
                surge = float(surge_match.group(1)) if surge_match else np.nan

                sway_match = re.search('<sway>(.*)</sway>', comment)
                sway = float(sway_match.group(1)) if sway_match else np.nan

                lat_match = re.search('<lat>(.*)</lat>', comment)
                lon_match = re.search('<lon>(.*)</lon>', comment)

                lat_str = lat_match.group(1) if lat_match else None
                lon_str = lon_match.group(1) if lon_match else None

                if lat_str and lon_str:
                    # These convert into latitude and longitude (decimal degrees)
                    signlat = 1
                    if lat_str.strip().upper().endswith("S"): # Use .strip().upper() for robustness
                        signlat = -1
                    lenlat = len(lat_str) # Use len of the string itself
                    latCor = signlat * (float(lat_str[:2]) + float(lat_str[2:lenlat-1])/60.0) # Corrected slice: lenlat-1 to exclude last char (N/S)

                    signlon = 1
                    if lon_str.strip().upper().endswith("W"): # Use .strip().upper() for robustness
                        signlon = -1
                    lenlon = len(lon_str) # Use len of the string itself
                    lonCor = signlon * (float(lon_str[:3]) + float(lon_str[3:lenlon-1])/60.0) # Corrected slice: lenlon-1 to exclude last char (E/W)

                    
                    # Convert Lat/Lon to UTM Easting/Northing here 
                    if pd.notna(latCor) and pd.notna(lonCor):
                        try:
                            # Auto-determine UTM zone
                            utm_zone = int(np.floor((lonCor + 180) / 6) + 1)
                            is_northern = (latCor >= 0)

                            # Define the UTM projector from the utm above 
                            utm_proj = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84', north=is_northern)
                            
                            #Calculate the utm
                            easting_utm, northing_utm = utm_proj(lonCor, latCor)
                            print(f"Converted to UTM: Easting={easting_utm}, Northing={northing_utm}, UTM Zone={utm_zone}{'N' if is_northern else 'S'}")
                        except Exception as utm_e:
                            print(f"Error converting Lat/Lon to UTM for '{image_basename}': {utm_e}. Easting/Northing will be NaN.")
                            easting_utm, northing_utm = np.nan, np.nan
                    else:
                        print(f"Lat/Lon are NaN for '{image_basename}', skipping UTM conversion.")

            #Error handling
            except AttributeError as ae:
                print(f"Error: A required metadata tag was not found in 'File:Comment' for '{image_basename}'. Check regex patterns. Error: {ae}")
                # Set coordinates to NaN if parsing failed for them
                latCor, lonCor = np.nan, np.nan
            except ValueError as ve:
                print(f"Error: Could not convert extracted metadata to float for '{image_basename}'. Error: {ve}")
                latCor, lonCor = np.nan, np.nan # Set to NaN if conversion fails
            except Exception as e:
                print(f"An unexpected error occurred during comment parsing for '{image_basename}': {e}")
                latCor, lonCor = np.nan, np.nan # Set to NaN for safety
        else:
            print(f"Warning: No 'File:Comment' found for '{image_basename}'. Metadata will be NaN.")

        # Convert depth to negative if it's an absolute value (e.g., in meters below surface)
        display_depth = str(-float(depth)) if pd.notna(depth) else ''

        # Apply image enhancements
        save_image_path = None
        try:
            with Image.open(file_path).convert('RGB') as img_pil:
                im1 = np.array(img_pil, dtype=float)

            out2 = None
            if image_enhancement_method == "AverageSubtraction":
                # Get the average image for the current file's folder
                current_folder_path = os.path.dirname(file_path)
                folder_avg_img = folder_average_images.get(current_folder_path)

                #Save the corrected image
                if folder_avg_img is not None and folder_avg_img.shape == im1.shape:
                    imcor = im1 - folder_avg_img
                    out2 = skimage.exposure.rescale_intensity(imcor, out_range='uint8')
                else:
                    print(f"Warning: No valid folder average image found for '{current_folder_path}' or dimensions mismatch. Skipping AverageSubtraction for '{image_basename}'. Saving original.")
                    out2 = im1.astype(np.uint8)
            elif image_enhancement_method == "CLAHE":
                # For CLAHE, ensure the input is uint8 and grayscale for direct skimage application
                # or apply it channel-wise if a color image is desired.
                # Assuming here that im1 is already float, convert to uint8 for CLAHE.
                im1_uint8 = im1.astype(np.uint8)
                # If it's a color image, apply CLAHE to each channel or convert to grayscale.
                # For simplicity, applying to luminosity channel or first channel if grayscale.
                # For RGB, CLAHE is usually applied to the V channel of HSV or L of Lab color space.
                # Here, a simple approach: if RGB, convert to grayscale, apply CLAHE, then convert back.
                # Or, apply channel-wise, which might change color balance.
                # Let's assume applying to grayscale equivalent for now, or channel-wise if needed.
                if im1_uint8.ndim == 3 and im1_uint8.shape[2] == 3: # RGB image
                    # Convert to grayscale for CLAHE, then back to RGB (simple approach)
                    img_gray = skimage.color.rgb2gray(im1_uint8)
                    equalized_gray = skimage.exposure.equalize_adapthist(img_gray)
                    # Convert back to RGB (by replicating channels) and rescale
                    out2 = (skimage.color.gray2rgb(equalized_gray) * 255).astype(np.uint8)
                else: # Grayscale image or single channel
                    out2 = skimage.exposure.rescale_intensity(
                        skimage.exposure.equalize_adapthist(im1_uint8),
                        out_range='uint8'
                    )
                print(f"Applied CLAHE to '{image_basename}'.")
            else:
                print(f"Warning: Unknown image enhancement method '{image_enhancement_method}'. Saving original image.")
                out2 = im1.astype(np.uint8)

            # Save processed photo
            if out2 is not None:
                save_image_path = os.path.join(processed_images_save_folder, image_basename)
                io.imsave(save_image_path, out2)
            else:
                print(f"Error: Processed image array for '{image_basename}' is None after enhancement. Skipping save.")

        except Exception as e:
            print(f"Error applying enhancement or saving image '{image_basename}': {e}. Skipping image processing for this file.")
            save_image_path = file_path # Fallback: use original path if processing fails

        # Prepare data for DataFrame row nice and clean
        row_data = {
            'Image_Name': image_basename,
            'path': save_image_path,
            'easting': easting_utm if pd.notna(easting_utm) else np.nan, # Assuming lon as easting (REVISIT FOR UTM)
            'northing': northing_utm if pd.notna(northing_utm) else np.nan, # Assuming lat as northing (REVISIT FOR UTM)
            'altitude': str(altitude) if pd.notna(altitude) else '',
            'depth': display_depth,
            'heading': str(heading) if pd.notna(heading) else '',
            'lat': str(latCor) if pd.notna(latCor) else '',
            'lon': str(lonCor) if pd.notna(lonCor) else '',
            'pitch': str(pitch) if pd.notna(pitch) else '',
            'roll': str(roll) if pd.notna(roll) else '',
            'surge': str(surge) if pd.notna(surge) else '',
            'sway': str(sway) if pd.notna(sway) else '',
            'label': "unlabelled" # Default label
        }
       
        #Create a df of this 
        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

    #And save and return this df
    output_csv_path = os.path.join(processed_images_save_folder, 'coords.csv')
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"Metadata saved to '{output_csv_path}' (Total {df.shape[0]} entries).")
    except Exception as e:
        print(f"Error saving metadata CSV to '{output_csv_path}': {e}")
    print("--- Optical image pre-processing completed. ---")

    return df


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
                              'path' (full path to original image), and optionally 'label', 'lat', 'lon'.
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
    #Print some parameters
    print(f"CSV file: '{csv_file_path}'")
    print(f"Original images folder: '{original_images_folder}'")
    print(f"GeoTIFF files: {geotiff_files_paths}")
    print(f"Output root folder: '{output_root_folder}'")
    print(f"Window size for grid patches: {window_size_meters} meters")

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
    for row_idx, row in enumerate(rows):
        image_name_original = row.get('Image_Name', f"Unknown_Image_{row_idx}")
        print(f"\n--- Attempting to process entry {row_idx+1} for image: '{image_name_original}' ---")

        try:
            # Retrieve original image path, potentially correcting it if only basename is in CSV
            original_image_filename_from_csv = row.get('Image_Name')
            original_image_full_path = row.get('path')

            # Robustly derive full path if 'path' column is relative or just filename
            if original_image_full_path and not os.path.isabs(original_image_full_path):
                # If the 'path' in CSV is relative to original_images_folder
                original_image_full_path = os.path.join(original_images_folder, os.path.basename(original_image_full_path))
            elif not original_image_full_path and original_image_filename_from_csv:
                # If 'path' is missing, assume Image_Name is filename and combine with original_images_folder
                original_image_full_path = os.path.join(original_images_folder, original_image_filename_from_csv)
            
            if not original_image_full_path or not os.path.exists(original_image_full_path):
                print(f"Warning: Original optical image not found at '{original_image_full_path}' (or path missing) for '{image_name_original}'. Skipping this entry.")
                continue # Skip to next row

            label = row.get('label', "unlabelled")

            print(f"--- Processing entry for image: '{image_name_original}' ---")

            # Create a dedicated output subfolder for this image entry.
            output_folder_for_entry = os.path.join(output_root_folder, os.path.splitext(image_name_original)[0])
            os.makedirs(output_folder_for_entry, exist_ok=True)
            print(f"Created/ensured output subfolder: '{output_folder_for_entry}'")

            # Copy the original optical image into its dedicated subfolder
            try:
                shutil.copy(original_image_full_path, output_folder_for_entry)
                print(f"Copied original image '{image_name_original}' to '{output_folder_for_entry}'")
            except Exception as e: # Catch all exceptions during copy
                print(f"Error copying original image '{original_image_full_path}' to '{output_folder_for_entry}': {e}. Skipping copy for this entry.")

            # Save the current row's data (excluding image-specific paths/names) as a CSV file.
            csv_output_path = os.path.join(output_folder_for_entry, 'row_data.csv')
            try:
                # Create a mutable copy of the row for modifications
                current_row_data_for_csv = dict(row)

                # Dynamically filter out columns that are redundant or not desired in `row_data.csv`.
                # Removed 'easting', 'northing', 'lat', 'lon' and derived columns from exclusion logic
                columns_to_exclude = ['Image_Name', 'path'] # original path
                
                relevant_header = [h for h in current_row_data_for_csv.keys() if h not in columns_to_exclude]
                relevant_values = [current_row_data_for_csv[h] for h in relevant_header]

                with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(relevant_header)
                    writer.writerow(relevant_values)
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

            # Extract easting and northing from the row for extract_grid_patch if needed
            easting_from_row = row.get('easting')
            northing_from_row = row.get('northing')

            if easting_from_row is None or northing_from_row is None:
                print(f"Error: 'easting' or 'northing' data missing for '{image_name_original}', cannot extract grid patch. Skipping this GeoTIFF processing.")
                continue # Skip to next row for GeoTIFF processing
            
            try:
                # Convert easting/northing to float
                easting_from_row = float(easting_from_row)
                northing_from_row = float(northing_from_row)
            except ValueError:
                print(f"Error: 'easting' or 'northing' values for '{image_name_original}' are not valid numbers. Skipping GeoTIFF processing.")
                continue


            for geotiff_path in geotiff_files_paths:
                # Passing easting_from_row and northing_from_row instead of easting/northing
                extracted_patch_info = extract_grid_patch(geotiff_path, easting_from_row, northing_from_row, window_size_meters)

                if extracted_patch_info:
                    data = extracted_patch_info['data']
                    geotiff_filename_base = extracted_patch_info['geotiff_filename_base']
                    geotiff_type = extracted_patch_info['geotiff_type']

                    filename_parts = geotiff_filename_base.split("_")
                    final_three_parts = "_".join(filename_parts[-3:])

                    # Changed output_image_name to no longer include easting/northing
                    output_image_name = f"grid_{final_three_parts}.png"
                    output_image_path = os.path.join(output_folder_for_entry, output_image_name)

                    try:
                        if geotiff_type.lower() == 'bathy':
                            if data.ndim == 3 and data.shape[0] >= 2:
                                img_ch1 = Image.fromarray(data[0].astype(np.uint8))
                                img_ch2 = Image.fromarray(data[1].astype(np.uint8))

                                img_ch1.save(os.path.join(output_folder_for_entry, "output_channel_1.png"))
                                img_ch2.save(os.path.join(output_folder_for_entry, "output_channel_2.png"))
                                print(f"Saved 'output_channel_1.png' and 'output_channel_2.png' from Bathy to '{output_folder_for_entry}'")
                            else:
                                print(f"Warning: Bathy GeoTIFF '{geotiff_filename_base}' has insufficient (expected >=2) or incorrect dimensions ({data.ndim}). Skipping channel save.")
                        else:
                            if data.ndim == 3:
                                img = Image.fromarray(data[0].astype(np.uint8))
                            else:
                                img = Image.fromarray(data.astype(np.uint8))
                            img.save(output_image_path)
                            print(f"Saved '{output_image_name}' from {geotiff_type} to '{output_folder_for_entry}'")

                    except Exception as e:
                        print(f"Error saving image patch from '{geotiff_path}' to '{output_folder_for_entry}': {e}")
                else:
                    print(f"Skipping patch extraction for '{os.path.basename(geotiff_path)}' due to previous warnings/errors in `extract_grid_patch`.")
        except Exception as e:
            # Catch any unexpected errors during the processing of a single CSV row
            print(f"Critical Error processing row for image '{image_name_original}': {e}. Skipping this row.")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess AUV sonar and optical image data for machine learning tasks. "
                    "Extracts sonar grid patches, copies original images, and organizes metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )

    # Required Arguments
    parser.add_argument(
        "--raw_optical_images_folder",
        type=str,
        required=True,
        help="Path to the folder containing raw JPG optical image files."
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
        help="The root directory where all processed and organized output data will be saved. "
             "This will also be the target for processed optical images and their metadata CSV."
    )
    parser.add_argument(
        "--exiftool_path",
        type=str,
        required=True,
        help="Path to the directory containing the exiftool.exe executable. E.g., 'C:/exiftool/'."
    )

    # Optional Arguments with Default Values
    parser.add_argument(
        "--window_size_meters",
        type=float,
        default=20.0,
        help="The desired side length (in meters) for the square patches extracted from GeoTIFFs."
    )
    parser.add_argument(
        "--image_enhancement_method",
        type=str,
        default="AverageSubtraction",
        choices=["AverageSubtraction", "CLAHE"],
        help="Method to enhance optical images: 'AverageSubtraction' or 'CLAHE'."
    )
    parser.add_argument(
        "--skip_bathy_combine",
        action="store_true",
        help="If set, the post-processing step to combine bathymetry channels will be skipped."
    )

    args = parser.parse_args()


    # --- Step 1: Pre-process Optical Images and Generate Metadata CSV ---
    print("\n--- Step 1: Pre-processing optical images and generating metadata CSV ---")
    # The processed images and their metadata CSV will be saved into the main output_folder
    ## The 'path' column in the generated CSV will point to these processed images.
    processed_metadata_df = preprocess_optical_images(
        raw_images_path=args.raw_optical_images_folder,
        processed_images_save_folder=args.output_folder, # Use main output folder for processed images
        exiftool_executable_path=args.exiftool_path,
        image_enhancement_method=args.image_enhancement_method
    )

    if processed_metadata_df.empty:
        print("Optical image pre-processing resulted in no valid metadata. Exiting.")
        sys.exit(1)

    # Set the CSV path for the subsequent steps to the newly generated one
    generated_csv_path = os.path.join(args.output_folder, 'coords.csv')
    if not os.path.exists(generated_csv_path):
        print(f"Error: Expected metadata CSV not found at '{generated_csv_path}' after pre-processing. Exiting.")
        sys.exit(1)
    args.csv_path = generated_csv_path # Update the CSV path for the next steps
    
    # The original_images_base_folder for process_and_save_data should now point to
    # the folder where the *processed* optical images are, which is args.output_folder.
    args.original_images_base_folder = args.output_folder
    
    print("\n--- Optical image pre-processing completed. ---")


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
