import os
import rasterio
from rasterio.windows import Window
from PIL import Image
import csv
import numpy as np # Added for potential array operations in extract_grid_patch
from typing import Union

def get_pixel_resolution(geotiff_path: str) -> tuple[Union[float, None], Union[float, None]]:
    """
    Retrieves the pixel resolution (x and y) from a GeoTIFF file.

    Args:
        geotiff_path (str): Path to the GeoTIFF file.

    Returns:
        tuple: A tuple containing the x and y resolutions (in georeferenced units, e.g., meters)
               or (None, None) if an error occurs.
    """
    try:
        with rasterio.open(geotiff_path) as dataset:
            transform = dataset.transform
            x_resolution = transform[0]
            y_resolution = transform[4] # transform[4] is typically negative for Y resolution in geospatial data
            return x_resolution, abs(y_resolution) # Return absolute value for Y resolution for clarity
    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening GeoTIFF {geotiff_path}: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while getting resolution for {geotiff_path}: {e}")
        return None, None


def extract_grid_patch(
    geotiff_path: str,
    easting: float,
    northing: float,
    window_size_meters: float
) -> Union[dict, None]:
    """
    Extracts a square data patch from a GeoTIFF centered at given Easting/Northing coordinates.

    Args:
        geotiff_path (str): Path to the GeoTIFF file.
        easting (float): Easting coordinate of the center of the patch.
        northing (float): Northing coordinate of the center of the patch.
        window_size_meters (float): The desired side length of the square patch in meters.

    Returns:
        Union[dict, None]: A dictionary containing:
            - 'data': The extracted NumPy array data from the GeoTIFF patch.
            - 'pixel_size_x': X resolution of the GeoTIFF.
            - 'pixel_size_y': Y resolution of the GeoTIFF.
            - 'geotiff_filename_base': Base name of the GeoTIFF file (without extension).
            - 'geotiff_type': 'Bathy' or other, derived from filename.
            - 'extracted_easting_center': The actual easting of the extracted patch's center.
            - 'extracted_northing_center': The actual northing of the extracted patch's center.
        Returns None if the patch cannot be extracted (e.g., coordinates out of bounds, no data).
    """
    try:
        with rasterio.open(geotiff_path) as src:
            # Get pixel sizes
            pixel_size_x = src.transform[0]
            pixel_size_y = abs(src.transform[4]) # Y resolution is usually negative

            # Convert window size from meters to pixels
            # Ensure window dimensions are at least 1 pixel
            window_size_pixels_horizontal = max(1, int(window_size_meters / pixel_size_x))
            window_size_pixels_vertical = max(1, int(window_size_meters / pixel_size_y))

            # Get row, col of the center coordinate
            # rasterio.index returns (row, col)
            row_pixel_center, col_pixel_center = src.index(easting, northing)

            # Calculate the top-left corner of the desired window in pixel coordinates
            row_start = row_pixel_center - (window_size_pixels_vertical // 2)
            col_start = col_pixel_center - (window_size_pixels_horizontal // 2)

            # Define the desired window
            desired_window = Window(
                col_start,
                row_start,
                window_size_pixels_horizontal,
                window_size_pixels_vertical
            )

            # Use rasterio.windows.get_data_window to get the valid data window within src bounds.
            # This handles cases where the desired window goes out of bounds by clipping it.
            # It returns the effective window that can be read.
            effective_window = desired_window.intersection(Window(0, 0, src.width, src.height))

            # Check if the effective window is empty (i.e., desired window was completely out of bounds)
            if effective_window.width <= 0 or effective_window.height <= 0:
                print(f"Warning: Desired window for Easting {easting}, Northing {northing} for {os.path.basename(geotiff_path)} is completely out of GeoTIFF bounds or results in zero size. Skipping.")
                return None

            # Read data from the effective window
            data = src.read(window=effective_window)

            # Handle cases where the data might contain fill values or no-data values
            # and effectively be empty (e.g., if it's over a mask)
            # You might want to refine this check based on your specific no-data values.
            if data.size == 0 or np.all(data == src.nodata) if src.nodata is not None else np.all(data == 0):
                print(f"Warning: No valid data found in the extracted window for Easting {easting}, Northing {northing} from {os.path.basename(geotiff_path)}. Skipping.")
                return None

            geotiff_filename_base = os.path.splitext(os.path.basename(geotiff_path))[0]
            # A more robust way to get geotiff_type is to check if 'Bathy' is in name.
            # Splitting by '_' and taking the last part is fragile if names change.
            # Assuming 'Bathy' is indicative of Bathymetry.
            geotiff_type = 'Bathy' if 'Bathy' in geotiff_filename_base else 'SSS' # Or other types

            # Calculate the actual center of the extracted patch in map coordinates
            # This is useful if the window was clipped at the edges of the raster
            extracted_easting_center, extracted_northing_center = src.xy(
                effective_window.row_off + effective_window.height // 2,
                effective_window.col_off + effective_window.width // 2
            )

            return {
                'data': data,
                'pixel_size_x': pixel_size_x,
                'pixel_size_y': pixel_size_y,
                'geotiff_filename_base': geotiff_filename_base,
                'geotiff_type': geotiff_type,
                'extracted_easting_center': extracted_easting_center, # Add actual center
                'extracted_northing_center': extracted_northing_center # Add actual center
            }

    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening GeoTIFF {geotiff_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during patch extraction for {geotiff_path} (Easting: {easting}, Northing: {northing}): {e}")
        return None
