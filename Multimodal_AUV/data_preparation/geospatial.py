import os
import rasterio
from rasterio.windows import Window
from PIL import Image
import csv
import numpy as np # Added for potential array operations in extract_grid_patch

def get_pixel_resolution(geotiff_path: str) -> tuple[float | None, float | None]:
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
) -> dict | None:
    """
    Extracts a square data patch from a GeoTIFF centered at given Easting/Northing coordinates.

    Args:
        geotiff_path (str): Path to the GeoTIFF file.
        easting (float): Easting coordinate of the center of the patch.
        northing (float): Northing coordinate of the center of the patch.
        window_size_meters (float): The desired side length of the square patch in meters.

    Returns:
        dict | None: A dictionary containing:
            - 'data': The extracted NumPy array data from the GeoTIFF patch.
            - 'pixel_size_x': X resolution of the GeoTIFF.
            - 'pixel_size_y': Y resolution of the GeoTIFF.
            - 'geotiff_filename': Base name of the GeoTIFF file (without extension).
            - 'geotiff_type': 'Bathy' or other, derived from filename.
        Returns None if the patch cannot be extracted (e.g., coordinates out of bounds, no data).
    """
    try:
        with rasterio.open(geotiff_path) as src:
            # Get pixel sizes
            pixel_size_x = src.transform[0]
            pixel_size_y = abs(src.transform[4]) # Y resolution is usually negative

            # Convert window size from meters to pixels
            window_size_pixels_horizontal = int(window_size_meters / pixel_size_x)
            window_size_pixels_vertical = int(window_size_meters / pixel_size_y)

            # Get row, col of the center coordinate
            row_pixel, col_pixel = src.index(easting, northing)

            # Define the window to read
            window = Window(
                col_pixel - (window_size_pixels_horizontal // 2),
                row_pixel - (window_size_pixels_vertical // 2),
                window_size_pixels_horizontal,
                window_size_pixels_vertical
            )

            # Ensure window is within image bounds
            if not src.window_intersects(window):
                print(f"Warning: Desired window for Easting {easting}, Northing {northing} is out of GeoTIFF bounds. Skipping.")
                return None

            # Read data from the window
            data = src.read(window=window)

            if not data.any():
                print(f"Warning: No data found in the extracted window for Easting {easting}, Northing {northing}. Skipping.")
                return None

            geotiff_filename_base = os.path.splitext(os.path.basename(geotiff_path))[0]
            geotiff_type = geotiff_filename_base.split("_")[-1]

            return {
                'data': data,
                'pixel_size_x': pixel_size_x,
                'pixel_size_y': pixel_size_y,
                'geotiff_filename_base': geotiff_filename_base,
                'geotiff_type': geotiff_type
            }

    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening GeoTIFF {geotiff_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during patch extraction for {geotiff_path}: {e}")
        return None