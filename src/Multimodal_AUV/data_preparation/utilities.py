import os
import pandas as pd
import csv

def is_geotiff(file: str) -> bool:
    """Checks if a file is a GeoTIFF."""
    return file.lower().endswith(('.tif', '.tiff'))

def filter_csv_by_image_names(csv_file_path: str, image_folder_path: str) -> pd.DataFrame:
    """
    Loads a CSV file and filters rows where 'Image_Name' corresponds to an image file
    present in the specified image folder.

    Args:
        csv_file_path (str): Path to the input CSV file.
        image_folder_path (str): Path to the folder containing image files.

    Returns:
        pd.DataFrame: A DataFrame containing only the filtered rows.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        print(f"Error loading CSV {csv_file_path}: {e}")
        return pd.DataFrame()

    # Get a set of image names from the folder for efficient lookup
    if os.path.exists(image_folder_path):
        image_names_in_folder = set(os.listdir(image_folder_path))
    else:
        print(f"Warning: Image folder not found at {image_folder_path}. No image filtering will occur.")
        image_names_in_folder = set()

    if 'Image_Name' in df.columns:
        filtered_df = df[df['Image_Name'].apply(lambda x: x in image_names_in_folder)]
        print(f"Filtered CSV to {len(filtered_df)} rows based on images in {image_folder_path}")
        return filtered_df
    else:
        print("Error: 'Image_Name' column not found in CSV. Returning original DataFrame.")
        return df

def update_csv_path(csv_file_path: str, old_prefix: str, new_prefix: str):
    """
    Reads a CSV file, updates the 'path' column by replacing old_prefix with new_prefix,
    and saves the modified CSV back to the same file.

    Args:
        csv_file_path (str): The path to the CSV file.
        old_prefix (str): The string to be replaced.
        new_prefix (str): The replacement string.
    """
    rows = []
    header = None

    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            if 'path' not in header:
                raise ValueError("'path' column not found in CSV header.")

            path_index = header.index('path')
            for row in reader:
                if row: # Check the row is not empty
                    # Ensure the row has enough columns before accessing path_index
                    if len(row) > path_index:
                        row[path_index] = row[path_index].replace(old_prefix, new_prefix)
                rows.append(row)

        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)

        print(f"CSV file '{csv_file_path}' updated successfully.")

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")