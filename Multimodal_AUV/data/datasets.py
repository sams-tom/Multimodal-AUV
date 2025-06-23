#For file and path handling:
import os
import glob
#For image processing
from PIL import Image
#For data handling
import numpy as np
#For label encoding aand dataset splitting
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#Load pytorch 
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import torch

#For file renaming
import re

#For tracking errors
import logging

class CustomImageDataset_1(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.data = []
        # Define a consistent transform for all images to Tensor
        self.tensor_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        # Separate transform for the main image with normalization
        self.main_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[62.19902423 / 255.0, 62.31835045 / 255.0, 61.53444229 / 255.0],
                                 std=[41.46890313 / 255.0, 43.39430715 / 255.0, 41.72083641 / 255.0])
        ])
        self._load_data()


    def _load_data(self):
        processed_folder_count = 0
        successful_load_count = 0

        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            processed_folder_count += 1

            main_image_path = self._find_main_image(folder_path)
            sss_image_path = self._find_sss_image(folder_path)
            channel_path = self._find_channel_image(folder_path)

            # Reject if any are missing or the channel is marked as 'empty_image.png'
            if (main_image_path is None or
                sss_image_path is None or
                channel_path in [None, "empty_image.png"]):
                continue

            # Verify all files exist on disk
            image_paths = [main_image_path, sss_image_path, channel_path]
            if not all(os.path.exists(p) for p in image_paths):
                continue

            # Verify image content (non-empty)
            is_valid = True
            for path in image_paths:
                try:
                    with Image.open(path) as img:
                        if np.array(img).sum() == 0:
                            is_valid = False
                            break
                except Exception as e:
                    print(f"Error reading image {path} in folder {folder}: {e}")
                    is_valid = False
                    break

            if not is_valid:
                continue

            # Append only if all are valid
            self.data.append({
                'main_image': main_image_path,
                'channel_image': channel_path,
                'sss_image': sss_image_path,
            })
            successful_load_count += 1

        print(f"Total folders successfully loaded: {successful_load_count} total folders processed: {processed_folder_count}")



    def _find_main_image(self, folder_path):
        matching_files = glob.glob(os.path.join(folder_path, "[fF]rame*.jpg"))
        if not matching_files:
            return None
        return matching_files[0]

    def _find_sss_image(self, folder_path):
        sss_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if "SSS" in f and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')) and "patch_" not in f]
        selected_sss = None
        max_nonzero = -1
        for image_path in sss_images:
            try:
                with Image.open(image_path).convert("L") as sss_image:
                    nonzero_count = np.count_nonzero(np.array(sss_image))
                    if nonzero_count > max_nonzero:
                        max_nonzero = nonzero_count
                        selected_sss = image_path
            except Exception as e:
                print(f"Error loading SSS image {image_path}: {e}")
        if selected_sss is None:
            print(f"No valid SSS image found in {folder_path}")
        return selected_sss

    def _find_channel_image(self, folder_path):
        path1 = os.path.join(folder_path, "combined_rgb_bathymetry.jpg")
        path2 = os.path.join(folder_path, "combined_channels.jpg")
        if os.path.exists(path1):
            return path1
        elif os.path.exists(path2):
            return path2
        else:
            print(f"Missing channel data in {folder_path}. Will load empty if accessed.")
            return "empty_image.png"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        images = {}
        empty_image = Image.new('RGB', (512, 512), color='black')
        empty_image_gray = Image.new('L', (512, 512), color='black')
        image_name = os.path.basename(data_item.get('main_image', ''))

        for key, path in data_item.items():
            if path:
                try:
                    with Image.open(path) as img:
                        if "sss" in key or "channel" in key and path == "empty_image.png": # Keep channel as grayscale if it's the placeholder
                            img = img.convert("L")
                        else:
                            img = img.convert("RGB")

                        if key == 'main_image':
                            transformed_img = self.main_transform(img)
                        else:
                            transformed_img = self.tensor_transform(img) # Apply tensor transform consistently
                        images[key] = transformed_img
                except FileNotFoundError:
                    print(f"Warning: Missing file {path}. Using empty image for {key}.")
                    images[key] = empty_image_gray if "sss" in key or "channel" in key else empty_image
                except Exception as e:
                    print(f"Error loading {path} for {key}: {e}. Using empty image.")
                    images[key] = empty_image_gray if "sss" in key or "channel" in key else empty_image
            else:
                images[key] = empty_image_gray if "sss" in key or "channel" in key else empty_image

        return (
            images.get('main_image'),
            images.get('channel_image'),
            images.get('sss_image'),
            image_name
        )

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.data_paths = [] # Renamed from 'data' to be more explicit about storing paths
        self.labels = []

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        self.transform_1 = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[62.19902423 / 255.0, 62.31835042 / 255.0, 61.53444229 / 255.0],
                std=[41.46890313 / 255.0, 43.39430715 / 255.0, 41.72083641 / 255.0],
            ),
        ])

        # NEW: Collect all unique patch sizes during initialization
        self.all_discovered_patch_sizes = set()

        all_labels = []
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path): continue

            main_image, sss_image = None, None
            try:
                main_image = glob.glob(os.path.join(folder_path, "*frame*.jpg"))
                if not main_image: raise FileNotFoundError("Main image not found")
                main_image = main_image[0]

                sss_candidates = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                                  if "SSS" in f and "patch_" not in f]
                if not sss_candidates: raise FileNotFoundError("SSS image not found")
                sss_image = max(sss_candidates, key=lambda x: np.count_nonzero(np.array(Image.open(x).convert("L"))))
            except Exception as e:
                logging.debug(f"Skipping folder {folder_path} due to missing main/SSS image: {e}")
                continue

            label = None
            try:
                label_files = [f for f in os.listdir(folder_path) if f.endswith(".txt") and not f.startswith("_")]
                if not label_files: raise FileNotFoundError("Label file not found")
                label_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
                label = os.path.splitext(label_files[0])[0]
            except Exception as e:
                logging.debug(f"Skipping folder {folder_path} due to missing label: {e}")
                continue

            channel_image = os.path.join(folder_path, "combined_rgb_bathymetry.jpg")
            if not os.path.exists(channel_image):
                logging.debug(f"Skipping folder {folder_path} due to missing channel image.")
                continue

            patch_channels_found = {}
            patch_sss_found = {}
            found_any_patch = False # Track if any patch of any size was found for this sample

            for file in os.listdir(folder_path):
                channel_match = re.match(r"patch_(\d+m)_combined_bathy\.png", file)
                sss_match = re.match(r"patch_(\d+m)_.*_SSS\.(png|jpg)", file)

                if channel_match:
                    size = channel_match.group(1)
                    patch_channels_found[size] = os.path.join(folder_path, file)
                    self.all_discovered_patch_sizes.add(size) # Learn the size
                    found_any_patch = True
                elif sss_match:
                    size = sss_match.group(1)
                    patch_sss_found[size] = os.path.join(folder_path, file)
                    self.all_discovered_patch_sizes.add(size) # Learn the size
                    found_any_patch = True

            if not found_any_patch:
                 logging.debug(f"Skipping folder {folder_path} as no patches were found.")
                 continue

            # Check for normalised_meta.csv
            extracted_data_path = os.path.join(folder_path, "normalised_meta.csv")
            if not os.path.exists(extracted_data_path):
                logging.debug(f"Skipping folder {folder_path} due to missing normalised_meta.csv.")
                continue


            self.data_paths.append({
                "main_image": main_image,
                "channel_image": channel_image,
                "sss_image": sss_image,
                "patch_channels": patch_channels_found,
                "patch_sss": patch_sss_found,
            })
            all_labels.append(label)

        if not self.data_paths:
            raise RuntimeError("No valid data samples found in root_dir. Check your data paths and filters.")

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_labels)
        self.labels = self.label_encoder.transform(all_labels)

        # Convert discovered patch sizes to a sorted list for consistent ordering
        self.all_discovered_patch_sizes = sorted(list(self.all_discovered_patch_sizes))
        logging.info(f"Discovered patch sizes: {self.all_discovered_patch_sizes}")


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        sample_paths = self.data_paths[idx]
        label = self.labels[idx]

        with Image.open(sample_paths["main_image"]).convert("RGB") as img:
            main_img = self.transform_1(img)

        with Image.open(sample_paths["channel_image"]).convert("RGB") as img:
            channel_img = self.transform(img)

        with Image.open(sample_paths["sss_image"]).convert("L") as img:
            sss_img = self.transform(img)

        # Define dummy tensors based on your transformations.
        # These will be used if a specific patch size is missing for a sample.
        dummy_channel_patch_tensor = torch.zeros(3, 512, 512) # RGB, 512x512
        dummy_sss_patch_tensor = torch.zeros(1, 512, 512)    # Grayscale, 512x512

        patch_channels_tensors = {}
        patch_sss_tensors = {}

        # Iterate over ALL discovered patch sizes from the dataset,
        # ensuring consistency across all __getitem__ calls
        for size in self.all_discovered_patch_sizes:
            # Handle channel patches
            channel_path = sample_paths["patch_channels"].get(size)
            if channel_path and os.path.exists(channel_path):
                try:
                    with Image.open(channel_path).convert("RGB") as img:
                        patch_channels_tensors[size] = self.transform(img)
                except Exception as e:
                    logging.warning(f"Error loading channel patch {channel_path}: {e}. Using dummy tensor.")
                    patch_channels_tensors[size] = dummy_channel_patch_tensor
            else:
                patch_channels_tensors[size] = dummy_channel_patch_tensor

            # Handle SSS patches
            sss_path = sample_paths["patch_sss"].get(size)
            if sss_path and os.path.exists(sss_path):
                try:
                    with Image.open(sss_path).convert("L") as img:
                        patch_sss_tensors[size] = self.transform(img)
                except Exception as e:
                    logging.warning(f"Error loading SSS patch {sss_path}: {e}. Using dummy tensor.")
                    patch_sss_tensors[size] = dummy_sss_patch_tensor
            else:
                patch_sss_tensors[size] = dummy_sss_patch_tensor

        return {
            "main_image": main_img,
            "channel_image": channel_img,
            "sss_image": sss_img,
            "patch_channels": patch_channels_tensors, # Now guaranteed to have all discovered sizes
            "patch_sss": patch_sss_tensors,         # Now guaranteed to have all discovered sizes
            "label": label,
        }

  