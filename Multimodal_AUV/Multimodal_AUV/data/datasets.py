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

#For file renaming
import re

class CustomImageDataset_1(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.data = []
        # Define a consistent transform for all images to Tensor
        self.tensor_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        # Separate transform for the main image with normalization
        self.main_transform = transforms.Compose([
            transforms.Resize((512, 512)),
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
            print(f"Processing folder: {folder}")

            main_image_path = self._find_main_image(folder_path)
            sss_image_path = self._find_sss_image(folder_path)
            channel_path = self._find_channel_image(folder_path)

            # Reject if any are missing or the channel is marked as 'empty_image.png'
            if (main_image_path is None or
                sss_image_path is None or
                channel_path in [None, "empty_image.png"]):
                print(f"Skipping folder {folder} due to missing or invalid images.")
                continue

            # Verify all files exist on disk
            image_paths = [main_image_path, sss_image_path, channel_path]
            if not all(os.path.exists(p) for p in image_paths):
                print(f"Skipping folder {folder} due to missing files on disk.")
                continue

            # Verify image content (non-empty)
            is_valid = True
            for path in image_paths:
                try:
                    with Image.open(path) as img:
                        if np.array(img).sum() == 0:
                            print(f"Skipping folder {folder} due to empty image: {path}")
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
            print(f"Successfully loaded data from folder: {folder}")

        print(f"Total folders processed: {processed_folder_count}")
        print(f"Total folders successfully loaded: {successful_load_count}")



    def _find_main_image(self, folder_path):
        matching_files = glob.glob(os.path.join(folder_path, "[fF]rame*.jpg"))
        if not matching_files:
            print(f"No main image found in {folder_path}")
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
        self.data = []
        self.labels = []
        self.extracted_data = []
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        self.transform_1 = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[62.19902423 / 255.0, 62.31835045 / 255.0, 61.53444229 / 255.0],
                std=[41.46890313 / 255.0, 43.39430715 / 255.0, 41.72083641 / 255.0],
            ),
        ])

        all_labels = []
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path): continue

            try:
                main_image = glob.glob(os.path.join(folder_path, "*frame*.jpg"))[0]
                sss_image = max(
                    [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                     if "SSS" in f and "patch_" not in f],
                    key=lambda x: np.count_nonzero(np.array(Image.open(x).convert("L")))
                )
            except Exception:
                continue

            try:
                label_files = [f for f in os.listdir(folder_path) if f.endswith(".txt") and not f.startswith("_")]
                label_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
                label = os.path.splitext(label_files[0])[0]
            except:
                continue

            channel_image = os.path.join(folder_path, "combined_rgb_bathymetry.jpg")
            if not os.path.exists(channel_image): continue

            # Find all patch_*m_combined_bathy.png and corresponding SSS
            patch_channels = {}
            patch_sss = {}
            for file in os.listdir(folder_path):
                if re.match(r"patch_\d+m_combined_bathy\.png", file):
                    size = re.search(r"patch_(\d+m)", file).group(1)
                    patch_channels[size] = os.path.join(folder_path, file)
                elif re.match(r"patch_\d+m_.*_SSS\.(png|jpg)", file):
                    size = re.search(r"patch_(\d+m)", file).group(1)
                    patch_sss[size] = os.path.join(folder_path, file)

            if not patch_channels or not patch_sss:
                continue

            extracted_data_path = os.path.join(folder_path, "normalised_meta.csv")
            if not os.path.exists(extracted_data_path): continue

           

            self.data.append({
                "main_image": main_image,
                "channel_image": channel_image,
                "sss_image": sss_image,
                "patch_channels": patch_channels,
                "patch_sss": patch_sss,
            })
            all_labels.append(label)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_labels)
        self.labels = self.label_encoder.transform(all_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        with Image.open(sample["main_image"]).convert("RGB") as img:
            main_img = self.transform_1(img)

        with Image.open(sample["channel_image"]).convert("RGB") as img:
            channel_img = self.transform(img)

        with Image.open(sample["sss_image"]).convert("L") as img:
            sss_img = self.transform(img)

        # Load dynamic patches
        patch_channels_tensor = {}
        patch_sss_tensor = {}

        for size, path in sample["patch_channels"].items():
            with Image.open(path).convert("RGB") as img:
                patch_channels_tensor[size] = self.transform(img)

        for size, path in sample["patch_sss"].items():
            with Image.open(path).convert("L") as img:
                patch_sss_tensor[size] = self.transform(img)

        return {
            "main_image": main_img,
            "channel_image": channel_img,
            "sss_image": sss_img,
            "patch_channels": patch_channels_tensor,
            "patch_sss": patch_sss_tensor,
            "label": label,
        }

  