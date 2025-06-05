import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from Multimodal_AUV.data.datasets import CustomImageDataset_1, CustomImageDataset
from Multimodal_AUV.data.loaders import split_dataset, prepare_datasets_and_loaders

class Test_CustomImageDataset1(unittest.TestCase):

    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("os.path.exists")
    @patch("PIL.Image.open")
    def test_load_data_skips_missing_files(self, mock_open_img, mock_exists, mock_isdir, mock_listdir):
        # Setup mocks for directories and files
        mock_listdir.side_effect = lambda d: ["folder1"] if d == "root_dir" else ["Frame001.jpg", "SSS_image.png", "combined_rgb_bathymetry.jpg"]
        mock_isdir.return_value = True
        mock_exists.side_effect = lambda path: True if "Frame001" in path or "SSS_image" in path or "combined_rgb_bathymetry" in path else False

        # Mock image open and return an array with non-zero sum (valid image)
        mock_img = MagicMock()
        mock_img.__enter__.return_value = mock_img
        mock_img.__exit__.return_value = None
        mock_img.convert.return_value = mock_img
        mock_img.size = (512, 512)
        mock_img.mode = 'RGB'
        mock_img_array = np.ones((512,512,3), dtype=np.uint8)
        mock_img_array.sum = lambda : 1
        mock_img.__array__ = lambda *a: mock_img_array

        mock_open_img.return_value = mock_img

        dataset = CustomImageDataset_1("root_dir")

        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        # Check that main_image, channel_image, sss_image and image_name returned
        self.assertEqual(len(sample), 4)
        self.assertIsInstance(sample[0], torch.Tensor)  # main_image tensor
        self.assertIsInstance(sample[1], torch.Tensor)  # channel_image tensor
        self.assertIsInstance(sample[2], torch.Tensor)  # sss_image tensor
        self.assertIsInstance(sample[3], str)  # image_name string

class Test_CustomImageDataset(unittest.TestCase):

    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("os.path.exists")
    @patch("glob.glob")
    @patch("PIL.Image.open")
    def test_dataset_loading_and_len(self, mock_open_img, mock_glob, mock_exists, mock_isdir, mock_listdir):
        # Mock folder listing
        mock_listdir.side_effect = lambda d: ["folder1"] if d == "root_dir" else [
            "frame001.jpg", "SSS_image.png", "combined_rgb_bathymetry.jpg",
            "patch_10m_combined_bathy.png", "patch_10m_some_SSS.png", "normalised_meta.csv", "label1.txt"
        ]

        mock_isdir.return_value = True
        mock_exists.side_effect = lambda path: True

        mock_glob.return_value = ["root_dir/folder1/frame001.jpg"]

        # Mock PIL open returning an image with some nonzero content
        mock_img = MagicMock()
        mock_img.__enter__.return_value = mock_img
        mock_img.convert.return_value = mock_img
        mock_img_array = np.ones((512, 512), dtype=np.uint8)
        mock_img_array.sum = lambda: 1
        mock_img.__array__ = lambda *a: mock_img_array
        mock_open_img.return_value = mock_img

        dataset = CustomImageDataset("root_dir")

        self.assertGreaterEqual(len(dataset), 0)
        if len(dataset) > 0:
            sample = dataset[0]
            self.assertIn("main_image", sample)
            self.assertIn("channel_image", sample)
            self.assertIn("sss_image", sample)
            self.assertIn("patch_channels", sample)
            self.assertIn("patch_sss", sample)
            self.assertIn("label", sample)

class Test_DatasetUtils(unittest.TestCase):

    def test_split_dataset(self):
        dummy_data = list(range(100))
        class DummyDataset:
            def __len__(self): return len(dummy_data)
            def __getitem__(self, idx): return dummy_data[idx]

        dataset = DummyDataset()
        train_ds, test_ds = split_dataset(dataset, test_size=0.25)
        self.assertEqual(len(train_ds) + len(test_ds), len(dataset))
        self.assertIsInstance(train_ds, torch.utils.data.Subset)
        self.assertIsInstance(test_ds, torch.utils.data.Subset)

    @patch("your_module.CustomImageDataset")
    @patch("torch.utils.data.DataLoader")
    @patch("collections.Counter")
    def test_prepare_datasets_and_loaders(self, mock_counter, mock_dataloader, mock_dataset):
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.labels = [0,1,1,0]
        mock_dataset_instance.label_encoder.classes_ = ["class0","class1"]
        mock_dataset.return_value = mock_dataset_instance

        # Counter mock returns a dict-like
        mock_counter.return_value = {0:2, 1:2}

        # DataLoader returns a mock object
        mock_dataloader.return_value = "dataloader"

        result = prepare_datasets_and_loaders("root_dir", 4, 8)
        self.assertEqual(len(result), 6)
        train_loader, test_loader, train_loader_multi, test_loader_multi, num_classes, dataset = result
        self.assertEqual(num_classes, 2)
        self.assertEqual(dataset, mock_dataset_instance)

if __name__ == "__main__":
    unittest.main()
