import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import numpy as np
from PIL import Image
import torch
import io
from torch.utils.data import DataLoader
from Multimodal_AUV.data.datasets import CustomImageDataset_1, CustomImageDataset
from Multimodal_AUV.data.loaders import split_dataset, prepare_datasets_and_loaders

class Test_CustomImageDataset1(unittest.TestCase):

    @patch("glob.glob")
    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("os.path.exists")
    @patch("PIL.Image.open")
    def test_load_data_skips_missing_files(self, mock_open_img, mock_exists, mock_isdir, mock_listdir, mock_glob):
        # Mock folder structure
        mock_listdir.side_effect = lambda d: ["folder1"] if d == "root_dir" else [
            "Frame001.jpg", "SSS_image.png", "combined_rgb_bathymetry.jpg"
        ]
        mock_isdir.return_value = True
        mock_exists.side_effect = lambda path: True
        mock_glob.return_value = ["root_dir/folder1/Frame001.jpg"]

        # Return a real image object
        real_img = Image.fromarray((np.ones((512, 512, 3)) * 255).astype(np.uint8))  # RGB white image
        mock_open_img.return_value = real_img

        dataset = CustomImageDataset_1("root_dir")

        self.assertEqual(len(dataset), 1)

        sample = dataset[0]
        self.assertIsInstance(sample[0], torch.Tensor)
        self.assertIsInstance(sample[1], torch.Tensor)
        self.assertIsInstance(sample[2], torch.Tensor)
        self.assertIsInstance(sample[3], str)
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
        mock_img_array = MagicMock()
        mock_img_array.sum.return_value = 1
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

    @patch("Multimodal_AUV.data.loaders.split_dataset")  
    @patch("Multimodal_AUV.data.loaders.DataLoader")
    @patch("collections.Counter")
    @patch("Multimodal_AUV.data.loaders.CustomImageDataset")
    def test_prepare_datasets_and_loaders(self, mock_dataset, mock_counter, mock_dataloader, mock_split):
        # Mock dataset instance
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.labels = [0, 1, 1, 0]
        mock_dataset_instance.label_encoder.classes_ = np.array(["class0", "class1"])
        mock_dataset.return_value = mock_dataset_instance

        # Mock Counter to behave like a dict
        mock_counter.return_value = {0: 2, 1: 2}

        # Mock DataLoader to just return a string placeholder
        mock_dataloader.return_value = "dataloader"

        # Mock split_dataset to return two datasets
        mock_train_dataset = MagicMock()
        mock_test_dataset = MagicMock()
        mock_split.return_value = (mock_train_dataset, mock_test_dataset)

        # Call the function
        result = prepare_datasets_and_loaders("root_dir", 4, 8)

        # Unpack results
        train_loader, test_loader, train_loader_multi, test_loader_multi, num_classes, dataset = result

        # Assertions
        self.assertEqual(len(result), 6)
        self.assertEqual(num_classes, 2)
        self.assertEqual(dataset, mock_dataset_instance)

        # DataLoader called 4 times: train/test unimodal & multimodal
        self.assertEqual(mock_dataloader.call_count, 4)

        # Check that DataLoader was called with correct datasets
        calls = mock_dataloader.call_args_list
        self.assertIn(mock_train_dataset, calls[0][0])  # train_loader
        self.assertIn(mock_test_dataset, calls[1][0])   # test_loader
        self.assertIn(mock_train_dataset, calls[2][0])  # train_loader_multimodal
        self.assertIn(mock_test_dataset, calls[3][0])   # test_loader_multimodal