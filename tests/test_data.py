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


import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import numpy as np
from PIL import Image
import torch
import io
import re # Needed for the dataset's regex matching
import logging # Needed for the dataset's logging calls
from torchvision import transforms # Needed for the dataset's transforms
import glob
# Suppress logging during test execution for cleaner output
logging.basicConfig(level=logging.CRITICAL)

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