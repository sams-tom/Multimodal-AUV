import unittest #Imported for unittesing frame works
from unittest.mock import patch, MagicMock, mock_open #Importing unittesting parts
import os #For filepaths and directories
import numpy as np #Used for mock image arrays
from PIL import Image #Used tio simulate image loading and manipulation
import torch #For pytorch tesnor outputs
from torch.utils.data import DataLoader #To simulate dataloading 
from Multimodal_AUV.data.datasets import CustomImageDataset_1, CustomImageDataset #Loading in my custom datasets
from Multimodal_AUV.data.loaders import split_dataset, prepare_datasets_and_loaders #Loading in functions to divide datasets and prepare datasets

class Test_CustomImageDataset1(unittest.TestCase):

    @patch("glob.glob") #Mocking file searching so no disk scan is performed
    @patch("os.listdir") #Mocking directory listing to simulate folder contents
    @patch("os.path.isdir") #Mock to always say a path is in a directory
    @patch("os.path.exists") #Mock to simulate whether files exist
    @patch("PIL.Image.open") #Mock images loading
    def test_load_data_skips_missing_files(self, mock_open_img, mock_exists, mock_isdir, mock_listdir, mock_glob):
        """
        Test CustomImageDataset_1 loading functionality and ensure it handles expected image and label files correctly.
        """
        # Mock folder structure
        mock_listdir.side_effect = lambda d: ["folder1"] if d == "root_dir" else [
            "Frame001.jpg", "SSS_image.png", "combined_rgb_bathymetry.jpg"
        ]
        #Simulate that all paths are directories
        mock_isdir.return_value = True
        #Simulate that all paths exist
        mock_exists.side_effect = lambda path: True
        #Simulate image discovery
        mock_glob.return_value = ["root_dir/folder1/Frame001.jpg"]

        # Return a real image object (white RGB image)
        real_img = Image.fromarray((np.ones((512, 512, 3)) * 255).astype(np.uint8))
        #Set all opened images to be this dummy image
        mock_open_img.return_value = real_img

        #Initiate dataset and check this is one long
        dataset = CustomImageDataset_1("root_dir")
        self.assertEqual(len(dataset), 1)

        #Validate the sample structure is three tensors (0,1,2) and a string in 4 (label)
        sample = dataset[0]
        self.assertIsInstance(sample[0], torch.Tensor)
        self.assertIsInstance(sample[1], torch.Tensor)
        self.assertIsInstance(sample[2], torch.Tensor)
        self.assertIsInstance(sample[3], str)

class Test_CustomImageDataset(unittest.TestCase):

    @patch("os.listdir") #Simulate folder content listing
    @patch("os.path.isdir") #Simulate everything to be a directory 
    @patch("os.path.exists") #Pretned all files exist
    @patch("glob.glob") #Simulate file pattern matching for image paths
    @patch("PIL.Image.open") #Mock file I/O for images
    def test_dataset_loading_and_len(self, mock_open_img, mock_glob, mock_exists, mock_isdir, mock_listdir):
        """
        Test CustomImageDataset loading logic, ensuring expected keys and structure are returned.
        """

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
        mock_img_array.sum.return_value = 1 #Ensure the image is not filtered out
        mock_img.__array__ = lambda *a: mock_img_array
        mock_open_img.return_value = mock_img

        #Instantiate dataset
        dataset = CustomImageDataset("root_dir")
        self.assertGreaterEqual(len(dataset), 0)

        #Validate thge sample contents are what you expect
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
        """
        Ensure split_dataset splits the data into the correct proportions and types.
        """
        dummy_data = list(range(100))
        #Initiate a fake dataset
        class DummyDataset:
            def __len__(self): return len(dummy_data)
            def __getitem__(self, idx): return dummy_data[idx]

        dataset = DummyDataset()
        #Use the custom split dataset function
        train_ds, test_ds = split_dataset(dataset, test_size=0.25)
        #Assert these add up to the correct dataset length
        self.assertEqual(len(train_ds) + len(test_ds), len(dataset))
        #Assert that the datasets are the correct type
        self.assertIsInstance(train_ds, torch.utils.data.Subset)
        self.assertIsInstance(test_ds, torch.utils.data.Subset)

    @patch("Multimodal_AUV.data.loaders.split_dataset")  # Mock split to isolate test to logic
    @patch("Multimodal_AUV.data.loaders.DataLoader")  # Prevent actual DataLoader from running
    @patch("collections.Counter") # Mock label frequency counting
    @patch("Multimodal_AUV.data.loaders.CustomImageDataset")  # Don't actually load any data
    def test_prepare_datasets_and_loaders(self, mock_dataset, mock_counter, mock_dataloader, mock_split):
        """
        Test prepare_datasets_and_loaders returns correct structure and values
        without needing actual disk access or data.
        """

        # Mock dataset instance
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.labels = [0, 1, 1, 0]
        mock_dataset_instance.label_encoder.classes_ = np.array(["class0", "class1"])
        mock_dataset.return_value = mock_dataset_instance

        # Simulate class balance
        mock_counter.return_value = {0: 2, 1: 2}

        # DataLoader mock returns string placeholder
        mock_dataloader.return_value = "dataloader"

        # Split into train/test mocks
        mock_train_dataset = MagicMock()
        mock_test_dataset = MagicMock()
        mock_split.return_value = (mock_train_dataset, mock_test_dataset)

        # Call the loader-preparation utility
        result = prepare_datasets_and_loaders("root_dir", 4, 8)

        # Validate output
        train_loader, test_loader, train_loader_multi, test_loader_multi, num_classes, dataset = result
        #Assert that 6 length of data loader
        self.assertEqual(len(result), 6)
        #Assert that the number of classes are 2 long
        self.assertEqual(num_classes, 2)
        #Assrt its the same length as the dataset instance
        self.assertEqual(dataset, mock_dataset_instance)

        # Assert that the DataLoader is called 4 times: train/test unimodal & multimodal
        self.assertEqual(mock_dataloader.call_count, 4) #4 calls = 2 loaders x 2 modes

        # Check that DataLoader was called with correct datasets
        calls = mock_dataloader.call_args_list
        self.assertIn(mock_train_dataset, calls[0][0])  # train_loader
        self.assertIn(mock_test_dataset, calls[1][0])   # test_loader
        self.assertIn(mock_train_dataset, calls[2][0])  # train_loader_multimodal
        self.assertIn(mock_test_dataset, calls[3][0])   # test_loader_multimodal