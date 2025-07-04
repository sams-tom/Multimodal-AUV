from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from Multimodal_AUV.data.datasets import CustomImageDataset
import os
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from collections import Counter
import logging 
from typing import Tuple
import sys
 # Split dataset into train and test subsets
def split_dataset(dataset, test_size=0.2):
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset

def prepare_datasets_and_loaders(root_dir: str, batch_size_unimodal: int, batch_size_multimodal: int) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, int, object]:
    """
    Prepare training and testing datasets and dataloaders.
    """
    try:
        if sys.platform.startswith('win'):
            num_workers = 0
           
        else:
            # For Linux/macOS, if num_workers is 0 (default), set it to a reasonable value
            # based on CPU count, unless explicitly passed as 0.
                cpu_count = os.cpu_count() or 1
                num_workers = max(1, cpu_count - 2) # Use a few less than total cores
                logging.info(f"Using {num_workers} num_workers for DataLoader on non-Windows OS.")
        # Define the dataset
        dataset = CustomImageDataset(root_dir)

        # Count all labels
        label_counts = Counter(dataset.labels)

        # Print the class distribution
        for label_id, count in label_counts.items():
            class_name = dataset.label_encoder.inverse_transform([label_id])[0]
            logging.info(f"{class_name} (ID {label_id}): {count} samples")

        # Split the dataset into train and test
        train_dataset, test_dataset = split_dataset(dataset)

        # Define dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size_unimodal, shuffle=True, pin_memory=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_unimodal, shuffle=False, pin_memory=True,num_workers=num_workers)
        train_loader_multimodal = DataLoader(train_dataset, batch_size=batch_size_multimodal, shuffle=True, pin_memory=True, num_workers=num_workers)
        test_loader_multimodal = DataLoader(test_dataset, batch_size=batch_size_multimodal, shuffle=False, pin_memory=True, num_workers=num_workers)

        num_classes = len(dataset.label_encoder.classes_)
        logging.info(f"Number of classes: {num_classes}")

        return train_loader, test_loader, train_loader_multimodal, test_loader_multimodal, num_classes, dataset

    except Exception as e:
        logging.error(f"Error preparing datasets and loaders: {e}", exc_info=True)
        raise

def encode_labels(dataset):
    labels = [dataset[i][2] for i in range(len(dataset))]
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder