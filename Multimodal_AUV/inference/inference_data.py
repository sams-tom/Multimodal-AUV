import os
import logging 
from Multimodal_AUV.data.datasets import CustomImageDataset_1
from torch.utils.data import DataLoader , ConcatDataset
def prepare_inference_datasets_and_loaders(dir_1: str, dir_2: str, batch_size: int) -> DataLoader:
    """
    Prepares inference DataLoader by combining datasets from two directories.
    """
    try:
        os.makedirs(dir_1, exist_ok=True)
        os.makedirs(dir_2, exist_ok=True)

        strangford_dataset = CustomImageDataset_1(dir_1)
        mulroy_dataset = CustomImageDataset_1(dir_2)
        multimodal_inference_dataset = ConcatDataset([strangford_dataset, mulroy_dataset])
        dataloader_whole_survey = DataLoader(multimodal_inference_dataset, batch_size=batch_size, shuffle=False)

        return dataloader_whole_survey

    except Exception as e:
        logging.error(f"Error preparing inference datasets and loaders: {e}", exc_info=True)
        raise
