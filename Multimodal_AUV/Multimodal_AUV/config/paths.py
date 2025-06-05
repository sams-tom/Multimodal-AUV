import logging
import torch
from typing import Tuple, List

def get_environment_paths() -> Tuple[str, str, str, str]:
    """
    Prompt user to select environment and return corresponding directory paths.

    Returns:
        Tuple[str, str, str, str]: Paths for root_dir, models_dir, strangford_dir, and mulroy_dir.
    """

    #Ask in terminal which environment people want
    environment = input("Do you want to run on 'server' or 'local'? ").strip().lower()

    #Depending on response set the paths
    if environment == 'server':
        root_dir = '/home/tommorgan/Documents/data/representative_sediment_sample/'
        models_dir = '/home/tommorgan/Documents/data/models/'
        strangford_dir = "/home/tommorgan/Documents/data/all_strangford_images_and_sonar/"
        mulroy_dir = "/home/tommorgan/Documents/data/all_mulroy_images_and_sonar/"
        logging.info("Environment set to server.")

    elif environment == 'local':
        root_dir = 'D:/Dataset_AUV_IRELAND/representative_sediment_sample/'
        models_dir = 'D:/Dataset_AUV_IRELAND/models/'
        strangford_dir = "D:/Dataset_AUV_IRELAND/all_strangford_images_and_sonar/"
        mulroy_dir = "D:/Dataset_AUV_IRELAND/all_mulroy_images_and_sonar/"
        logging.info("Environment set to local.")

    else:
        print("Invalid environment! Defaulting to local.")
        root_dir = 'D:/Dataset_AUV_IRELAND/representative_sediment_sample/'
        models_dir = 'D:/Dataset_AUV_IRELAND/models/'
        strangford_dir = "D:/Dataset_AUV_IRELAND/all_strangford_images_and_sonar/"
        mulroy_dir = "D:/Dataset_AUV_IRELAND/all_mulroy_images_and_sonar/"
        logging.warning("Invalid environment! Defaulting to local.")

    #Return paths
    return root_dir, models_dir, strangford_dir, mulroy_dir

def setup_environment_and_devices(print_devices: bool = False, force_cpu: bool = False) -> Tuple[str, str, str, str, List[torch.device]]:
    """
    Sets up directories and device configurations based on CUDA availability.

    Args:
        print_devices (bool): If True, prints the selected devices.
        force_cpu (bool): If True, forces CPU even if GPUs are available.

    Returns:
        Tuple containing:
            - root_dir (str): Root dataset directory path.
            - models_dir (str): Models directory path.
            - strangford_dir (str): Strangford dataset directory path.
            - mulroy_dir (str): Mulroy dataset directory path.
            - devices (List[torch.device]): List of available devices.
    """
    # Call the environmental paths function (user-defined, assumed available)
    root_dir, models_dir, strangford_dir, mulroy_dir = get_environment_paths()

    devices = []

    if not force_cpu and torch.cuda.is_available():
        num_cuda_devices = torch.cuda.device_count()
        devices = [torch.device(f"cuda:{i}") for i in range(num_cuda_devices)]
    else:
        devices = [torch.device("cpu")]

    if print_devices:
        for i, d in enumerate(devices):
            logging.info(f"Model {i} running on {d}")

    return root_dir, models_dir, strangford_dir, mulroy_dir, devices
