import logging
import torch
from typing import Tuple, List
import pynvml

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

def get_empty_gpus(threshold_mb=1000):
    """
    Returns a list of torch.device objects for GPUs with memory usage below a given threshold.

    Args:
        threshold_mb (int): Maximum allowed memory usage in MiB (Megabytes) for a GPU to be considered "empty".

    Returns:
        list: A list of torch.device objects (e.g., [torch.device('cuda:0'), torch.device('cuda:2')]).
              Returns an empty list if no GPUs are available or meet the criteria.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPUs to check.")
        return []

    empty_devices = []
    num_cuda_devices = torch.cuda.device_count()

    try:
        pynvml.nvmlInit()
        for i in range(num_cuda_devices):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # info.used is in bytes, convert to MiB
            gpu_memory_used_mib = info.used / (1024 * 1024)

            print(f"GPU {i}: Used Memory = {gpu_memory_used_mib:.2f} MiB / Total Memory = {info.total / (1024 * 1024):.2f} MiB")

            if gpu_memory_used_mib < threshold_mb:
                empty_devices.append(torch.device(f"cuda:{i}"))
                print(f"  -> GPU {i} considered empty and added.")
            else:
                print(f"  -> GPU {i} is busy (used {gpu_memory_used_mib:.2f} MiB) and skipped.")

    except pynvml.NVMLError as error:
        print(f"Error accessing NVIDIA GPUs with pynvml: {error}")
        print("Falling back to only checking torch.cuda.memory_allocated(), which might be less accurate for multi-process scenarios.")
        # Fallback if pynvml fails (e.g., driver not installed, permissions)
        for i in range(num_cuda_devices):
            # This only checks memory allocated by *this* process
            allocated_by_this_process_mb = torch.cuda.memory_allocated(i) / (1024 * 1024)
            print(f"GPU {i} (this process allocated): {allocated_by_this_process_mb:.2f} MiB")
            if allocated_by_this_process_mb < threshold_mb:
                empty_devices.append(torch.device(f"cuda:{i}"))
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass # Already shut down or never initialized

    return empty_devices

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
        devices = get_empty_gpus(threshold_mb=1000) # You can adjust this threshold
        if not devices:
            print("No empty GPUs found or CUDA not available. Falling back to CPU.")
            devices = [torch.device("cpu")]
        else:
            print(f"Selected empty GPUs: {[str(d) for d in devices]}")
    else:
        devices = [torch.device("cpu")]
        print("Force CPU is enabled. Using CPU.")

    return root_dir, models_dir, strangford_dir, mulroy_dir, devices

