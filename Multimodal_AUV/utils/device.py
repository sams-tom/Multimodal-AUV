import logging
from typing import Dict, List, Optional
import torch
import torch.nn as nn

def move_model_to_device(
    model: nn.Module,
    device: torch.device,
    device_ids: Optional[List[int]] = None
) -> nn.Module:
    """
    Moves the given model to the specified device and wraps with DataParallel if multiple GPUs are specified.
    """
    try:
        model = model.to(device)

        if device_ids and len(device_ids) > 1:
            logging.info(f"Using multiple GPUs: {device_ids}")
            model = nn.DataParallel(model, device_ids=device_ids)
        else:
            logging.info(f"Using single device: {device}")

        return model

    except Exception as e:
        logging.error(f"Error moving model to device: {e}", exc_info=True)
        raise


def move_models_to_device(
    models_dict: Dict[str, Optional[nn.Module]],
    devices: List[torch.device],
    use_multigpu_for_multimodal: bool = True
) -> Dict[str, nn.Module]:
    """
    Moves unimodal models to the first device, and multimodal model to all devices (if enabled).
    """
    try:
        primary_device = devices[0]
        device_ids = list(range(len(devices))) if use_multigpu_for_multimodal and len(devices) > 1 else None

        logging.info(f"Moving models to device(s): {devices}")
        for name, model in models_dict.items():
            if model is None:
                continue
            if "multimodal_model" in name:
                models_dict[name] = move_model_to_device(model, primary_device, device_ids)
            else:
                models_dict[name] = move_model_to_device(model, primary_device)

        return models_dict

    except Exception as e:
        logging.error(f"Error moving models to devices: {e}", exc_info=True)
        raise


def check_model_devices(model, expected_device):
        """
        Check whether all parameters of the model are located on the expected device.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to check.
        expected_device : torch.device or str
            The device (e.g., 'cpu' or 'cuda') on which the model parameters are expected to reside.

        Returns
        -------
        bool
            True if all model parameters are on the expected device; False otherwise.
    
        Logs a warning for each parameter found on an unexpected device.
        Logs an info message if all parameters are on the expected device.
        """
        for name, param in model.named_parameters():
            if param.device != expected_device:
                logging.warning(f"Param {name} is on {param.device}, expected {expected_device}")
                return False
        logging.info("All model parameters are on the expected device.")
        return True
