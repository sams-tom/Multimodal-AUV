import os
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Set, List

def save_model(model, csv_path, patch_type):
    """
    Saves the state dictionary of a PyTorch model to a designated models directory 
    relative to the given CSV file path.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model whose state dictionary will be saved.
    csv_path : str
        The file path to a CSV file; used to determine the base directory for saving the model.
    patch_type : str
        A string identifier used to name the saved model file, indicating the model type or patch type.

    Returns
    -------
    None
        Saves the model to disk and prints the save location.
    """
    try:
        #Try adding in pythons
        base_path = os.path.dirname(os.path.dirname(csv_path))
        models_dir = os.path.join(base_path, "models")

        #Make the directory if this doesnt exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            logging.info(f"Created models directory at: {models_dir}")
        else:
            logging.info(f"Models directory exists at: {models_dir}")

        #Save the model to path
        model_path = os.path.join(models_dir, f"bayesian_model_type{patch_type}.pth")
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved successfully to {model_path}")

    except Exception as e:
        logging.error(f"Error saving model: {e}", exc_info=True)

def load_and_fix_state_dict(model: nn.Module, model_path: str, device: torch.device) -> Tuple[bool, List[str]]:
    """
    Loads a model state dictionary from a file and adapts keys to match the given model's keys.
    It handles 'module.' prefixes from DataParallel.
    
    A single WARNING message is logged globally if any 'module.' prefix stripping occurs.
    Details of skipped layers are collected and returned for later consolidated reporting.
    
    Returns:
        Tuple[bool, List[str]]: 
            - bool: True if the model state_dict was successfully loaded, False otherwise.
            - List[str]: A list of strings, each describing a skipped layer and the reason.
    """
    skipped_layers_details: List[str] = [] # Local list to collect skipped layer info
    my_logger = logging.getLogger('my_silent_logger')
    my_logger.setLevel(logging.INFO) # Set the level for this logger

    # Add a NullHandler to it. This handler does nothing.
    my_logger.addHandler(logging.NullHandler())

    if not os.path.exists(model_path):
        logging.warning(f"Model checkpoint not found at: {model_path}. Skipping load.")
        return False, skipped_layers_details # Return False and an empty list on failure

    logging.info(f"Attempting to load state dict from: {model_path}")
    
    _module_prefix_stripped_in_this_specific_call = False 

    try:
        state_dict_from_checkpoint = torch.load(model_path, map_location=device)
        model_state_dict = model.state_dict()
        new_state_dict_for_load = {}
        
        for k_checkpoint, v_checkpoint in state_dict_from_checkpoint.items():
            k_model = k_checkpoint
            if k_checkpoint.startswith('module.'):
                k_model = k_checkpoint[len('module.'):]
                _module_prefix_stripped_in_this_specific_call = True 
            
            if k_model in model_state_dict:
                if v_checkpoint.shape == model_state_dict[k_model].shape:
                    new_state_dict_for_load[k_model] = v_checkpoint
                else:
                    # Collect details for shape mismatch instead of logging immediately
                    skipped_layers_details.append(
                        f"  - Key '{k_checkpoint}' (mapped to '{k_model}') "
                        f"due to shape mismatch: checkpoint has {v_checkpoint.shape}, "
                        f"model has {model_state_dict[k_model].shape}"
                    )
            else:
                # Collect details for key not found instead of logging immediately
                skipped_layers_details.append(
                    f"  - Key '{k_checkpoint}' (mapped to '{k_model}') "
                    f"as it is not found in the current model."
                )

        model.load_state_dict(new_state_dict_for_load, strict=False)
        
     
        logging.info("Model state_dict loaded successfully.") 
        logging.info(f"Skipped layers: {skipped_layers_details}")

        return True # Return True and the collected details

    except Exception as e:
        logging.error(f"An unexpected error occurred while loading state_dict for {model_path}: {e}", exc_info=True)
        return False # Return False and any collected details on error
