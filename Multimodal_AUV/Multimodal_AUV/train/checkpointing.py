import os
import logging
import torch


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


def load_and_fix_state_dict(model, model_path, device):
        """
        Loads a model state dictionary from a file and adapts keys to match the given model's keys,
        handling cases where keys might be prefixed (e.g., from DataParallel models) or shapes don't match.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model into which the state dictionary will be loaded.
        model_path : str
            Path to the saved state dictionary file.
        device : torch.device or str
            Device on which to map the loaded state dictionary.

        Returns
        -------
        None
            Loads the fixed state dictionary into the model.
    
        Logs info about loading progress, warnings for skipped keys, and exceptions on size mismatches.
        """
        logging.info(f"Loading state dict from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if k.startswith('module.model.'):
                name = k[13:]  # Remove 'module.model.'
            if name in model.state_dict():
                new_state_dict[name] = v
            else:
                logging.warning(f"Skipping unmatched key: {name}")

        try:
            model.load_state_dict(new_state_dict, strict=False)
        except RuntimeError as e:
            logging.exception("Size mismatch error while loading state_dict.")
            # Load only the keys that match in shape.
            final_state_dict = {}
            model_keys = model.state_dict().keys()
            for key, value in new_state_dict.items():
                if key in model_keys:
                    model_shape = model.state_dict()[key].shape
                    if value.shape == model_shape:
                        final_state_dict[key] = value
                    else:
                        logging.warning(f"Skipping unmatched key: {name}, due to shape mismatch")
            model.load_state_dict(final_state_dict, strict=False)
        logging.info("Model state_dict loaded successfully.")