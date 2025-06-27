import torch
import torch.nn as nn
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List # Keep imports for load_and_fix_state_dict if you re-add it

from huggingface_hub import HfApi, create_repo, login
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import PyTorchModelHubMixin # Import the mixin

# NEW: Import your custom models from the new file
# Adjust this import based on where you save model_definitions.py
# If it's in the same directory as this script:
from model_definitions import Identity, AdditiveAttention, ResNet50Custom, MultiModalModel
# If it's in Multimodal_AUV/models/:
# from Multimodal_AUV.models.model_definitions import Identity, AdditiveAttention, ResNet50Custom, MultiModalModel

# Assuming dnn_to_bnn is correctly implemented in your bayesian_torch.models.dnn_to_bnn
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
YOUR_HF_USERNAME = "sams-tom" # <--- IMPORTANT: CHANGE THIS TO YOUR ACTUAL USERNAME!

# Define your constant BNN prior parameters.
const_bnn_prior_parameters = {
    "prior_mu": 0.0,
    "prior_sigma": 1.0,
    "posterior_mu_init": 0.0,
    "posterior_rho_init": -3.0,
    "type": "Reparameterization",
    "moped_enable": True,
    "moped_delta": 0.1,
}

COMMON_NUM_CLASSES = 7 # <--- Adjust this to your actual num_classes

# Configure a specific logger for the load_and_fix_state_dict function to control its output
_load_logger = logging.getLogger('load_and_fix_state_dict_logger')
_load_logger.setLevel(logging.INFO)
if not _load_logger.handlers: # Prevent adding handlers multiple times
    _load_logger.addHandler(logging.NullHandler()) # Default to NullHandler if no other handlers are configured

def load_and_fix_state_dict(model: nn.Module, model_path: str, device: torch.device, model_key_name: str) -> Tuple[bool, List[str]]:
    """
    Loads a model state dictionary from a file and adapts keys to match the given model's keys.
    It handles 'module.' prefixes from DataParallel and attempts to fix common nested prefixes
    like 'image_model_feat.conv1' -> 'image_model_feat.model.conv1' for Bayesian ResNet sub-modules
    within the MultiModalModel.

    Args:
        model (nn.Module): The target PyTorch model instance (already initialized and possibly BNN-converted).
        model_path (str): Path to the saved model file.
        device (torch.device): The device to load the model onto (e.g., 'cpu', 'cuda').
        model_key_name (str): The key name for the model (e.g., "multimodal_model", "image_model")
                              to apply specific remapping logic.

    Returns:
        Tuple[bool, List[str]]:
            - bool: True if the model state_dict was successfully loaded, False otherwise.
            - List[str]: A list of strings, each describing a skipped/remapped layer and the reason.
    """
    skipped_layers_details: List[str] = []

    if not os.path.exists(model_path):
        _load_logger.warning(f"Model checkpoint not found at: {model_path}. Skipping load.")
        return False, ["Model checkpoint not found."]

    _load_logger.info(f"Attempting to load state dict from: {model_path}")

    try:
        state_dict_from_checkpoint = torch.load(model_path, map_location=device)
        model_state_dict = model.state_dict() # Get current model's state_dict keys

        new_state_dict_for_load: Dict[str, torch.Tensor] = {}

        # First pass: Build the new_state_dict with remapped keys
        for k_checkpoint, v_checkpoint in state_dict_from_checkpoint.items():
            current_k = k_checkpoint

            # 1. Handle 'module.' prefix (from DataParallel)
            if current_k.startswith('module.'):
                current_k = current_k[len('module.'):]

            # 2. Handle nested 'model.' prefix for feature extractors within MultiModalModel
            remapped_k = current_k
            if model_key_name == "multimodal_model": # Apply this specific remapping only for the multimodal model
                if current_k.startswith('image_model_feat.') and 'model.' not in current_k.split('image_model_feat.')[1]:
                    remapped_k = current_k.replace('image_model_feat.', 'image_model_feat.model.', 1)
                    _load_logger.debug(f"Remapping '{current_k}' to '{remapped_k}' (image_model_feat)")
                elif current_k.startswith('bathy_model_feat.') and 'model.' not in current_k.split('bathy_model_feat.')[1]:
                    remapped_k = current_k.replace('bathy_model_feat.', 'bathy_model_feat.model.', 1)
                    _load_logger.debug(f"Remapping '{current_k}' to '{remapped_k}' (bathy_model_feat)")
                elif current_k.startswith('sss_model_feat.') and 'model.' not in current_k.split('sss_model_feat.')[1]:
                    remapped_k = current_k.replace('sss_model_feat.', 'sss_model_feat.model.', 1)
                    _load_logger.debug(f"Remapping '{current_k}' to '{remapped_k}' (sss_model_feat)")

            # Now try to match remapped_k (or original k if no remapping) to the model_state_dict
            if remapped_k in model_state_dict:
                if v_checkpoint.shape == model_state_dict[remapped_k].shape:
                    new_state_dict_for_load[remapped_k] = v_checkpoint
                else:
                    skipped_layers_details.append(
                        f"Skipped key '{k_checkpoint}' (mapped to '{remapped_k}') due to shape mismatch: "
                        f"checkpoint has {v_checkpoint.shape}, model expects {model_state_dict[remapped_k].shape}"
                    )
            else:
                    # Check if the original key without remapping would have matched
                    if current_k in model_state_dict:
                        skipped_layers_details.append(
                            f"Skipped key '{k_checkpoint}' due to remapping to '{remapped_k}', but original key '{current_k}' exists. "
                            f"Consider if remapping is needed for this layer."
                        )
                    else:
                        skipped_layers_details.append(
                            f"Skipped key '{k_checkpoint}' (remapped to '{remapped_k}' or original if no remapping) "
                            f"as target key is not found in the current model."
                        )

        # Attempt to load the state_dict
        model.load_state_dict(new_state_dict_for_load, strict=False)

        _load_logger.info("Model state_dict loaded successfully (possibly with skipped layers).")
        if skipped_layers_details:
            _load_logger.warning("Details of skipped/remapped layers during loading:")
            for detail in skipped_layers_details:
                _load_logger.warning(detail)
        else:
            _load_logger.info("No layers were skipped or remapped during loading.")

        return True, skipped_layers_details

    except Exception as e:
        _load_logger.error(f"An unexpected error occurred while loading state_dict for {model_path}: {e}", exc_info=True)
        return False, skipped_layers_details # Return False and any collected details on error

# --- Hugging Face API setup ---
API_TOKEN = "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" # Replace with your actual token
try:
    login(token=API_TOKEN)
    logging.info("Successfully logged into Hugging Face Hub.")
except Exception as e:
    logging.error(f"Failed to log in: {e}")
    exit() # Exit if login fails

api = HfApi(token=API_TOKEN)
device = torch.device("cpu") # Process on CPU for upload safety and portability
MAIN_HUB_REPO_ID = f"{YOUR_HF_USERNAME}/multimodal-auv-bathy-bnn-classifier"



# Define the paths to your *existing* .pth files and their corresponding model configurations
model_configs_for_upload = {
     "image_model": {
         "class": ResNet50Custom,
         "pth_path": "D:/2506 bayes results new labelling/models/bayesian_model_typeimage.pth",
         "init_params": {"input_channels": 3, "num_classes": COMMON_NUM_CLASSES},
         "description": "Bayesian ResNet50 Classifier for AUV image data.",
         "subfolder_name": "unimodal-image-bnn"
     },
     "bathy_model": {
         "class": ResNet50Custom,
         "pth_path": "D:/2506 bayes results new labelling/models/bayesian_model_typebathy.pth",
         "init_params": {"input_channels": 3, "num_classes": COMMON_NUM_CLASSES},
         "description": "Bayesian ResNet50 Classifier for AUV bathymetry data.",
         "subfolder_name": "unimodal-bathy-bnn"
     },
     "sss_model": {
         "class": ResNet50Custom,
         "pth_path": "D:/2506 bayes results new labelling/models/bayesian_model_typesss.pth",
         "init_params": {"input_channels": 1, "num_classes": COMMON_NUM_CLASSES}, # SSS is 1 channel
         "description": "Bayesian ResNet50 Classifier for AUV side-scan sonar data.",
         "subfolder_name": "unimodal-sss-bnn"
     },
    "multimodal_model": { # NEW ENTRY for your MultiModalModel
        "class": MultiModalModel,
        "pth_path": "D:/2506 bayes results new labelling/models/bayesian_model_typemultimodal_bathy_patch30_sss_patch30.pth", # Your combined model path
        "init_params": {
            "image_input_channels": 3,
            "bathy_input_channels": 3,
            "sss_input_channels": 1,
            "num_classes": COMMON_NUM_CLASSES,
            "attention_type": "scaled_dot_product"
        },
        "description": "Bayesian MultiModal Classifier for AUV image, bathymetry, and side-scan sonar data.",
        "subfolder_name": "multimodal-bnn"
    }
}

# --- Main Upload Loop ---
for model_key, model_info in model_configs_for_upload.items():
    logging.info(f"\n--- Processing {model_key} from {model_info['pth_path']} ---")

    pth_path = model_info["pth_path"]
    model_class = model_info["class"]
    init_params = model_info["init_params"]
    description = model_info["description"]
    subfolder_name = model_info["subfolder_name"]

    repo_id_for_upload = MAIN_HUB_REPO_ID

    if not os.path.exists(pth_path):
        logging.error(f"Error: .pth file not found at {pth_path} for {model_key}. Skipping.")
        continue

    try:
        model_instance = model_class(**init_params).to(device)
        logging.info(f"Instantiated {model_key} with init_params: {init_params}")
        print(f"Is {model_key} an instance of PyTorchModelHubMixin? {isinstance(model_instance, PyTorchModelHubMixin)}")

        dnn_to_bnn(model_instance, const_bnn_prior_parameters)
        logging.info(f"Converted {model_key} to BNN architecture.")

        # Use the robust load_and_fix_state_dict function
        success, skipped_details = load_and_fix_state_dict(model_instance, pth_path, device, model_key)

        if not success:
            logging.error(f"Failed to load state_dict for {model_key}. Skipping upload.")
            if skipped_details:
                logging.error("Skipped/Error details:")
                for detail in skipped_details:
                    logging.error(f"  - {detail}")
            continue

        model_instance.eval()

        local_model_temp_dir = f"./hf_upload_temp/{subfolder_name}"
        os.makedirs(local_model_temp_dir, exist_ok=True)
        logging.info(f"Created temporary local directory for {model_key}: {local_model_temp_dir}")

        model_instance.save_pretrained(local_model_temp_dir)
        logging.info(f"Model {model_key} saved locally to {local_model_temp_dir}")

        bnn_params_file = os.path.join(local_model_temp_dir, "bnn_params.json")
        with open(bnn_params_file, "w") as f:
            json.dump(const_bnn_prior_parameters, f, indent=4)
        logging.info(f"BNN prior parameters saved to {bnn_params_file}")

        # >>>>>> IMPORTANT: ENSURE THIS PATH IS CORRECT FOR YOUR LOCAL `model_definitions.py` <<<<<<
        # This path should point to the model_definitions.py file you created in Step 1.
        # If model_definitions.py is in the same directory as this script:
        local_model_definition_file = "model_definitions.py"
        # If it's in a subdirectory, e.g., 'my_project/models/model_definitions.py':
        # local_model_definition_file = "my_project/models/model_definitions.py"


        path_in_repo_for_definitions = "model_definitions.py" # This will be at the root of the HF repo

        if os.path.exists(local_model_definition_file):
            api.upload_file(
                path_or_fileobj=local_model_definition_file,
                path_in_repo=path_in_repo_for_definitions,
                repo_id=repo_id_for_upload,
                commit_message=f"Add custom model definitions ({path_in_repo_for_definitions})",
            )
            logging.info(f"Uploaded {path_in_repo_for_definitions} to main repo.")
        else:
            # This warning means the file was not found locally.
            # To avoid this warning, make sure local_model_definition_file points to the correct file.
            logging.error(f"ERROR: Could not find {local_model_definition_file}. Model definitions will NOT be available via `trust_remote_code=True` without local code.")
            logging.error("Please verify the path to your model_definitions.py file.")


        api.upload_folder(
            folder_path=local_model_temp_dir,
            path_in_repo=subfolder_name,
            repo_id=repo_id_for_upload,
            commit_message=f"Upload {model_key} ({description})",
        )
        logging.info(f"Successfully pushed {model_key} to subfolder '{subfolder_name}' in https://huggingface.co/{repo_id_for_upload}")

        import shutil
        shutil.rmtree(local_model_temp_dir)
        logging.info(f"Cleaned up local temporary directory: {local_model_temp_dir}")

    except Exception as e:
        logging.error(f"Failed to process and upload {model_key}: {e}", exc_info=True)

logging.info("\n--- All specified models processed. ---")