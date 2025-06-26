import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin # NEW: Import the mixin

# Identity class (as you provided) - keep it, MultiModalModel might use it
class Identity(nn.Module):
    def forward(self, x):
        return x

# AdditiveAttention (as you provided) - keep it, MultiModalModel uses it
class AdditiveAttention(nn.Module):
    def __init__(self, d_model, hidden_dim=128):
        super(AdditiveAttention, self).__init__()
        self.query_projection = nn.Linear(d_model, hidden_dim)
        self.key_projection = nn.Linear(d_model, hidden_dim)
        self.value_projection = nn.Linear(d_model, hidden_dim)
        self.attention_mechanism = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query):
        keys = self.key_projection(query)
        values = self.value_projection(query)
        queries = self.query_projection(query)

        attention_scores = torch.tanh(queries + keys)
        attention_weights = F.softmax(self.attention_mechanism(attention_scores), dim=1)

        attended_values = values * attention_weights
        return attended_values

# ResNet50Custom (Modified to include PyTorchModelHubMixin and config, no is_feature_extractor)
class ResNet50Custom(nn.Module, PyTorchModelHubMixin): # Inherit from PyTorchModelHubMixin
    def __init__(self, input_channels: int, num_classes: int): # Added type hints for clarity
        super(ResNet50Custom, self).__init__()

        # Store config for PyTorchModelHubMixin to serialize to config.json
        # These are the parameters needed to re-instantiate the model.
        self.config = {
            "input_channels": input_channels,
            "num_classes": num_classes,
        }

        self.input_channels = input_channels # Keep this if you use it elsewhere

        # Load pretrained ResNet50 model
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to accept custom input channels
        self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def get_feature_size(self):
        return self.model.fc.in_features


import torch
import torch.nn as nn
import os
import json
import logging
from typing import Dict, Any

from huggingface_hub import HfApi, create_repo

# Import your custom models from your package
# Make sure the path is correct relative to where you run this script
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
YOUR_HF_USERNAME = "sams-tom" # <--- IMPORTANT: CHANGE THIS TO YOUR ACTUAL USERNAME!


# Define your constant BNN prior parameters.
# These MUST be the exact parameters used during the original dnn_to_bnn conversion when training.
const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.1,
    }

# The number of classes these models classify into.
# This should be the 'num_classes' that was passed to ResNet50Custom during their training.
# Assuming this is 'num_classes' from your main.py setup.
COMMON_NUM_CLASSES = 7 # <--- Adjust this to your actual num_classes

# Define the paths to your *existing* .pth files and their corresponding model configurations
# Use the paths from your `model_paths` dictionary in your `main.py`.
model_configs_for_upload = {
    "image_model": {
        "pth_path": "D:/2506 bayes results new labelling/models/bayesian_model_typeimage.pth", # <--- CHANGE THIS!
        "init_params": {"input_channels": 3, "num_classes": COMMON_NUM_CLASSES},
        "repo_id": f"{YOUR_HF_USERNAME}/multimodal-auv-image-bnn-classifier",
        "description": "Bayesian ResNet50 Classifier for AUV image data.",
        "subfolder_name": "image_model"

    },
    "bathy_model": {
        "pth_path": "D:/2506 bayes results new labelling/models/bayesian_model_typebathy.pth", # <--- CHANGE THIS!
        "init_params": {"input_channels": 3, "num_classes": COMMON_NUM_CLASSES},
        "repo_id": f"{YOUR_HF_USERNAME}/multimodal-auv-bathy-bnn-classifier",
        "description": "Bayesian ResNet50 Classifier for AUV bathymetry data.",
        "subfolder_name": "bathy_model"

    },
    "sss_model": {
        "pth_path": "D:/2506 bayes results new labelling/models/bayesian_model_typesss.pth", # <--- CHANGE THIS!
        "init_params": {"input_channels": 1, "num_classes": COMMON_NUM_CLASSES}, # SSS is 1 channel
        "repo_id": f"{YOUR_HF_USERNAME}/multimodal-auv-sss-bnn-classifier",
        "description": "Bayesian ResNet50 Classifier for AUV side-scan sonar data.",
        "subfolder_name": "sss_model"
    },
}
from huggingface_hub import login
# --- Hugging Face API setup ---
API_TOKEN = "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"
try:
    login(token=API_TOKEN)
    logging.info("Successfully logged into Hugging Face Hub.")
except Exception as e:
    logging.error(f"Failed to log in: {e}")
    exit()

api = HfApi( token=API_TOKEN)
device = torch.device("cpu") # Process on CPU for upload safety and portability
MAIN_HUB_REPO_ID = "sams-tom/multimodal-auv-bathy-bnn-classifier"

# --- Main Upload Loop ---
# --- Main Upload Loop ---
for model_key, model_info in model_configs_for_upload.items():
    logging.info(f"\n--- Processing {model_key} from {model_info['pth_path']} ---")

    pth_path = model_info["pth_path"]
    model_class = ResNet50Custom # Always ResNet50Custom for these models
    init_params = model_info["init_params"]
    description = model_info["description"]
    subfolder_name = model_info["subfolder_name"] # Get the unique subfolder name

    # Define the FULL repository ID for the upload (main repo ID)
    # The actual repo_id for the upload is always the main one
    repo_id_for_upload = MAIN_HUB_REPO_ID

    if not os.path.exists(pth_path):
        logging.error(f"Error: .pth file not found at {pth_path} for {model_key}. Skipping.")
        continue

    try:
        # 1. Instantiate the base model (non-Bayesian yet) using the parameters from init_params
        model_instance = model_class(**init_params).to(device)
        logging.info(f"Instantiated {model_key} with init_params: {init_params}")
        print(f"Is model_instance an instance of PyTorchModelHubMixin? {isinstance(model_instance, PyTorchModelHubMixin)}")

        # 2. Convert the instantiated model to a Bayesian Neural Network
        dnn_to_bnn(model_instance, const_bnn_prior_parameters)
        logging.info(f"Converted {model_key} to BNN architecture.")
        print(f"Is model_instance an instance of PyTorchModelHubMixin? {isinstance(model_instance, PyTorchModelHubMixin)}")

        # 3. Load the state_dict from your .pth file
        state_dict = torch.load(pth_path, map_location=device)

        # Handle 'module.' prefix from DataParallel if it exists in your .pth
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model_instance.load_state_dict(new_state_dict)
        model_instance.eval() # Set to evaluation mode for inference
        logging.info(f"Loaded .pth weights into {model_key} BNN instance.")

        # --- Now, prepare local directory for Hugging Face upload ---
        # Create a local directory for this specific model's files
        local_model_temp_dir = f"./hf_upload_temp/{model_key.replace('_', '-')}"
        os.makedirs(local_model_temp_dir, exist_ok=True)
        logging.info(f"Created temporary local directory for {model_key}: {local_model_temp_dir}")

        # 4. Save the model locally in Hugging Face format
        # This saves 'pytorch_model.bin' (BNN weights) and 'config.json' (from model_instance.config)
        model_instance.save_pretrained(local_model_temp_dir)
        logging.info(f"Model {model_key} saved locally to {local_model_temp_dir}")

        # 5. Save BNN prior parameters (essential for users to convert back to BNN on load)
        bnn_params_file = os.path.join(local_model_temp_dir, "bnn_params.json")
        with open(bnn_params_file, "w") as f:
            json.dump(const_bnn_prior_parameters, f, indent=4)
        logging.info(f"BNN prior parameters saved to {bnn_params_file}")

        # 6. Upload your custom model definition file(s) for `trust_remote_code=True`
        # This file will be uploaded directly to the root of the main repo, as it's shared.
        local_base_models_path = "Multimodal_AUV/models/base_models.py"
        base_models_path_in_repo = "base_models.py" # Will be at the root of MAIN_HUB_REPO_ID

        if os.path.exists(local_base_models_path):
            api.upload_file(
                path_or_fileobj=local_base_models_path,
                path_in_repo=base_models_path_in_repo, # Upload to the root of the main repo
                repo_id=repo_id_for_upload, # Use the main repo ID
                commit_message=f"Add base_models.py for BNN custom models",
            )
            logging.info(f"Uploaded {base_models_path_in_repo} to main repo.")
        else:
            logging.warning(f"Could not find {local_base_models_path}. Users will need to provide model definition locally.")


        # 7. Push the content of the *local model's temporary directory* to a *subfolder* in the main Hub repo
        # This is the key change for placing files in subfolders.
        api.upload_folder(
            folder_path=local_model_temp_dir,
            path_in_repo=subfolder_name, # <-- This tells it to put files into a subfolder
            repo_id=repo_id_for_upload, # Use the main repo ID
            commit_message=f"Upload {model_key} (Bayesian ResNet50 Classifier)",
            # delete_pointer_files=True # Uncomment if you have LFS pointer files that need to be deleted locally after upload
        )
        logging.info(f"Successfully pushed {model_key} to subfolder '{subfolder_name}' in https://huggingface.co/{repo_id_for_upload}")

        # Optional: Clean up the local temporary directory after successful upload
        import shutil
        shutil.rmtree(local_model_temp_dir)
        logging.info(f"Cleaned up local temporary directory: {local_model_temp_dir}")

    except Exception as e:
        logging.error(f"Failed to process and upload {model_key}: {e}", exc_info=True)

logging.info("\n--- All specified unimodal models processed. ---")