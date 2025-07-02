# run_installed_package_tests.py

import logging
import os
import sys
import torch
from typing import Dict, Any, List
from Multimodal_AUV.functions.functions import (
    run_auv_inference,
    run_auv_training,
    run_auv_preprocessing,
    run_AUV_training_from_scratch
)
# Configure basic logging for this test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Main execution block for testing ---
if __name__ == "__main__":
    logger.info("Starting test of multimodal-auv functions from the installed package.")

    # --- Configuration for Inference Test ---
    # !!! IMPORTANT: Replace these with actual paths to your inference data !!!
    inference_data_dir = "/home/tommorgan/Documents/data/all_mulroy_images_and_sonar" # e.g., folder with images/bathy
    inference_output_csv = "/home/tommorgan/Documents/data/test/inference_results.csv"

    print("\n" + "="*50)
    print("       Testing `run_auv_inference` function")
    print("="*50)
    if not os.path.exists(inference_data_dir):
        logger.warning(f"Inference data directory '{inference_data_dir}' does not exist. `run_auv_inference` may fail.")
        logger.warning("Please update `inference_data_dir` in `run_installed_package_tests.py`.")
    try:
        run_auv_inference(
            data_directory=inference_data_dir,
            batch_size=4,
            output_csv=inference_output_csv,
            num_mc_samples=5,
            num_classes=7 # Adjust if your model has a different number of classes
        )
        logger.info("`run_auv_inference` test completed successfully (or attempted). Check `inference_results.csv`.")
    except Exception as e:
        logger.error(f"Error running `run_auv_inference`: {e}", exc_info=True)


    # --- Configuration for Preprocessing Test ---
    # !!! IMPORTANT: Replace these with actual paths to your raw data and ExifTool !!!
    raw_optical_images_folder_path = "/home/tommorgan/Documents/data/Newfolder"
    geotiff_folder_path = "/home/tommorgan/Documents/data/Newfolder/sonar"
    preprocessing_output_folder = "/home/tommorgan/Documents/data/test/processed_auv_data_output"
    # For Windows: "C:/Program Files/ExifTool/exiftool.exe"
    # For Linux: "/usr/bin/exiftool" or wherever you installed it
    exiftool_executable = "/usr/bin/exiftool"

    print("\n" + "="*50)
    print("       Testing `run_auv_preprocessing` function")
    print("="*50)
    if not os.path.exists(raw_optical_images_folder_path):
        logger.warning(f"Raw optical images folder '{raw_optical_images_folder_path}' does not exist. `run_auv_preprocessing` may fail.")
        logger.warning("Please update `raw_optical_images_folder_path` in `run_installed_package_tests.py`.")
    if not os.path.exists(geotiff_folder_path):
        logger.warning(f"GeoTIFF folder '{geotiff_folder_path}' does not exist. `run_auv_preprocessing` may fail.")
        logger.warning("Please update `geotiff_folder_path` in `run_installed_package_tests.py`.")
    if not os.path.exists(exiftool_executable) and not sys.platform.startswith('win'): # Exiftool on Windows can be just the dir
        logger.warning(f"ExifTool executable not found at '{exiftool_executable}'. `run_auv_preprocessing` may fail.")
        logger.warning("Please update `exiftool_executable` in `run_installed_package_tests.py`.")

    try:
        run_auv_preprocessing(
            raw_optical_images_folder=raw_optical_images_folder_path,
            geotiff_folder=geotiff_folder_path,
            output_folder=preprocessing_output_folder,
            exiftool_path=exiftool_executable,
            window_size_meters=20.0,
            image_enhancement_method="AverageSubtraction",
            skip_bathy_combine=False
        )
        logger.info("`run_auv_preprocessing` test completed successfully (or attempted). Check output in `preprocessing_output_folder`.")
    except Exception as e:
        logger.error(f"Error running `run_auv_preprocessing`: {e}", exc_info=True)


    # --- Configuration for Training Test (from scratch) ---
    # !!! IMPORTANT: These are minimal dummy values. Real training requires careful configuration !!!
    # !!! YOU MUST REPLACE THESE WITH YOUR ACTUAL TRAINING PARAMETERS AND DATA PATHS !!!
    training_root_dir = "/home/tommorgan/Documents/data/representative_sediment_sample"
    num_classes_for_training = 7 # Example: Adjust to your actual number of classes

    # Define devices: try to use CUDA if available, otherwise CPU
    training_devices = [torch.device("cuda:0")] if torch.cuda.is_available() else [torch.device("cpu")]
    logger.info(f"Training will attempt to use devices: {[str(d) for d in training_devices]}")

    dummy_const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "kl_reweighting_factor": 1.0
    }
    dummy_optimizer_params = {
        "multimodal_model": {"lr": 0.001, "weight_decay": 1e-5}
        # Add other models if your `define_optimizers_and_schedulers` expects them
    }
    dummy_scheduler_params = {
        "multimodal_model": {"T_0": 10, "eta_min": 1e-6} # CosineAnnealingWarmRestarts example
        # Add other models if your `define_optimizers_and_schedulers` expects them
    }
    dummy_training_params = {
        "batch_size_multimodal": 4,
        "batch_size_unimodal": 1, # Often 1 for unimodal if not directly used in this loop
        "num_epochs_multimodal": 2, # Keep low for a quick test
        "bathy_patch_base": "patch_multichannel_10m", # Example from your code
        "sss_patch_base": "patch_multichannel_10m",   # Example from your code
        "num_mc": 5 # Monte Carlo samples
    }
    # !!! Also need to ensure csvs are in os.path.join(training_root_dir, "csvs")

    print("\n" + "="*50)
    print("       Testing `run_AUV_training_from_scratch` function")
    print("       (Requires actual data and careful config!)")
    print("="*50)
    if not os.path.exists(training_root_dir):
        logger.warning(f"Training data root directory '{training_root_dir}' does not exist. `run_AUV_training_from_scratch` will fail.")
        logger.warning("Please update `training_root_dir` in `run_installed_package_tests.py`.")
    else:
        logger.info(f"Attempting `run_AUV_training_from_scratch` with root_dir: {training_root_dir}")
        logger.info("NOTE: This training function needs proper data, and the dummy parameters might not be sufficient.")
        try:
            success = run_AUV_training_from_scratch(
                const_bnn_prior_parameters=dummy_const_bnn_prior_parameters,
                optimizer_params=dummy_optimizer_params,
                scheduler_params=dummy_scheduler_params,
                training_params=dummy_training_params,
                root_dir=training_root_dir,
                devices=training_devices,
                num_classes=num_classes_for_training
            )
            if success:
                logger.info("`run_AUV_training_from_scratch` test completed (or attempted) successfully.")
            else:
                logger.warning("`run_AUV_training_from_scratch` test did not complete successfully.")
        except Exception as e:
            logger.error(f"Error running `run_AUV_training_from_scratch`: {e}", exc_info=True)


    # Example of how you would call run_auv_training (for retraining a model)
    # This would require a path to an existing model's weights to start from.
    # print("\n" + "="*50)
    # print("       Testing `run_auv_training` function (retraining)")
    # print("       (Requires an existing model and actual data!)")
    # print("="*50)
    # existing_model_weights = "./path/to/your/existing_model_weights.bin"
    # if not os.path.exists(existing_model_weights):
    #     logger.warning(f"Model weights not found at '{existing_model_weights}'. Skipping `run_auv_training` test.")
    #     logger.warning("Please update `existing_model_weights` if you wish to test retraining.")
    # else:
    #     try:
    #         run_auv_training(
    #             multimodal_model_weights_path=existing_model_weights,
    #             optimizer_params=dummy_optimizer_params,
    #             scheduler_params=dummy_scheduler_params,
    #             training_params=dummy_training_params,
    #             root_dir=training_root_dir,
    #             devices=training_devices,
    #             const_bnn_prior_parameters=dummy_const_bnn_prior_parameters,
    #             num_classes=num_classes_for_training
    #         )
    #         logger.info("`run_auv_training` test completed (or attempted) successfully.")
    #     except Exception as e:
    #         logger.error(f"Error running `run_auv_training`: {e}", exc_info=True)


    logger.info("\nAll specified multimodal-auv function tests have been initiated.")
    logger.info("Please review the logs and any generated files for actual test outcomes.")