from huggingface_hub import hf_hub_download # NEW: Import for Hugging Face Hub download
import logging
import torch
import os
from typing import Dict, Any, List
import sys
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.nn as nn
from bayesian_torch.layers import LinearReparameterization
from Multimodal_AUV.Examples.Example_Inference_model import prepare_inference_dataloader, load_and_prepare_multimodal_model
from Multimodal_AUV.inference.predictors import multimodal_predict_and_save 
from Multimodal_AUV.data.loaders import prepare_datasets_and_loaders
from Multimodal_AUV.models.model_utils import define_models
from Multimodal_AUV.Examples.Example_Retraining_model import load_and_prepare_multimodal_model_custom
from Multimodal_AUV.train.loop_utils import train_and_evaluate_unimodal_model, train_and_evaluate_multimodal_model, define_optimizers_and_schedulers
from Multimodal_AUV.utils.device import move_models_to_device
from Multimodal_AUV.Examples.Example_data_preparation import preprocess_optical_images, process_and_save_data
from Multimodal_AUV.data_preparation.utilities import is_geotiff, filter_csv_by_image_names, update_csv_path
from Multimodal_AUV.data_preparation.geospatial import get_pixel_resolution, extract_grid_patch
from Multimodal_AUV.data_preparation.image_processing import process_frame_channels_in_subfolders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) 

def run_auv_inference(
    data_directory: str,
    batch_size: int = 4, # Default values here make it easier to call
    output_csv: str = "./inference_results.csv",
    num_mc_samples: int = 5,
    num_classes: int = 7
):
    """
    Main function to run the multimodal AUV inference process.
    This is the primary function you'll call from other scripts.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
       
        multimodal_model_hf_repo_id = "sams-tom/multimodal-auv-bathy-bnn-classifier"
    
        # --- CRITICAL FIX: Define the subfolder with forward slashes directly,
        #                  or ensure any backslashes are replaced immediately. ---
        # Option 1 (Most direct if you know it's always "multimodal-bnn"):
        multimodal_model_hf_subfolder = "multimodal-bnn" 
    
        # Option 2 (More robust if 'multimodal_model_hf_subfolder' could come from a path operation elsewhere):
        multimodal_model_hf_subfolder = multimodal_model_hf_subfolder.replace('\\', '/')

        # Now, construct the filename using the guaranteed-forward-slash subfolder
        model_weights_filename = f"{multimodal_model_hf_subfolder}/pytorch_model.bin"

        logging.info(f"DEBUG: repo_id being used: '{multimodal_model_hf_repo_id}'")
        logging.info(f"DEBUG: filename being used: '{model_weights_filename}'") # This debug print will confirm the path
        logging.info(f"Attempting to download multimodal model weights from '{multimodal_model_hf_repo_id}' with filename '{model_weights_filename}'...")
    
        downloaded_model_weights_path = hf_hub_download(
            repo_id=multimodal_model_hf_repo_id,
            filename=model_weights_filename,
        )
        logging.info(f"Multimodal model weights downloaded to: {downloaded_model_weights_path}")

        inference_dataloader = prepare_inference_dataloader(data_directory, batch_size)
        multimodal_model = load_and_prepare_multimodal_model(downloaded_model_weights_path, device, num_classes)

        with torch.no_grad():
            multimodal_predict_and_save(
                multimodal_model=multimodal_model,
                dataloader=inference_dataloader,
                device=device,
                csv_path=output_csv,
                num_mc_samples=num_mc_samples,
                model_type="multimodal"
            )
        logging.info("Final inference process completed successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred during the inference process: {e}", exc_info=True)
        raise # Re-raise the exception for programmatic callers

def run_auv_retraining(
    root_dir: str,
    devices: List[torch.device],
    const_bnn_prior_parameters: Dict[str, Any], # Keep this passed in as requested
    num_classes: int, # num_classes_for_training will be passed here

    # Direct parameters for optimizer and training:
    lr_multimodal: float = 1e-5,
    multimodal_weight_decay: float = 1e-5,
    epochs_multimodal: int = 20,
    num_mc: int = 5,
    bathy_patch_base: int = 30,
    sss_patch_base: int = 30,
    batch_size_multimodal: int = 1,

    # NEW: Direct parameters for multimodal scheduler:
    scheduler_multimodal_step_size: int = 7,
    scheduler_multimodal_gamma: float = 0.752,
):
    """
    Core function to run the multimodal AUV model retraining process.
    This function contains the main training orchestration logic.
    """
    # Setup logging and tensor board
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    tb_log_dir = os.path.join("tensorboard_logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    sum_writer = SummaryWriter(log_dir=tb_log_dir)
    sum_writer.add_text("Init", "TensorBoard logging started", 0)

    logger = logging.getLogger(__name__) # Re-initialize logger after handlers are set

    # Some preliminary information
    logger.info("Logging initialized.")
    logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
    logger.info("Setting up environment and devices...")
    logger.info(f"Using devices: {[str(d) for d in devices]}")


    # --- Define Dictionaries INSIDE the Function, using passed parameters ---
    optimizer_params = {
        "image_model": {"lr": 1e-5}, # Example fixed LR for unimodal (can make arg if needed)
        "bathy_model": {"lr": 0.01}, # Example fixed LR for unimodal (can make arg if needed)
        "sss_model": {"lr": 1e-5},    # Example fixed LR for unimodal (can make arg if needed)
        "multimodal_model": {"lr": lr_multimodal, "weight_decay": multimodal_weight_decay} # Use passed values
    }

    scheduler_params = {
        "image_model": {"step_size": 7, "gamma": 0.1},
        "bathy_model": {"step_size": 5, "gamma": 0.5},
        "sss_model": {"step_size": 7, "gamma": 0.7},
        "multimodal_model": {
            "step_size": scheduler_multimodal_step_size, # Use passed value
            "gamma": scheduler_multimodal_gamma          # Use passed value
        }
    }

    training_params = {
        "num_epochs_unimodal": 1, # Fixed as per your example, can be made arg
        "num_epochs_multimodal": epochs_multimodal, # Use passed value
        "num_mc": num_mc, # Use passed value
        "bathy_patch_base": f"patch_{bathy_patch_base}_bathy", # Use passed value
        "sss_patch_base": f"patch_{sss_patch_base}_sss",      # Use passed value
        "bathy_patch_types": ["patch_2_bathy", "patch_5_bathy", "patch_10_bathy", "patch_30_bathy", "patch_50_bathy"], # Fixed list
        "sss_patch_types": ["patch_2_sss", "patch_5_sss", "patch_10_sss", "patch_30_sss", "patch_50_sss"],   # Fixed list
        "batch_size_unimodal" : 1, # Fixed, used by prepare_datasets_and_loaders
        "batch_size_multimodal" : batch_size_multimodal # Use passed value
    }
    # --- End of Dictionary Definitions ---

    #preparing dataset loaders
    logger.info("Preparing datasets and data loaders for training...")

    _, _, multimodal_train_loader, multimodal_test_loader, actual_num_classes, _ = \
        prepare_datasets_and_loaders(
            root_dir=root_dir,
            batch_size_multimodal=training_params["batch_size_multimodal"],
            batch_size_unimodal=training_params["batch_size_unimodal"]
        )
    if num_classes != actual_num_classes:
        logger.warning(f"Configured num_classes ({num_classes}) differs from detected num_classes ({actual_num_classes}) from dataset. Using detected.")
        num_classes = actual_num_classes

    logger.info(f"Number of classes (used for model): {num_classes}")
    logger.info(f"Multimodal: {len(multimodal_train_loader.dataset)} training samples, {len(multimodal_test_loader.dataset)} test samples")

    try:
       
            multimodal_model_hf_repo_id = "sams-tom/multimodal-auv-bathy-bnn-classifier"
    
            # --- CRITICAL FIX: Define the subfolder with forward slashes directly,
            #                  or ensure any backslashes are replaced immediately. ---
            # Option 1 (Most direct if you know it's always "multimodal-bnn"):
            multimodal_model_hf_subfolder = "multimodal-bnn" 
    
            # Option 2 (More robust if 'multimodal_model_hf_subfolder' could come from a path operation elsewhere):
            multimodal_model_hf_subfolder = multimodal_model_hf_subfolder.replace('\\', '/')

            # Now, construct the filename using the guaranteed-forward-slash subfolder
            model_weights_filename = f"{multimodal_model_hf_subfolder}/pytorch_model.bin"

            logging.info(f"DEBUG: repo_id being used: '{multimodal_model_hf_repo_id}'")
            logging.info(f"DEBUG: filename being used: '{model_weights_filename}'") # This debug print will confirm the path
            logging.info(f"Attempting to download multimodal model weights from '{multimodal_model_hf_repo_id}' with filename '{model_weights_filename}'...")
    
            downloaded_model_weights_path = hf_hub_download(
                repo_id=multimodal_model_hf_repo_id,
                filename=model_weights_filename,
            )
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not load custom Multimodal Model from {downloaded_model_weights_path}. Cannot proceed with training. Error: {e}")
        sys.exit(1)
    logging.info(f"Multimodal model weights downloaded to: {downloaded_model_weights_path}")
    try:
        multimodal_model_instance = load_and_prepare_multimodal_model_custom(
            downloaded_model_weights_path,
            devices[0], num_classes=num_classes
        )
        logger.info("Custom Multimodal Model loaded successfully with its weights.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not load custom Multimodal Model from {downloaded_model_weights_path}. Cannot proceed with training. Error: {e}")
        sys.exit(1)
    print(multimodal_model_instance)
    
   
    # 2. Define optimiser and schedulers
    logging.info("Defining models...")
    models_dict = define_models(device=devices[0], num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)
    models_dict = move_models_to_device(models_dict, devices, use_multigpu_for_multimodal=True)
    logging.info("Models moved to devices.")
    torch.cuda.empty_cache()

    logging.info("Setting up criterion, optimizers and schedulers...")
    criterion, optimizers, schedulers = define_optimizers_and_schedulers(models_dict, optimizer_params, scheduler_params)

    # 3. Call the main loop with the saved model to retrain
    logger.info("Starting multimodal training with the loaded custom model...")
    print("Starting multimodal training...")
    train_and_evaluate_multimodal_model(
        train_loader=multimodal_train_loader,
        test_loader=multimodal_test_loader,
        multimodal_model=multimodal_model_instance,
        criterion=criterion,
        optimizer=optimizers["multimodal_model"],
        lr_scheduler=schedulers["multimodal_model"],
        num_epochs=training_params["num_epochs_multimodal"],
        device=devices[0],
        model_type="multimodal",
        bathy_patch_type=training_params["bathy_patch_base"],
        sss_patch_type=training_params["sss_patch_base"],
        csv_path=os.path.join(root_dir, "csvs"),
        num_mc=training_params["num_mc"],
        sum_writer=sum_writer
    )
    logger.info("Multimodal training complete.")
    sum_writer.close()
    logger.info("TensorBoard writer closed.")


def run_auv_preprocessing(
    raw_optical_images_folder: str,
    geotiff_folder: str,
    output_folder: str,
    exiftool_path: str,
    window_size_meters: float = 20.0,
    image_enhancement_method: str = "AverageSubtraction",
    skip_bathy_combine: bool = False
):
    """
    Preprocesses AUV sonar and optical image data for machine learning tasks.
    Extracts sonar grid patches, copies original images, and organizes metadata.

    Args:
        raw_optical_images_folder (str): Path to the folder containing raw JPG optical image files.
        geotiff_folder (str): Path to the folder containing all GeoTIFF files (e.g., Bathymetry, Side-Scan Sonar).
        output_folder (str): The root directory where all processed and organized output data will be saved.
                             This will also be the target for processed optical images and their metadata CSV.
        exiftool_path (str): Path to the directory containing the exiftool.exe executable.
        window_size_meters (float, optional): The desired side length (in meters) for the square patches
                                              extracted from GeoTIFFs. Defaults to 20.0.
        image_enhancement_method (str, optional): Method to enhance optical images: 'AverageSubtraction' or 'CLAHE'.
                                                  Defaults to "AverageSubtraction".
        skip_bathy_combine (bool, optional): If True, the post-processing step to combine bathymetry channels will be skipped.
                                             Defaults to False.
    """
    print("\n--- Starting AUV Data Preprocessing ---")
    print(f"Raw Optical Images: {raw_optical_images_folder}")
    print(f"GeoTIFF Folder: {geotiff_folder}")
    print(f"Output Folder: {output_folder}")
    print(f"ExifTool Path: {exiftool_path}")
    print(f"Window Size (meters): {window_size_meters}")
    print(f"Image Enhancement Method: {image_enhancement_method}")
    print(f"Skip Bathy Combine: {skip_bathy_combine}")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # --- Step 1: Pre-process Optical Images and Generate Metadata CSV ---
    print("\n--- Step 1: Pre-processing optical images and generating metadata CSV ---")
    # The processed images and their metadata CSV will be saved into the main output_folder
    ## The 'path' column in the generated CSV will point to these processed images.
    processed_metadata_df = preprocess_optical_images(
        raw_images_path=raw_optical_images_folder,
        processed_images_save_folder=output_folder, # Use main output folder for processed images
        exiftool_executable_path=exiftool_path,
        image_enhancement_method=image_enhancement_method
    )

    if processed_metadata_df.empty:
        print("Optical image pre-processing resulted in no valid metadata. Exiting.")
        sys.exit(1) # Or raise an exception if this function is meant to be called programmatically

    # Set the CSV path for the subsequent steps to the newly generated one
    generated_csv_path = os.path.join(output_folder, 'coords.csv')
    if not os.path.exists(generated_csv_path):
        print(f"Error: Expected metadata CSV not found at '{generated_csv_path}' after pre-processing. Exiting.")
        sys.exit(1) # Or raise an exception
    # The original_images_base_folder for process_and_save_data should now point to
    # the folder where the *processed* optical images are, which is output_folder.
    original_images_base_folder_for_copy = output_folder # Renamed variable for clarity

    print("\n--- Optical image pre-processing completed. ---")


    # --- 2. Identify GeoTIFF files and report resolutions ---
    print("\n--- Step 2: Identifying GeoTIFF files and reporting resolutions ---")
    all_files_in_geotiff_folder = os.listdir(geotiff_folder)
    geotiff_filenames = [f for f in all_files_in_geotiff_folder if is_geotiff(f)]
    geotiff_full_paths = [os.path.join(geotiff_folder, f) for f in geotiff_filenames]

    if not geotiff_full_paths:
        print(f"Warning: No GeoTIFF files found in '{geotiff_folder}'. Sonar data will not be processed.")
    else:
        for f_path in geotiff_full_paths:
            x_res, y_res = get_pixel_resolution(f_path)
            print(f"  Found GeoTIFF: '{os.path.basename(f_path)}', X Resolution: {x_res:.2f}m, Y Resolution: {y_res:.2f}m")
    print("GeoTIFF identification completed.")

    # --- 3. Main Data Processing: Extracting grids, copying images, saving metadata ---
    print("\n--- Step 3: Starting main data processing (extracting grids, copying, saving metadata) ---")
    process_and_save_data(
        csv_file_path=generated_csv_path, # Use the CSV generated in Step 1
        geotiff_files_paths=geotiff_full_paths,
        output_root_folder=output_folder,
        window_size_meters=window_size_meters,
        original_images_folder=original_images_base_folder_for_copy # Use the folder with processed images
    )
    print("\n--- Main data processing completed. ---")

    # --- 4. Post-processing: Combine Bathymetry Channels ---
    if not skip_bathy_combine:
        print("\n--- Step 4: Starting post-processing (combining bathymetry channels) ---")
        process_frame_channels_in_subfolders(output_folder)
        print("\n--- Post-processing completed. ---")
    else:
        print("\n--- Step 4: Skipping bathymetry channel combination as requested. ---")

    print("\n--- AUV Data Preprocessing Finished ---")

def run_AUV_training_from_scratch(
    const_bnn_prior_parameters: Dict[str, Any],
    # ONLY dynamic parameters from 'args' are passed here
    lr_multimodal_model: float,
    num_epochs_multimodal: int,
    num_mc: int,
    bathy_patch_base_raw: int, # Raw integer for patch base
    sss_patch_base_raw: int,   # Raw integer for patch base
    batch_size_multimodal: int,
    # General pipeline params
    root_dir: str,
    devices: List[torch.device],
    num_classes: int
) -> bool:
    """
    Orchestrates the full multimodal AUV model training pipeline from scratch.
    Fixed parameters are defined internally.

    Args:
        const_bnn_prior_parameters (Dict[str, Any]): Parameters for Bayesian Neural Network priors.
        lr_multimodal_model (float): Learning rate for the multimodal model.
        num_epochs_multimodal (int): Number of epochs for multimodal training.
        num_mc (int): Number of Monte Carlo samples.
        bathy_patch_base_raw (int): Base patch size for bathy data (e.g., 30).
        sss_patch_base_raw (int): Base patch size for SSS data (e.g., 30).
        batch_size_unimodal (int): Batch size for unimodal data loaders.
        batch_size_multimodal (int): Batch size for multimodal data loaders.
        root_dir (str): Root directory for datasets and outputs.
        devices (List[torch.device]): List of PyTorch devices to use for training.
        num_classes (int): The number of classes for the classification task.
        
    Returns:
        bool: True if training completes successfully, False otherwise.
    """
    try:
        # --- Logging and TensorBoard Setup (unchanged) ---
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "training.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        tb_log_dir = os.path.join("tensorboard_logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        sum_writer = SummaryWriter(log_dir=tb_log_dir)
        sum_writer.add_text("Init", "TensorBoard logging started", 0)
        logging.info("Logging initialized.")
        logging.info("To view TensorBoard logs, run in terminal: tensorboard --logdir=tensorboard_logs --port=6006")
        logging.info("Then navigate to: http://localhost:6006 in your browser.")

        # --- Device Setup (unchanged) ---
        if not devices:
            primary_device = torch.device("cpu")
            logging.warning("No devices provided or detected. Defaulting to CPU.")
        else:
            primary_device = devices[0]
        logging.info("Setting up environment and devices...")
        logging.info(f"Using primary device: {str(primary_device)}")
        if len(devices) > 1:
            logging.info(f"Additional devices available: {[str(d) for d in devices[1:]]}")


        # --- Define fixed parameters and construct dictionaries INTERNALLY ---
        # Optimizer LRs
        _FIXED_LR_IMAGE_MODEL = 1e-5
        _FIXED_LR_BATHY_MODEL = 0.01
        _FIXED_LR_SSS_MODEL = 1e-5

        optimizer_params = {
            "image_model": {"lr": _FIXED_LR_IMAGE_MODEL},
            "bathy_model": {"lr": _FIXED_LR_BATHY_MODEL},
            "sss_model": {"lr": _FIXED_LR_SSS_MODEL},
            "multimodal_model": {"lr": lr_multimodal_model} # This one is dynamic
        }

        # Scheduler params
        _FIXED_SCHEDULER_STEP_SIZE_IMAGE = 7
        _FIXED_SCHEDULER_GAMMA_IMAGE = 0.1
        _FIXED_SCHEDULER_STEP_SIZE_BATHY = 5
        _FIXED_SCHEDULER_GAMMA_BATHY = 0.5
        _FIXED_SCHEDULER_STEP_SIZE_SSS = 7
        _FIXED_SCHEDULER_GAMMA_SSS = 0.7
        _FIXED_SCHEDULER_STEP_SIZE_MULTIMODAL = 7
        _FIXED_SCHEDULER_GAMMA_MULTIMODAL = 0.752

        scheduler_params = {
            "image_model": {"step_size": _FIXED_SCHEDULER_STEP_SIZE_IMAGE, "gamma": _FIXED_SCHEDULER_GAMMA_IMAGE},
            "bathy_model": {"step_size": _FIXED_SCHEDULER_STEP_SIZE_BATHY, "gamma": _FIXED_SCHEDULER_GAMMA_BATHY},
            "sss_model": {"step_size": _FIXED_SCHEDULER_STEP_SIZE_SSS, "gamma": _FIXED_SCHEDULER_GAMMA_SSS},
            "multimodal_model": {"step_size": _FIXED_SCHEDULER_STEP_SIZE_MULTIMODAL, "gamma": _FIXED_SCHEDULER_GAMMA_MULTIMODAL}
        }

        # Training params
        _FIXED_NUM_EPOCHS_UNIMODAL = 30
        _FIXED_BATHY_PATCH_TYPES = ["patch_2_bathy", "patch_5_bathy", "patch_10_bathy", "patch_30_bathy", "patch_50_bathy"]
        _FIXED_SSS_PATCH_TYPES = ["patch_2_sss", "patch_5_sss", "patch_10_sss", "patch_30_sss", "patch_50_sss"]
        _FIXED_UNIMODAL_BATCH_SIZE = 1 
        training_params = {
            "num_epochs_unimodal": _FIXED_NUM_EPOCHS_UNIMODAL,
            "num_epochs_multimodal": num_epochs_multimodal, # Dynamic
            "num_mc": num_mc, # Dynamic
            "bathy_patch_base": f"patch_{bathy_patch_base_raw}_bathy", # Dynamic (from raw int)
            "sss_patch_base": f"patch_{sss_patch_base_raw}_sss",     # Dynamic (from raw int)
            "bathy_patch_types": _FIXED_BATHY_PATCH_TYPES, # Fixed
            "sss_patch_types": _FIXED_SSS_PATCH_TYPES,     # Fixed
            "batch_size_unimodal" : _FIXED_UNIMODAL_BATCH_SIZE, # Dynamic
            "batch_size_multimodal" : batch_size_multimodal # Dynamic
        }
        # --- End internal dictionary definition ---


        # 2. Dataset and DataLoader Preparation (unchanged in logic)
        logging.info("Preparing datasets and data loaders...")
        _, _, multimodal_train_loader, multimodal_test_loader, actual_num_classes, _ = prepare_datasets_and_loaders(
            root_dir,
            batch_size_unimodal=training_params["batch_size_unimodal"],
            batch_size_multimodal=training_params["batch_size_multimodal"]
        )

        if num_classes is None or num_classes == 0:
            num_classes = actual_num_classes
            logging.info(f"Using num_classes ({num_classes}) derived from dataset.")
        elif num_classes != actual_num_classes:
            logging.warning(f"Configured num_classes ({num_classes}) differs from detected num_classes ({actual_num_classes}) from dataset. Using configured.")
            
        logging.info(f"Number of classes (used for model): {num_classes}")

        # 3. Model Definition and Initialization (unchanged in logic)
        logging.info("Defining models...")
        models_dict = define_models(device=primary_device, num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)
        models_dict = move_models_to_device(models_dict, devices, use_multigpu_for_multimodal=True)
        logging.info("Models defined and moved to devices.")
        torch.cuda.empty_cache()

        # 4. Optimizers and Schedulers (now using the internally defined dictionaries)
        logging.info("Setting up criterion, optimizers and schedulers...")
        criterion, optimizers, schedulers = define_optimizers_and_schedulers(models_dict, optimizer_params, scheduler_params)

        # 5. Run Base Multimodal Training (unchanged in logic)
        logging.info("Starting base multimodal training...")
        print("Starting base multimodal training...") # Use print for immediate visibility

        if "multimodal_model" not in optimizers:
            raise ValueError("Optimizer for 'multimodal_model' not found in optimizers dictionary.")
        if "multimodal_model" not in schedulers:
            raise ValueError("Scheduler for 'multimodal_model' not found in schedulers dictionary.")
        if "multimodal_model" not in models_dict:
            raise ValueError("Multimodal model instance not found in models_dict.")


        train_and_evaluate_multimodal_model(
            train_loader=multimodal_train_loader,
            test_loader=multimodal_test_loader,
            multimodal_model=models_dict["multimodal_model"],
            criterion=criterion,
            optimizer=optimizers["multimodal_model"],
            lr_scheduler=schedulers["multimodal_model"],
            num_epochs=training_params["num_epochs_multimodal"],
            device=primary_device,
            model_type="multimodal",
            bathy_patch_type=training_params["bathy_patch_base"],
            sss_patch_type=training_params["sss_patch_base"],
            csv_path=os.path.join(root_dir, "csvs"),
            num_mc=training_params["num_mc"],
            sum_writer=sum_writer
        )
        logging.info("Base multimodal training complete.")

        sum_writer.close()
        logging.info("TensorBoard writer closed.")
        logging.info("Full training pipeline finished.")
        return True
    except Exception as e:
        logging.exception(f"An error occurred during AUV training from scratch: {e}")
        sum_writer.close()
        return False
