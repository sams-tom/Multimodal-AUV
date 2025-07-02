from huggingface_hub import hf_hub_download # NEW: Import for Hugging Face Hub download
import logging
import torch
import os
from typing import Dict, Any, List
import sys
from torch.utils.tensorboard import SummaryWriter
import datetime
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
        multimodal_model_hf_subfolder = "multimodal-bnn"
        model_weights_filename = os.path.join(multimodal_model_hf_subfolder, "pytorch_model.bin")

        logging.info(f"Attempting to download multimodal model weights from '{multimodal_model_hf_repo_id}/{model_weights_filename}'...")
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


def run_auv_training(
    optimizer_params: Dict[str, Dict[str, Any]],
    scheduler_params: Dict[str, Dict[str, Any]],
    training_params: Dict[str, Any],
    root_dir: str,
    devices: List[torch.device], # Changed to List[torch.device]
    const_bnn_prior_parameters: Dict[str, Any],
    num_classes: int
):
    """
    Core function to run the multimodal AUV model training process.
    This function contains the main training orchestration logic.
    """
    # Setup logging and tensor board
    # Clear existing handlers to prevent duplicate logs if called multiple times
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

    # Some preliminary information
    logger.info("Logging initialized.")
    logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
    logger.info("Setting up environment and devices...")
    logger.info(f"Using devices: {[str(d) for d in devices]}")

    logger.info("Preparing datasets and data loaders for training...")

    # Define the multimodal loaders and get the number of classes. Note _ are unimodal and not needed
    _, _, multimodal_train_loader, multimodal_test_loader, actual_num_classes, _ = \
        prepare_datasets_and_loaders(
            root_dir=root_dir,
            batch_size_multimodal=training_params["batch_size_multimodal"],
            batch_size_unimodal=1
        )
    # If the num_classes derived from data is different from the argument, log a warning or error
    if num_classes != actual_num_classes:
        logger.warning(f"Configured num_classes ({num_classes}) differs from detected num_classes ({actual_num_classes}) from dataset. Using detected.")
        num_classes = actual_num_classes # Prioritize detected classes from data

    logger.info(f"Number of classes (used for model): {num_classes}")
    logger.info(f"Multimodal: {multimodal_train_loader.dataset_size} training samples, {multimodal_test_loader.dataset_size} test samples")

    multimodal_model_hf_repo_id = "sams-tom/multimodal-auv-bathy-bnn-classifier"
    multimodal_model_hf_subfolder = "multimodal-bnn"
    model_weights_filename = os.path.join(multimodal_model_hf_subfolder, "pytorch_model.bin")

    logging.info(f"Attempting to download multimodal model weights from '{multimodal_model_hf_repo_id}/{model_weights_filename}'...")
    downloaded_model_weights_path = hf_hub_download(
            repo_id=multimodal_model_hf_repo_id,
            filename=model_weights_filename,
        )
    logging.info(f"Multimodal model weights downloaded to: {downloaded_model_weights_path}")

    # 2. Define optimiser and schedulers
    # Defining the models so the optimisers know what to do
    logging.info("Defining models...")
    # NOTE: Your `define_models` currently defines ALL models, but `training_main` only uses `multimodal_model_instance`
    # passed directly. Ensure `define_models` provides other models if they are needed elsewhere in your training logic.
    models_dict = define_models(device=devices[0], num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)
    # Add the loaded multimodal_model_instance to the dictionary for consistent handling if needed later
    models_dict = move_models_to_device(models_dict, devices, use_multigpu_for_multimodal=True)
    logging.info("Models moved to devices.")
    torch.cuda.empty_cache()

    # Call the optimisers and schedulers
    logging.info("Setting up criterion, optimizers and schedulers...")
    criterion, optimizers, schedulers = define_optimizers_and_schedulers(models_dict, optimizer_params, scheduler_params)

    # 3. Call the main loop with the saved model to retrain
    logger.info("Starting multimodal training with the loaded custom model...")
    print("Starting multimodal training...") # Use print for immediate visibility
    train_and_evaluate_multimodal_model(
        train_loader=multimodal_train_loader,
        test_loader=multimodal_test_loader,
        multimodal_model=multimodal_model_instance,
        criterion=criterion,
        optimizer=optimizers["multimodal_model"],
        lr_scheduler=schedulers["multimodal_model"],
        num_epochs=training_params["num_epochs_multimodal"],
        device=devices[0], # The primary device for training
        model_type="multimodal",
        bathy_patch_type=training_params["bathy_patch_base"],
        sss_patch_type=training_params["sss_patch_base"],
        csv_path=os.path.join(root_dir, "csvs"), # Assumes csvs dir relative to root_dir
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
    optimizer_params: Dict[str, Dict[str, Any]],
    scheduler_params: Dict[str, Dict[str, Any]],
    training_params: Dict[str, Any],
    root_dir: str,
    devices: List[torch.device], # Corrected type hint to List[torch.device]
    num_classes: int # Added num_classes to the function signature
) -> bool: # Added return type hint for success
    """
    Orchestrates the full multimodal AUV model training pipeline.

    Args:
        const_bnn_prior_parameters (Dict[str, Any]): Parameters for Bayesian Neural Network priors.
        optimizer_params (Dict[str, Dict[str, Any]]): Parameters for optimizers for different models.
        scheduler_params (Dict[str, Dict[str, Any]]): Parameters for learning rate schedulers for different models.
        training_params (Dict[str, Any]): General training parameters (epochs, batch sizes, patch types, etc.).
        root_dir (str): Root directory for datasets and outputs.
        devices (List[torch.device]): List of PyTorch devices to use for training (e.g., [torch.device("cuda:0")]).
        num_classes (int): The number of classes for the classification task.
    
    Returns:
        bool: True if training completes successfully, False otherwise.
    """
    try:
        # Get the root logger and configure it for this run
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Clear any existing handlers to prevent duplicate logs if run multiple times
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Setup file logging
        log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "training.log")

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Setup console logging
        console_handler = logging.StreamHandler(sys.stdout) # Explicitly use sys.stdout
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # Initialize TensorBoard writer
        tb_log_dir = os.path.join("tensorboard_logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        sum_writer = SummaryWriter(log_dir=tb_log_dir)
        sum_writer.add_text("Init", "TensorBoard logging started", 0)

        logging.info("Logging initialized.")

        # Determine primary device (first in the list, or CPU if list is empty/invalid)
        if not devices:
            primary_device = torch.device("cpu")
            logging.warning("No devices provided or detected. Defaulting to CPU.")
        else:
            primary_device = devices[0]

        # 1. Environment and Device Setup
        logging.info("Setting up environment and devices...")
        logging.info(f"Using primary device: {str(primary_device)}")
        if len(devices) > 1:
            logging.info(f"Additional devices available: {[str(d) for d in devices[1:]]}")

        # 2. Dataset and DataLoader Preparation
        logging.info("Preparing datasets and data loaders...")
        # NOTE: The actual num_classes should be derived from the dataset if possible,
        # or passed as an argument to the main function if fixed.
        # Here, we use the one returned by prepare_datasets_and_loaders but also check against the passed `num_classes`.
        _, _, multimodal_train_loader, multimodal_test_loader, actual_num_classes, _ = prepare_datasets_and_loaders(
            root_dir,
            batch_size_unimodal=training_params["batch_size_unimodal"],
            batch_size_multimodal=training_params["batch_size_multimodal"]
        )

        # Harmonize num_classes: prioritize the one from arguments if available, otherwise use detected
        if num_classes is None or num_classes == 0: # If num_classes not provided or invalid
            num_classes = actual_num_classes
            logging.info(f"Using num_classes ({num_classes}) derived from dataset.")
        elif num_classes != actual_num_classes:
            logging.warning(f"Configured num_classes ({num_classes}) differs from detected num_classes ({actual_num_classes}) from dataset. Using configured.")
            # Decide which to use, for training from scratch, it might be better to stick to configured
            # For retraining, usually dataset detected is preferred. Let's stick with provided for from_scratch.
            # If you want to use actual_num_classes here, uncomment: num_classes = actual_num_classes
        
        logging.info(f"Number of classes (used for model): {num_classes}")

        # 3. Model Definition and Initialization
        logging.info("Defining models...")
        models_dict = define_models(device=primary_device, num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)

        # Move models to appropriate devices (if DataParallel/Distributed used)
        models_dict = move_models_to_device(models_dict, devices, use_multigpu_for_multimodal=True)
        logging.info("Models defined and moved to devices.")
        torch.cuda.empty_cache()

        # 4. Optimizers and Schedulers
        logging.info("Setting up criterion, optimizers and schedulers...")
        criterion, optimizers, schedulers = define_optimizers_and_schedulers(models_dict, optimizer_params, scheduler_params)

        # 5. Run Base Multimodal Training
        logging.info("Starting base multimodal training...")
        print("Starting base multimodal training...") # Use print for immediate visibility

        # Ensure the 'multimodal_model' key exists in optimizers and schedulers
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
            device=primary_device, # Pass the primary device for training loop management
            model_type="multimodal",
            bathy_patch_type=training_params["bathy_patch_base"],
            sss_patch_type=training_params["sss_patch_base"],
            csv_path=os.path.join(root_dir, "csvs"), # Ensure this path is correctly formed
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