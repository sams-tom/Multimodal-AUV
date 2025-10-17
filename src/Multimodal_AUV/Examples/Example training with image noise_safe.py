# -*- coding: utf-8 -*-
import logging
import torch 
from torch import nn
import torch.optim as optim # Added for optimizer type hint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import csv
from typing import Optional, Tuple, Dict, Any
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
from Multimodal_AUV.train.checkpointing import save_model 
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from pathlib import Path 
from torch.nn.functional import softmax
import datetime
import sys
# Project-specific imports (assumed to be fully functional)
from Multimodal_AUV.models.model_utils import define_models
from Multimodal_AUV.utils.device import move_models_to_device
from Multimodal_AUV.data.loaders import prepare_datasets_and_loaders
from Multimodal_AUV.train.loop_utils import define_optimizers_and_schedulers
from Multimodal_AUV.Examples.Example_Retraining_model import load_and_prepare_multimodal_model_custom
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
import csv
from typing import Optional
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score # Added roc_auc_score for logging
import matplotlib.pyplot as plt # Assuming you have these plotting libraries

# Set up logger for immediate visibility
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


# --- Degradation Helper Function ---
# This remains unchanged and is crucial for the simulation.
def simulate_underwater_degradation(
    clean_image: torch.Tensor, 
    uniform_distance_map: torch.Tensor, 
    turbidity_factor: float,
    depth_value: float 
) -> torch.Tensor:
    """
    Applies the Underwater Image Formation Model (UIFM) to a clean image 
    and a uniform distance map (for a flat seabed).
    """
    B, C, H, W = clean_image.shape
    device = clean_image.device
    
    # 1. Define Wavelength-Dependent Attenuation (Beta)
    mu_R = 0.8  
    mu_G = 0.5
    mu_B = 0.3
    
    beta = torch.tensor(
        [mu_R, mu_G, mu_B], device=device, dtype=clean_image.dtype
    ).view(1, C, 1, 1) * turbidity_factor 

    # 2. Define Ambient Background Light (B_inf) - Backscatter Light
    B_inf = torch.tensor(
        [0.1, 0.3, 0.5], device=device, dtype=clean_image.dtype
    ).view(1, C, 1, 1)

    # 3. Calculate Transmission Map t(x)
    d = uniform_distance_map * depth_value 
    d_broadcast = d.expand(B, C, H, W)
    transmission_map = torch.exp(-beta * d_broadcast)

    # 4. Apply UIFM: I(x) = J(x) * t(x) + B_inf * (1 - t(x))
    direct_component = clean_image * transmission_map
    backscatter_component = B_inf * (1.0 - transmission_map)
    degraded_image = direct_component + backscatter_component
    degraded_image = torch.clamp(degraded_image, 0.0, 1.0)
    
    return degraded_image


# --- Your Updated train_and_evaluate_multimodal_model (with new args added) ---

def train_and_evaluate_multimodal_model(
    train_loader: Any,
    test_loader: Any,
    multimodal_model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    num_mc:int,
    device: torch.device,
    model_type: str,
    bathy_patch_type: str,
    sss_patch_type: str,
    csv_path: str, 
    sum_writer: SummaryWriter,
    # --- ADDED DEGRADATION ARGS ---
    apply_degradation: bool = False,
    turbidity_range: Tuple[float, float] = (0.3, 1.5),
    depth_value: int  = 1   
    # ------------------------------
) -> None:
    """
    Runs training and evaluation for a single multimodal model.
    """
    logging.info("Starting multimodal model training and evaluation")
    logging.info(f"Degradation Enabled: {apply_degradation}, Turbidity Range: {turbidity_range}") # Log the new parameters!
    logging.info(f"Model type: {model_type}")
    # ... (other logging as before)
    logging.info(f"Number of epochs: {num_epochs}")
    # ... (other logging as before)
    
    try:
        os.makedirs(csv_path, exist_ok=True)
        logging.info(f"Ensured CSV output directory exists: {csv_path}")
    except OSError as e:
        logging.error(f"Failed to create CSV output directory {csv_path}: {e}")
        
    for epoch in range(num_epochs): 
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Multimodal model training started.")
        
        # Pass the new arguments to train_multimodal_model
        train_loss, train_accuracy =train_multimodal_model(
            multimodal_model=multimodal_model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch, 
            total_num_epochs=num_epochs, 
            device=device,
            model_type=model_type,
            num_mc=num_mc,
            bathy_patch_type=bathy_patch_type,
            sss_patch_type=sss_patch_type,
            csv_path=os.path.join(csv_path, f"multimodal_training.csv"),
            sum_writer=sum_writer,
            # --- PASS DEGRADATION ARGS ---
            apply_degradation=apply_degradation,
            turbidity_range=turbidity_range,
            depth_value=depth_value
            # -----------------------------
        )
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Training complete.")

        # Your original code stepped the scheduler here, but also steps it after val.
        # It's typically only done once, usually after training/before validation, 
        # or after validation (as done later in your original code).
        # Keeping your original flow for now:
        # lr_scheduler.step() # Removed duplicate step

        val_accuracy = evaluate_multimodal_model(
            multimodal_model=multimodal_model,
            dataloader=test_loader,
            device=device,
            epoch=epoch, 
            total_num_epochs=num_epochs,
            num_mc=num_mc,
            csv_path=os.path.join(csv_path, f"multimodal_test.csv"),
            bathy_patch_type=bathy_patch_type,
            sss_patch_type=sss_patch_type,
            model_type=model_type,
            # --- NEW ---
            apply_degradation=apply_degradation, 
            turbidity_range=turbidity_range,
            depth_value=depth_value

        )

        # Your original code steps the scheduler here, keeping it.
        lr_scheduler.step() 
        sum_writer.add_scalar("train/loss/epoch", train_loss, epoch)
        sum_writer.add_scalar("val/accuracy/epoch", val_accuracy, epoch)
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Evaluation complete.")
        
    logging.info(f"Finished training & evaluation for multimodal model with C:{bathy_patch_type}, S:{sss_patch_type}")


# --- Modified Training Function (from previous response) ---

def train_multimodal_model(
    multimodal_model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    model_type: str ,
    total_num_epochs: int,
    num_mc: int,
    sum_writer: SummaryWriter,
    bathy_patch_type: Optional[str]=None,
    sss_patch_type: Optional[str]=None,
    csv_path: Optional[str]="",
    # --- DEGRADATION ARGS ARE NOW REQUIRED ---
    apply_degradation: bool = False,
    turbidity_range: Tuple[float, float] = (0.3, 1.5),
    depth_value: int  = 1
    # ----------------------------------------
) -> Tuple[float, float]:
    """
    Train a multimodal model... APPLIES UNIFORM UNDERWATER DEGRADATION if enabled.
    """
    multimodal_model.train()
    csv_path = str(Path(csv_path))

    file_exists = os.path.isfile(csv_path)
    try: 
        with open(csv_path, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            if not file_exists:
                csv_writer.writerow(["Epoch", "Model type", "Loss", "Accuracy", "lr", "kl loss", "cross entropy loss", "SSS Patch Type", "Channel Patch Type"])

            total_loss = 0
            correct = 0
            total = 0

            kl_weight = (2 ** (epoch + 1)) / (2 ** total_num_epochs)

            for i, batch in enumerate(dataloader):
                clean_image = batch["main_image"].to(device) 
                labels = batch["label"].long().to(device)
                
                # --- CORE DEGRADATION LOGIC FOR FLAT SEABED ---
                if apply_degradation:
                    B, C, H, W = clean_image.shape
                    
                    # 1. Create a UNIFORM depth map (tensor of ones for full, uniform attenuation)
                    uniform_depth_value = 1.0
                    depth_map = torch.full(
                        (B, 1, H, W), 
                        uniform_depth_value, 
                        device=device, 
                        dtype=clean_image.dtype
                    )
                    
                    # 2. Sample a random turbidity factor for augmentation diversity
                    turbidity = np.random.uniform(low=turbidity_range[0], high=turbidity_range[1])
                    
                    # 3. Apply the degradation model
                    inputs = simulate_underwater_degradation(
                        clean_image, 
                        depth_map, 
                        turbidity_factor=turbidity,
                        depth_value=depth_value
                    )
                else:
                    inputs = clean_image
                # --- END DEGRADATION LOGIC ---

                # Prepare other inputs
                bathy_tensor = batch["bathy_image"].to(device)
                sss_image = batch["sss_image"].to(device)

                patch_bathy = {k: v.to(device) for k, v in batch.get("patch_bathy", {}).items()}
                patch_sss = {k: v.to(device) for k, v in batch.get("patch_sss", {}).items()}
                patch_bathy["patch_30_bathy"] = bathy_tensor
                patch_sss["patch_30_sss"] = sss_image

                bathy_patch = patch_bathy.get(bathy_patch_type, bathy_tensor)
                sss_patch = patch_sss.get(sss_patch_type, sss_image)

                output_ensemble = []
                kl_losses = []

                for _ in range(num_mc):
                    if isinstance(multimodal_model, torch.nn.parallel.DistributedDataParallel):
                        outputs = multimodal_model.module(inputs, bathy_patch, sss_patch)
                    else:
                        outputs = multimodal_model(inputs, bathy_patch, sss_patch)
                    
                    kl_main = get_kl_loss(multimodal_model)
                    output_ensemble.append(outputs)
                    kl_losses.append(kl_main)

                output = torch.mean(torch.stack(output_ensemble), dim=0)
                scaled_kl = torch.mean(torch.stack(kl_losses), dim=0) / dataloader.batch_size * kl_weight
                cross_entropy_loss = criterion(output, labels)
                loss = cross_entropy_loss + scaled_kl

                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    logging.warning(f"Skipping batch {i} due to NaN/Inf loss: {loss}")
                    continue

                optimizer.zero_grad()
                loss.backward()

                if not any(torch.any(torch.isnan(p.grad)) or torch.any(torch.isinf(p.grad)) for p in multimodal_model.parameters() if p.grad is not None):
                    optimizer.step()
                else:
                    logging.warning("Skipping optimizer step due to NaN/Inf gradients")

                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_accuracy = correct / total
            train_loss = total_loss / len(dataloader)
            lr = optimizer.param_groups[0]['lr']

            logging.info(
                f"Epoch {epoch + 1} complete. Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, LR: {lr:.6f}"
            )
            sss_patch_size = sss_patch_type.replace("patch_", "").replace("_sss", "") if sss_patch_type else "full"
            bathy_patch_size = bathy_patch_type.replace("patch_", "").replace("_bathy", "") if bathy_patch_type else "full"

            csv_writer.writerow([
                epoch, model_type, train_loss, train_accuracy, lr,
                scaled_kl.item(), cross_entropy_loss.item(),
                sss_patch_size, bathy_patch_size
            ])

        if epoch % 5 ==0:
            save_model(multimodal_model, csv_path, f"{model_type}_bathy_patch{bathy_patch_size}_sss_patch{sss_patch_size}")

        torch.cuda.empty_cache()
    except Exception as e:
        logging.error(f"Error at epoch {epoch} in train_multimodal_model:", exc_info=True)
        train_loss, train_accuracy = 0.0, 0.0
    return train_loss, train_accuracy


# --- Evaluation Function (Included for completeness/context) ---


def evaluate_multimodal_model(
    multimodal_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    total_num_epochs: int,
    num_mc: int,
    model_type: str,
    bathy_patch_type: Optional[str] = None,
    sss_patch_type: Optional[str] = None,
    csv_path: Optional[str] = "",
    # --- NEW ---
    apply_degradation: bool = False,
    turbidity_range: Tuple[float, float] = (0.3, 1.5),
    depth_value: int  = 1
):
    ...

    """
    Evaluate the multimodal model using MC dropout for uncertainty estimation,
    saving both summary metrics and per-sample data for later deep analysis.
    """

    multimodal_model.train()  # Keep dropout active
    csv_path = Path(csv_path) # Use Path for easier handling
    file_exists = csv_path.exists()
    
    from torch.nn.functional import softmax

    try:  
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        correct = 0
        total = 0
        
        # --- Initialization for Metrics ---
        all_predicted = []
        all_labels = []
        all_predictive_uncertainty = []
        all_model_uncertainty = []
        all_aleatoric_uncertainty = [] 
        
        kl_weight = (2 ** (epoch + 1)) / (2 ** total_num_epochs)
        epsilon = 1e-8  
        all_mean_softmax = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                inputs = batch["main_image"].to(device)
                labels = batch["label"].long().to(device)
                bathy_tensor = batch["bathy_image"].to(device)
                sss_image = batch["sss_image"].to(device)

                patch_bathy = {k: v.to(device) for k, v in batch.get("patch_bathy", {}).items()}
                patch_sss = {k: v.to(device) for k, v in batch.get("patch_sss", {}).items()}
                patch_bathy["patch_30_bathy"] = bathy_tensor
                patch_sss["patch_30_sss"] = sss_image

                bathy_patch = patch_bathy.get(bathy_patch_type, bathy_tensor)
                sss_patch = patch_sss.get(sss_patch_type, sss_image)

                outputs_mc = []
                softmax_outputs_mc = []
                kl_mc = []
                # --- APPLY UNDERWATER DEGRADATION IF ENABLED ---
                if apply_degradation:
                    B, C, H, W = inputs.shape
                    # Uniform depth map
                    uniform_depth_value = 1.0
                    depth_map = torch.full(
                        (B, 1, H, W),
                        uniform_depth_value,
                        device=device,
                        dtype=inputs.dtype
                    )
                    # Random turbidity factor
                    turbidity = np.random.uniform(low=turbidity_range[0], high=turbidity_range[1])
    
                    # Apply degradation
                    inputs = simulate_underwater_degradation(
                        inputs,
                        depth_map,
                        turbidity_factor=turbidity,
                        depth_value = depth_value
                    )

                for _ in range(num_mc):
                        outputs = multimodal_model(inputs, bathy_patch, sss_patch)
                        outputs_mc.append(outputs)
                        softmax_outputs_mc.append(softmax(outputs, dim=1))
                        kl = get_kl_loss(multimodal_model)
                        kl_mc.append(kl)

                outputs_stack = torch.stack(outputs_mc)
                softmax_stack = torch.stack(softmax_outputs_mc)
                output_mean = torch.mean(outputs_stack, dim=0).squeeze(1) if outputs_stack.ndim == 4 else torch.mean(outputs_stack, dim=0)
                mean_softmax = torch.mean(softmax_stack, dim=0) 
                all_mean_softmax.append(mean_softmax.cpu().numpy()) # ADD THIS LINE

                kl_mean = torch.mean(torch.stack(kl_mc), dim=0) / len(dataloader)
                kl_scaled = kl_mean * kl_weight

                cross_entropy_loss = criterion(output_mean, labels)
                total_loss += (cross_entropy_loss + kl_scaled).item()

                _, predicted = torch.max(output_mean, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Uncertainty calculations (unchanged)
                mean_softmax = torch.mean(softmax_stack, dim=0)
                predictive_uncertainty_batch = -torch.sum(mean_softmax * torch.log(mean_softmax + epsilon), dim=1)
                entropy_per_mc_sample = -torch.sum(softmax_stack * torch.log(softmax_stack + epsilon), dim=2)
                aleatoric_uncertainty_batch = torch.mean(entropy_per_mc_sample, dim=0)
                model_uncertainty_batch = predictive_uncertainty_batch - aleatoric_uncertainty_batch

                # --- Extend lists ---
                all_predictive_uncertainty.extend(predictive_uncertainty_batch.cpu().detach().numpy())
                all_model_uncertainty.extend(model_uncertainty_batch.cpu().numpy())
                all_aleatoric_uncertainty.extend(aleatoric_uncertainty_batch.cpu().numpy()) # ?? EXTENDED
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # --- Epoch Summary Metrics ---
        test_accuracy = correct / total
        test_loss = total_loss / len(dataloader)
        predictive_uncertainty_mean = np.mean(all_predictive_uncertainty)
        model_uncertainty_mean = np.mean(all_model_uncertainty)
        aleatoric_uncertainty_mean = np.mean(all_aleatoric_uncertainty) # ?? ADDED: Mean Aleatoric

        
        # --- Save Summary Metrics to CSV ---
        with open(csv_path, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            if not file_exists:
                # ?? UPDATED CSV HEADER
                csv_writer.writerow([
                    "Epoch", "Model Type", "Test Loss", "Test Accuracy",
                    "Predictive Uncertainty", "Model Uncertainty", "Aleatoric Uncertainty",
                    "Scaled KL", "Cross Entropy Loss",
                    "bathy Patch Type", "SSS Patch Type"
                ])

            sss_patch_size = sss_patch_type.replace("patch_", "").replace("_sss", "") if sss_patch_type else "full"
            bathy_patch_size = bathy_patch_type.replace("patch_", "").replace("_bathy", "") if bathy_patch_type else "full"
            
            csv_writer.writerow([
                epoch + 1, model_type, test_loss, test_accuracy,
                predictive_uncertainty_mean, model_uncertainty_mean, aleatoric_uncertainty_mean,
                kl_scaled.item(), cross_entropy_loss.item(),
                bathy_patch_size, sss_patch_size
            ])
            logging.info(f"Epoch {epoch + 1}: Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")


        # --- Save Per-Sample Metrics for Deep Analysis (New Section) ---
        parent_dir = csv_path.parent
        per_sample_dir = parent_dir / "per_sample_metrics"
        per_sample_dir.mkdir(exist_ok=True) 

        # Define file name based on run parameters
        detailed_csv_filename = (
            f"per_sample_run_{model_type}_E{epoch+1}"
            f"_B{bathy_patch_size}_S{sss_patch_size}.csv"
        )
        detailed_csv_path = per_sample_dir / detailed_csv_filename

        per_sample_data = {
            'label': all_labels,
            'prediction': all_predicted,
            'predictive_uncertainty': all_predictive_uncertainty,
            'epistemic_uncertainty': all_model_uncertainty,
            'aleatoric_uncertainty': all_aleatoric_uncertainty # ?? ADDED
        }

        with open(detailed_csv_path, mode='w', newline='') as f:
            fieldnames = per_sample_data.keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Convert columns to rows for writing
            rows = [dict(zip(per_sample_data, t)) for t in zip(*per_sample_data.values())]
            writer.writerows(rows)

        logging.info(f"Per-sample analysis data saved to: {detailed_csv_path}")

        # --- Save AUROC to Main Metrics CSV (Added Section) ---
        try:
            # 1 where the model is wrong
            y_true_error = (np.array(all_predicted) != np.array(all_labels)).astype(int) 
            y_scores = all_predictive_uncertainty # Score is the uncertainty
            auroc = roc_auc_score(y_true_error, y_scores)
            logging.info(f"Uncertainty-Error AUROC (Correlation): {auroc:.4f}")
    
            # ?? ADDITIONS START HERE: Save AUROC to the main CSV ??
    
            # Read the existing main CSV content (assuming csv_path is defined earlier)
            # The last row should contain the metrics for the current run/epoch
            with open(csv_path, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                all_rows = list(reader)

            if all_rows:
                # Get the last row (the one just written for the current epoch/run)
                last_row = all_rows[-1]
        
                # Add the AUROC to the last row
                last_row['uncertainty_error_auroc'] = f"{auroc:.6f}" # Format as string
        
                # Define the new fieldnames (original + new auroc field)
                fieldnames = list(reader.fieldnames)
                if 'uncertainty_error_auroc' not in fieldnames:
                    fieldnames.append('uncertainty_error_auroc')
            
                # Write all rows back to the CSV, including the updated last row
                with open(csv_path, mode='w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_rows)
            
                logging.info(f"Uncertainty-Error AUROC saved to main metrics CSV: {csv_path}")

            # ?? ADDITIONS END HERE ??
    
        except Exception as e:
            logging.warning(f"Could not calculate or save Uncertainty-Error AUROC (required)")

        # --- Plotting Confusion Matrix (Existing Logic Retained) ---
        fig = None
                # --- Compute F1, Calibration Metrics, and Save with Turbidity & Depth ---
        softmax_np_full = np.concatenate(all_mean_softmax, axis=0) # MODIFIED VARIABLE NAME

        try:
            from sklearn.metrics import f1_score

            # --- F1 Score ---
            f1 = f1_score(all_labels, all_predicted, average="macro")

            # --- Expected Calibration Error (ECE) + Emax ---
            def calibration_metrics(probabilities, labels, n_bins=15):
                confidences = np.max(probabilities, axis=1)
                predictions = np.argmax(probabilities, axis=1)
                accuracies = predictions == labels

                bin_boundaries = np.linspace(0, 1, n_bins + 1)
                ece = 0.0
                emax = 0.0

                for i in range(n_bins):
                    in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
                    prop_in_bin = np.mean(in_bin)
                    if prop_in_bin > 0:
                        acc_in_bin = np.mean(accuracies[in_bin])
                        conf_in_bin = np.mean(confidences[in_bin])
                        ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
                        emax = max(emax, np.abs(acc_in_bin - conf_in_bin))
                return ece, emax

            # Use MC-mean softmax probabilities for calibration
            softmax_np = torch.mean(torch.stack(softmax_outputs_mc), dim=0).cpu().numpy()
            ece, emax = calibration_metrics(softmax_np_full, np.array(all_labels)) # MODIFIED CALL

            # --- Append these metrics to main CSV ---
            with open(csv_path, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames

            # Extend header if new fields missing
            new_fields = ["F1_Score", "ECE", "Emax", "Turbidity", "Depth"]
            for nf in new_fields:
                if nf not in fieldnames:
                    fieldnames.append(nf)

            # Update last row
            if rows:
                rows[-1]["F1_Score"] = f"{f1:.4f}"
                rows[-1]["ECE"] = f"{ece:.4f}"
                rows[-1]["Emax"] = f"{emax:.4f}"
                rows[-1]["Turbidity"] = f"{(turbidity_range[0] + turbidity_range[1]) / 2:.3f}"
                rows[-1]["Depth"] = str(depth_value)

            # Write back to CSV
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            logging.info(f"F1={f1:.4f}, ECE={ece:.4f}, Emax={emax:.4f}, Turbidity={(turbidity_range[0] + turbidity_range[1]) / 2:.3f}, Depth={depth_value}")

        except Exception as e:
            logging.warning(f"Could not compute or save F1/ECE/Emax metrics: {e}", exc_info=True)

        try:
            cm = confusion_matrix(all_labels, all_predicted)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots(figsize=(8, 8))
            disp.plot(cmap="Blues", ax=ax)
            plt.title(f"CM for Epoch {epoch}")
            
            parent_path = os.path.dirname(csv_path)
            conf_matrix_folder = os.path.join(parent_path, "confusion_matrices")
            os.makedirs(conf_matrix_folder, exist_ok=True)
            
            matrix_filename = f"conf_matrix_model_{model_type}_{epoch}.png"
            matrix_path = os.path.join(conf_matrix_folder, matrix_filename)
            plt.savefig(matrix_path)
            logging.info(f"Confusion matrix saved to: {matrix_path}")
        except Exception as e:
            logging.warning(f"Confusion matrix not saved due to plotting error: {e}", exc_info=True)
        finally:
            if fig is not None:
                plt.close(fig)

    except Exception as e:
        logging.error(f"Critical error at epoch {epoch} in evaluate_multimodal_model: {e}", exc_info=True)
        test_accuracy = 0.0
    return test_accuracy
def calibration_metrics(probabilities, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE) and Emax.
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    emax = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            emax = max(emax, np.abs(acc_in_bin - conf_in_bin))
    return ece, emax


# --- Main Training Runner (Your original function, updated with new args) ---

def training_main(
    multimodal_model_weights_path: str,
    optimizer_params: Dict[str, Dict[str, Any]],
    scheduler_params: Dict[str, Dict[str, Any]],
    training_params: Dict[str, Any],
    root_dir: str,
    devices: list,
    const_bnn_prior_parameters: Dict[str, Any],
    num_classes: int,
    # --- DEGRADATION CONTROL ARGS ---
    apply_degradation: bool = True,
    turbidity_range: Tuple[float, float] = (0.3, 1.5),
    depth_value: int  = 1
    # --------------------------------
):
    """
    Sets up the environment, loads the model, and initiates the training loop.
    """
    # Setup logging and tensor board
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    log_dir = os.path.join(root_dir, "logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    tb_log_dir = os.path.join(root_dir, "tensorboard_logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    sum_writer = SummaryWriter(log_dir=tb_log_dir)

    device = devices[0]
    logging.info(f"Using device: {device}")

    # 1. Prepare Data Loaders
    logging.info("Preparing datasets and data loaders...")
    _, _, multimodal_train_loader, multimodal_test_loader, num_classes, _ = \
        prepare_datasets_and_loaders(
            root_dir=root_dir,
            batch_size_multimodal=training_params["batch_size_multimodal"],
            batch_size_unimodal=1 
        )
    
    # 2. Load Model
    logging.info(f"Loading custom Multimodal Model from {multimodal_model_weights_path}...")
    models_dict = define_models(device=devices[0], num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)
    multimodal_model_instance=models_dict["multimodal_model"],

    
    # 3. Define Optimizer/Scheduler/Loss
    logging.info("Defining models, optimizers and schedulers...")
    models_dict = {"multimodal_model": multimodal_model_instance} # Use the loaded instance
    models_dict = define_models(device=device, num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)
    models_dict = move_models_to_device(models_dict, devices, use_multigpu_for_multimodal=True)
    
    # Need to correctly pass parameters for the actual model instance
    criterion, optimizers, schedulers = define_optimizers_and_schedulers(models_dict, optimizer_params, scheduler_params)
    
    # 4. Start Training
    logger.info(f"Starting multimodal training for {training_params['num_epochs_multimodal']} epochs.")
    
    # The actual call to the now-complete loop_utils function
    train_and_evaluate_multimodal_model(
        train_loader=multimodal_train_loader,
        test_loader=multimodal_test_loader,
        multimodal_model=models_dict["multimodal_model"],
        criterion=criterion,
        optimizer=optimizers["multimodal_model"],
        lr_scheduler=schedulers["multimodal_model"],
        num_epochs=training_params["num_epochs_multimodal"],
        device=device,
        num_mc=training_params["num_mc"], # num_mc is now passed correctly
        model_type="multimodal",
        bathy_patch_type=training_params["bathy_patch_base"],
        sss_patch_type=training_params["sss_patch_base"],
        csv_path=os.path.join(root_dir, "csvs"),
        sum_writer=sum_writer,
        
        # --- PASSING NEW DEGRADATION PARAMS ---
        apply_degradation=apply_degradation,
        turbidity_range=turbidity_range,
        depth_value = depth_value
        # ------------------------------------
    )
    
    logger.info("Multimodal training complete.")
    sum_writer.close()


if __name__ == "__main__":
    # --- Example Execution Block ---
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    devices = [device]

    # 2. Define Parameters
    ROOT_DIR = "/home/tommorgan/Documents/data/representative_sediment_sample"
    os.makedirs(os.path.join(ROOT_DIR, "csvs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "tensorboard_logs"), exist_ok=True)

   
    # Placeholder for the model path variable used in training_main
    MODEL_WEIGHTS_PATH = None 

    

    # Mock Parameter Dictionaries (Fixed for all runs)

     # Define optimizer and scheduler parameters (only multimodal_model needed)
    MOCK_OPTIMIZER_PARAMS = {
        "image_model": {"lr": 1e-5},
        "bathy_model": {"lr": 0.01},
        "sss_model": {"lr": 1e-5},
        "multimodal_model": {"lr": 5e-5}
    }

    MOCK_SCHEDULER_PARAMS = {
        "image_model": {"step_size": 7, "gamma": 0.1},
        "bathy_model": {"step_size": 5, "gamma": 0.5},
        "sss_model": {"step_size": 7, "gamma": 0.7},
        "multimodal_model": {"step_size": 7, "gamma": 0.752}
    }

    # Define training parameters (only multimodal-specific)
    MOCK_TRAINING_PARAMS = {
        "batch_size_multimodal":8,
        "num_epochs_multimodal": 20,
        "num_mc": 5,
        "bathy_patch_base": "patch_30_bathy",
        "sss_patch_base": "patch_30_sss",
    }
    MOCK_BNN_PRIOR_PARAMS = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.1,
    }
    NUM_CLASSES = 7
    

    # Define 6 steps of degradation from minimal to severe turbidity
    # Steps: 0.05 (near-clean), 0.45, 0.85, 1.25, 1.65, 2.05 (severe)
    turbidity_centers = np.linspace(0.05, 2.05, 6)
    turbidity_delta = 0.05 # Use a small delta to keep the range tight and reproducible
    DEPTH_METERS = np.linspace(0, 25, 6)  
    DEPTH_LEVELS = np.linspace(0.0, 1.0, 6)  # normalized 01 for your UIFM

    # Create the list of (min, max) turbidity tuples
    DEGRADATION_RANGES = [
        (center - turbidity_delta, center + turbidity_delta) 
        for center in turbidity_centers
    ]
    # --- DEGRADATION LOOP EXECUTION ---
    for depth_norm, depth_m in zip(DEPTH_LEVELS, DEPTH_METERS):
        logging.info("#" * 70)
        logging.info(f"STARTING DEPTH LEVEL: {depth_m:.1f} m (normalized {depth_norm:.2f})")
        logging.info("#" * 70)

        for step_index, (turb_min, turb_max) in enumerate(DEGRADATION_RANGES):
            turb_mean = (turb_min + turb_max) / 2
            logging.info("=" * 70)
            logging.info(f"STARTING DEGRADATION STEP {step_index + 1}/6: Turbidity Center: {turb_mean:.2f}")
            logging.info(f"Depth: {depth_m:.1f} m | Turbidity Range: ({turb_min:.2f}, {turb_max:.2f})")
            logging.info("=" * 70)

            # Run your multimodal model training/evaluation
            training_main(
                multimodal_model_weights_path=MODEL_WEIGHTS_PATH,
                optimizer_params=MOCK_OPTIMIZER_PARAMS,
                scheduler_params=MOCK_SCHEDULER_PARAMS,
                training_params=MOCK_TRAINING_PARAMS,
                root_dir=ROOT_DIR,
                devices=devices,
                const_bnn_prior_parameters=MOCK_BNN_PRIOR_PARAMS,
                num_classes=NUM_CLASSES,

                # --- DEGRADATION CONTROL FOR THIS STEP ---
                apply_degradation=True,
                turbidity_range=(turb_min, turb_max),
                depth_value=depth_norm   # <--- NEW ARG (float 01, passed into UIFM)
                # ----------------------------------------
            )