
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

def simulate_underwater_degradation(
    clean_image: torch.Tensor, 
    # uniform_distance_map: torch.Tensor, # Removed from signature as it's confusing with fixed depth
    fixed_depth_norm: float, # NEW ARG: normalized depth (0.0 to 1.0)
    turbidity_factor: float
) -> torch.Tensor:
    """
    Applies the Underwater Image Formation Model (UIFM) to a clean image 
    at a specific, uniform depth (for a flat seabed).
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
    D_max = 25.0 
    
    # NEW: Create the distance map from the fixed normalized depth
    d_norm = torch.ones((B, 1, H, W), device=device, dtype=clean_image.dtype) * fixed_depth_norm
    d = d_norm * D_max
    
    d_broadcast = d.expand(B, C, H, W)
    transmission_map = torch.exp(-beta * d_broadcast)

    # 4. Apply UIFM: I(x) = J(x) * t(x) + B_inf * (1 - t(x))
    direct_component = clean_image * transmission_map
    backscatter_component = B_inf * (1.0 - transmission_map)
    degraded_image = direct_component + backscatter_component
    degraded_image = torch.clamp(degraded_image, 0.0, 1.0)
    
    return degraded_image

# --- Your Updated train_and_evaluate_multimodal_model (MODIFIED) ---
def train_and_evaluate_unimodal_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    model_name: str,
    save_dir: str,
    num_mc: int,
    sum_writer: SummaryWriter,
    apply_degradation: bool = False,
    turbidity_range: Tuple[float, float] = (0.3, 1.0),
    fixed_depth_norm: float = 0.5, # NEW ARG
) -> None:
    logging.info("Starting training of unimodal model")
    logging.info(f"Model: {model.__class__.__name__}")
    logging.info(f"Number of epochs: {num_epochs}")
    logging.info(f"Device: {device}")
    logging.info(f"Model name: {model_name}")
    logging.info(f"Save directory: {save_dir}")
    logging.info(f"Number of Monte Carlo samples (num_mc): {num_mc}")
    logging.info(f"Fixed Normalized Depth: {fixed_depth_norm:.3f}") # NEW LOGGING


    os.makedirs(save_dir, exist_ok=True)


    for epoch in range(1, num_epochs + 1):
        logging.debug(f"Epoch {epoch}/{num_epochs} - Training")
        train_accuracy, train_loss = train_unimodal_model(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            total_num_epochs=num_epochs,
            num_mc=num_mc,
            sum_writer=sum_writer,
            device=device,
            model_type=model_name,
            csv_path=os.path.join(save_dir, f"{model_name}.csv"),
            patch_type=None,
            apply_degradation=apply_degradation,
            turbidity_range=turbidity_range,
            fixed_depth_norm=fixed_depth_norm, # PASS NEW ARG
        )


        logging.debug(f"Epoch {epoch}/{num_epochs} - Evaluation")
        val_accuracy = evaluate_unimodal_model(
            model=model,
            dataloader=test_loader,
            device=device,
            epoch=epoch,
            total_num_epochs=num_epochs,
            num_mc=num_mc,
            model_type=model_name,
            csv_path=os.path.join(save_dir, f"{model_name}_evaluate.csv"),
            patch_type=None,
            apply_degradation=apply_degradation,
            turbidity_range=turbidity_range,
            fixed_depth_norm=fixed_depth_norm, # PASS NEW ARG
        )


        scheduler.step()
        sum_writer.add_scalar("train/loss/epoch", train_loss, epoch)
        sum_writer.add_scalar("val/accuracy/epoch", val_accuracy, epoch)


        logging.info(f"Epoch {epoch} completed for unimodal model: {model_name}")
    

# --- Modified Training Function (MODIFIED) ---
def train_unimodal_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_num_epochs: int,
    num_mc: int,
    sum_writer: SummaryWriter,
    device: torch.device,
    model_type: str = "image",
    csv_path: str = "",
    patch_type: Optional[str] = None,
    apply_degradation: bool = False,
    turbidity_range: Tuple[float, float] = (0.3, 1.0),
    fixed_depth_norm: float = 0.5, # NEW ARG
):
    """
    Trains for one epoch. If apply_degradation=True, applies UIFM-based
    degradation on the fly using the fixed_depth_norm value.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    model.train()
    model.to(device)


    kl_weight = (2 ** (epoch + 1)) / (2 ** total_num_epochs)
    file_exists = os.path.isfile(csv_path)


    logging.info(f"Starting training epoch {epoch + 1}/{total_num_epochs} for model type: {model_type}")
    try:
        with open(csv_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["Epoch", "Model type", "Loss", "Accuracy", "lr"])


            total_loss, correct, total = 0.0, 0, 0


            for i, batch in enumerate(dataloader):
                logging.debug(f"Train batch {i+1}/{len(dataloader)} - Model: {model_type}")


                inputs = batch["main_image"].to(device)
                labels = batch["label"].long().to(device)
                bathy_tensor = batch.get("bathy_image")
                sss_image = batch.get("sss_image")


                if bathy_tensor is not None:
                    bathy_tensor = bathy_tensor.to(device)
                if sss_image is not None:
                    sss_image = sss_image.to(device)


                patch_bathy = {k: v.to(device) for k, v in batch.get("patch_bathy", {}).items()}
                patch_sss = {k: v.to(device) for k, v in batch.get("patch_sss", {}).items()}


                all_patches = {**patch_bathy, **patch_sss}
                if bathy_tensor is not None:
                    all_patches["patch_30_bathy"] = bathy_tensor
                if sss_image is not None:
                    all_patches["patch_30_sss"] = sss_image


                patch = all_patches.get(patch_type) if patch_type else None


                # Apply degradation if requested
                if apply_degradation:
                    # Determine turbidity factor for this batch
                    t_min, t_max = turbidity_range
                    turbidity_factor = float(np.random.uniform(t_min, t_max))
                    
                    # Call degradation with the fixed_depth_norm
                    inputs = simulate_underwater_degradation(
                        inputs, 
                        # distance_map, # Removed the confusing distance_map from args
                        fixed_depth_norm, # Pass the new depth arg
                        turbidity_factor
                    )
                
                # The rest of the training loop remains the same...
                optimizer.zero_grad()
                if model_type == "image":
                    model_input = inputs
                elif model_type == "sss":
                    model_input = sss_image
                elif model_type == "bathy":
                    model_input = bathy_tensor
                else:
                    logging.error(f"Unknown model_type: {model_type}")
                    raise ValueError(f"Unknown model_type: {model_type}")


                outputs, kl_losses = [], []
                for _ in range(num_mc):
                    out = model(model_input)
                    outputs.append(out)
                    kl_losses.append(get_kl_loss(model))


                output = torch.mean(torch.stack(outputs), dim=0)
                scaled_kl = torch.mean(torch.stack(kl_losses), dim=0) / max(getattr(dataloader, "batch_size", 1), 1)


                cross_entropy_loss = criterion(output, labels)
                loss = cross_entropy_loss + (kl_weight * scaled_kl)


                loss.backward()
                optimizer.step()


                output = output.float()
                _, predicted = output.max(1)


                total_loss += loss.item() * labels.size(0)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                sum_writer.add_scalar("Loss/train", loss.item(), epoch * len(dataloader) + i)


            train_accuracy = correct / total if total > 0 else 0.0
            train_loss = total_loss / total if total > 0 else 0.0
            lr = optimizer.param_groups[0]["lr"]


            logging.info(f"Epoch {epoch} | Final Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f} | LR: {lr:.6f}")
            writer.writerow([epoch, model_type, train_loss, train_accuracy, lr])


        if epoch % 5 == 0:
            save_model(model, csv_path, model_type)


    except Exception as e:
        save_model(model, csv_path, model_type)
        logging.error(f"Error at epoch {epoch}: {e}", exc_info=True)
        train_accuracy, train_loss = 0.0, 0.0


    return train_accuracy, train_loss


# --- Modified Evaluation Function (MODIFIED) ---
def evaluate_unimodal_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    csv_path: str,
    total_num_epochs: int,
    num_mc: int,
    model_type: str = "image",
    patch_type: Optional[str] = None,
    apply_degradation: bool = False,
    turbidity_range: Tuple[float, float] = (0.3, 1.0),
    fixed_depth_norm: float = 0.5, # NEW ARG
):
    """
    Evaluate the model; if apply_degradation=True the same degradation logic
    as training is applied (draw new turbidity per-batch) using fixed_depth_norm.
    Returns accuracy.
    """
    model.train() # keep dropout active for MC dropout
    criterion = nn.CrossEntropyLoss()
    kl_weight = (2 ** (epoch + 1)) / (2 ** total_num_epochs)


    file_exists = os.path.isfile(csv_path)
    logging.info(f"Starting evaluation for epoch {epoch} (model type: {model_type})")


    try:
        with open(csv_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["Epoch", "Model Type", "Test Loss", "Test Accuracy", "predictive_uncertainty", "model_uncertainty"])


            correct, total, total_loss = 0, 0, 0.0
            # ... (omitted initial uncertainty lists for brevity)
            all_predictive_uncertainties = []
            all_aleatoric_uncertainties = []
            all_predicted = []
            all_labels = []
            all_predictive_uncertainty = []
            all_model_uncertainty = []
            all_mean_softmax = []
            epsilon = 1e-8  

            for i, batch in enumerate(dataloader):
                logging.debug(f"Eval batch {i+1}/{len(dataloader)} - Model: {model_type}")

                softmax_outputs_mc = []

                inputs = batch["main_image"].to(device)
                labels = batch["label"].long().to(device)
                bathy_tensor = batch.get("bathy_image")
                sss_image = batch.get("sss_image")

                # ... (omitted patch/bathy/sss loading for brevity)
                if bathy_tensor is not None:
                    bathy_tensor = bathy_tensor.to(device)
                if sss_image is not None:
                    sss_image = sss_image.to(device)
                patch_bathy = {k: v.to(device) for k, v in batch.get("patch_bathy", {}).items()}
                patch_sss = {k: v.to(device) for k, v in batch.get("patch_sss", {}).items()}
                all_patches = {**patch_bathy, **patch_sss}
                if bathy_tensor is not None:
                    all_patches["patch_30_bathy"] = bathy_tensor
                if sss_image is not None:
                    all_patches["patch_30_sss"] = sss_image
                patch = all_patches.get(patch_type) if patch_type else None


                # Degrade if requested
                if apply_degradation:
                    t_min, t_max = turbidity_range
                    turbidity_factor = float(np.random.uniform(t_min, t_max))

                    # Call degradation with the fixed_depth_norm
                    inputs = simulate_underwater_degradation(
                        inputs, 
                        # distance_map, # Removed the confusing distance_map from args
                        fixed_depth_norm, # Pass the new depth arg
                        turbidity_factor
                    )
                # ... (omitted model input selection for brevity)
                if model_type == "image":
                    model_input = inputs
                elif model_type == "sss":
                    model_input = sss_image
                elif model_type == "bathy":
                    model_input = bathy_tensor
                else:
                    logging.error(f"Unknown model_type: {model_type}")
                    raise ValueError(f"Unknown model_type: {model_type}")

                outputs_mc_logits, kl_losses = [], []
                for _ in range(num_mc):
                    out_logits = model(model_input)
                    outputs_mc_logits.append(out_logits)
                    kl_losses.append(get_kl_loss(model))
                    softmax_outputs_mc.append(softmax(out_logits, dim=1))

                softmax_stack = torch.stack(softmax_outputs_mc)
                output_mean_logits = torch.mean(torch.stack(outputs_mc_logits), dim=0)
                scaled_kl = torch.mean(torch.stack(kl_losses), dim=0) / max(getattr(dataloader, "batch_size", 1), 1)
                cross_entropy_loss = criterion(output_mean_logits, labels)
                loss = cross_entropy_loss + (kl_weight * scaled_kl)
                mean_softmax = torch.mean(softmax_stack, dim=0) 
                all_mean_softmax.append(mean_softmax.cpu().detach().numpy()) 

                probabilities_mean = torch.softmax(output_mean_logits, dim=-1)
                _, predicted = probabilities_mean.max(1)


                total_loss += loss.item() * labels.size(0)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)


                prob_outputs_list = [torch.softmax(out_logits, dim=-1) for out_logits in outputs_mc_logits]
                prob_outputs_mc = torch.stack(prob_outputs_list, dim=0)


                # Variance-based Epistemic Uncertainty (Predictive Uncertainty in context)
                variance_per_class = torch.var(prob_outputs_mc, dim=0)
                epistemic_uncertainty_batch = variance_per_class.mean(dim=1)
                all_predictive_uncertainties.extend(epistemic_uncertainty_batch.cpu().detach().numpy())
                    
                # Entropy-based Predictive, Aleatoric, and Model Uncertainty
                mean_softmax = torch.mean(softmax_stack, dim=0)
                predictive_uncertainty_batch = -torch.sum(mean_softmax * torch.log(mean_softmax + epsilon), dim=1)
                
                # Aleatoric uncertainty (mean of entropies of individual MC samples)
                entropy_per_mc_sample = -torch.sum(softmax_stack * torch.log(softmax_stack + epsilon), dim=2)
                aleatoric_uncertainty_batch = torch.mean(entropy_per_mc_sample, dim=0)
                
                # Model (Epistemic) Uncertainty is the difference: Predictive - Aleatoric
                model_uncertainty_batch = predictive_uncertainty_batch - aleatoric_uncertainty_batch


                all_predictive_uncertainty.extend(predictive_uncertainty_batch.cpu().detach().numpy())
                all_model_uncertainty.extend(model_uncertainty_batch.cpu().detach().numpy())
                all_aleatoric_uncertainties.extend(aleatoric_uncertainty_batch.cpu().detach().numpy())
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            accuracy = correct / total if total > 0 else 0.0
            avg_loss = total_loss / total if total > 0 else 0.0


            avg_predictive_uncertainty = np.mean(all_predictive_uncertainty) if all_predictive_uncertainty else 0.0
            avg_aleatoric_uncertainty = np.mean(all_aleatoric_uncertainties) if all_aleatoric_uncertainties else 0.0
            avg_model_uncertainty = np.mean(all_model_uncertainty) if all_model_uncertainty else 0.0

            logging.info(
            f"Eval Epoch {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | "
            f"Predictive UQ: {avg_predictive_uncertainty:.6f} | Aleatoric UQ: {avg_aleatoric_uncertainty:.6f}"
            )
            # Re-write the last row to include the correct aleatoric
            writer.writerow([
            epoch,
            model_type,
            avg_loss,
            accuracy,
            avg_predictive_uncertainty, # Using entropy-based predictive UQ for log/csv
            avg_model_uncertainty, # Using entropy-based model UQ for log/csv
            ])
            
            # ... (omitted confusion matrix code for brevity)
            try:
                cm = confusion_matrix(all_labels, all_predicted)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                fig, ax = plt.subplots(figsize=(8, 8))
                disp.plot(cmap="Blues", ax=ax)
                plt.title(f"Confusion Matrix for Epoch {epoch}")
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
                try:
                    plt.close(fig)
                except Exception:
                     pass
            
            # --- Save Per-Sample Metrics for Deep Analysis (MODIFIED) ---
            parent_path = os.path.dirname(csv_path)
            per_sample_dir = os.path.join(parent_path, "per_sample_metrics")
            os.makedirs(per_sample_dir, exist_ok=True)

            # Define file name based on run parameters
            detailed_csv_filename = (
                f"per_sample_run_{model_type}_E{epoch}" # use epoch instead of epoch+1
            )
            detailed_csv_path = os.path.join(per_sample_dir, detailed_csv_filename)

            per_sample_data = {
                'label': all_labels,
                'prediction': all_predicted,
                'predictive_uncertainty': all_predictive_uncertainty,
                'epistemic_uncertainty': all_model_uncertainty,
                'aleatoric_uncertainty': all_aleatoric_uncertainties # NOW CORRECTLY SAVED
            }

            with open(detailed_csv_path, mode='w', newline='') as f:
                fieldnames = per_sample_data.keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Convert columns to rows for writing
                rows = [dict(zip(per_sample_data, t)) for t in zip(*per_sample_data.values())]
                writer.writerows(rows)

            logging.info(f"Per-sample analysis data saved to: {detailed_csv_path}")

            # --- Save AUROC, F1, ECE, Emax, Turbidity, and Depth to Main Metrics CSV (MODIFIED) ---
            try:
                # 1 where the model is wrong
                y_true_error = (np.array(all_predicted) != np.array(all_labels)).astype(int) 
                y_scores = all_predictive_uncertainty # Score is the uncertainty
                auroc = roc_auc_score(y_true_error, y_scores)
                logging.info(f"Uncertainty-Error AUROC (Correlation): {auroc:.4f}")
        
                # Read the existing main CSV content (assuming csv_path is defined earlier)
                with open(csv_path, mode='r', newline='') as f:
                    reader = csv.DictReader(f)
                    all_rows = list(reader)

                # Fetching the un-normalized depth in meters for logging/CSV
                depth_value = fixed_depth_norm * 25.0 # Max depth in UIFM is 25.0
                softmax_np_full = np.concatenate(all_mean_softmax, axis=0) # MODIFIED VARIABLE NAME

                if all_rows:
                    # Get the last row (the one just written for the current epoch/run)
                    last_row = all_rows[-1]
        
                    # Define the new fieldnames
                    fieldnames = list(reader.fieldnames)
                    new_metrics = ['uncertainty_error_auroc', "F1_Score", "ECE", "Emax", "Turbidity", "Depth"]
                    for nf in new_metrics:
                        if nf not in fieldnames:
                            fieldnames.append(nf)
                    
                    # Compute F1 and ECE/Emax
                    from sklearn.metrics import f1_score
                    f1 = f1_score(all_labels, all_predicted, average="macro")
                    
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
                    
                    ece, emax = calibration_metrics(softmax_np_full, np.array(all_labels))


                    # Add the metrics to the last row
                    last_row['uncertainty_error_auroc'] = f"{auroc:.6f}"
                    last_row["F1_Score"] = f"{f1:.4f}"
                    last_row["ECE"] = f"{ece:.4f}"
                    last_row["Emax"] = f"{emax:.4f}"
                    last_row["Turbidity"] = f"{(turbidity_range[0] + turbidity_range[1]) / 2:.3f}"
                    last_row["Depth"] = f"{depth_value:.1f}" # Save the un-normalized depth in meters
        
                    # Write all rows back to the CSV, including the updated last row
                    with open(csv_path, mode='w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(all_rows)
        
                    logging.info(f"Uncertainty-Error AUROC saved to main metrics CSV: {csv_path}")
                    logging.info(f"F1={f1:.4f}, ECE={ece:.4f}, Emax={emax:.4f}, Turbidity={(turbidity_range[0] + turbidity_range[1]) / 2:.3f}, Depth={depth_value:.1f}m")

            except Exception as e:
                logging.warning(f"Could not calculate or save AUROC/F1/ECE/Emax metrics: {e}", exc_info=True)

    except Exception as e:
        save_model(model, csv_path, model_type)
        logging.error(f"Error at epoch {epoch}: {e}", exc_info=True)
        accuracy = 0.0


    return accuracy

# --- Main Training Runner (MODIFIED) ---
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
    fixed_depth_norm: float = 0.5, # NEW ARG
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
    
    models_dict = define_models(device=device, num_classes=num_classes, const_bnn_prior_parameters=const_bnn_prior_parameters)
    models_dict = move_models_to_device(models_dict, devices, use_multigpu_for_multimodal=True)
    
    # Need to correctly pass parameters for the actual model instance
    criterion, optimizers, schedulers = define_optimizers_and_schedulers(models_dict, optimizer_params, scheduler_params)
    
    # 4. Start Training
    logger.info(f"Starting unimodal training for {training_params['num_epochs_multimodal']} epochs.")
    
    # The actual call to the now-complete loop_utils function
    train_and_evaluate_unimodal_model(
        train_loader=multimodal_train_loader,
        test_loader=multimodal_test_loader,
        model=models_dict["image_model"],
        criterion=criterion,
        optimizer=optimizers["image_model"],
        scheduler=schedulers["image_model"],
        num_epochs=training_params["num_epochs_multimodal"],
        device=device,
        num_mc=training_params["num_mc"], # num_mc is now passed correctly
        model_name="image",
        
        save_dir=os.path.join(root_dir, "csvs"),
        sum_writer=sum_writer,
        
        # --- PASSING NEW DEGRADATION PARAMS ---
        apply_degradation=apply_degradation,
        turbidity_range=turbidity_range,
        fixed_depth_norm=fixed_depth_norm, # PASS NEW ARG
        # ------------------------------------
    )
    
    logger.info("Unimodal training complete.")
    sum_writer.close()

    
if __name__ == "__main__":
    # 1. Setup Device - prefer second GPU (cuda:1) if available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device("cuda:1")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.warning("Only one CUDA device found; defaulting to cuda:0")
    else:
        device = torch.device("cpu")


    devices = [device]


    # 2. Define Parameters
    ROOT_DIR = "/home/tommorgan/Documents/data/representative_sediment_sample"
    os.makedirs(os.path.join(ROOT_DIR, "csvs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "tensorboard_logs"), exist_ok=True)


    MODEL_WEIGHTS_PATH = None


    MOCK_OPTIMIZER_PARAMS = {
    "image_model": {"lr": 1e-5},
    "bathy_model": {"lr": 0.01},
    "sss_model": {"lr": 1e-5},
    "multimodal_model": {"lr": 5e-5},
    }


    MOCK_SCHEDULER_PARAMS = {
    "image_model": {"step_size": 7, "gamma": 0.1},
    "bathy_model": {"step_size": 5, "gamma": 0.5},
    "sss_model": {"step_size": 7, "gamma": 0.7},
    "multimodal_model": {"step_size": 7, "gamma": 0.752},
    }


    MOCK_TRAINING_PARAMS = {
    "batch_size_multimodal": 8,
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


    # --- NEW DEGRADATION CONFIGURATION ---
    turbidity_centers = np.linspace(0.05, 2.05, 6)
    turbidity_delta = 0.05
    DEPTH_METERS = np.linspace(0, 25, 6)
    DEPTH_LEVELS = np.linspace(0.0, 1.0, 6) # normalized 0-1 for your UIFM

    # Create the list of (min, max) turbidity tuples
    DEGRADATION_RANGES = [
        (center - turbidity_delta, center + turbidity_delta) 
        for center in turbidity_centers
    ]
    # -------------------------------------


    # --- DEGRADATION LOOP EXECUTION (MODIFIED FOR NESTED DEPTH/TURBIDITY) ---
    for depth_norm, depth_m in zip(DEPTH_LEVELS, DEPTH_METERS):
        logging.info("#" * 70)
        logging.info(f"STARTING DEPTH LEVEL: {depth_m:.1f} m (normalized {depth_norm:.2f})")
        logging.info("#" * 70)

        for step_index, (turb_min, turb_max) in enumerate(DEGRADATION_RANGES):
            turb_mean = (turb_min + turb_max) / 2
            logging.info("=" * 70)
            logging.info(f"STARTING DEGRADATION STEP {step_index + 1}/{len(DEGRADATION_RANGES)}: Turbidity Center: {turb_mean:.2f}")
            logging.info(f"Depth: {depth_m:.1f} m | Turbidity Range: ({turb_min:.2f}, {turb_max:.2f})")
            logging.info("=" * 70)

            training_main(
                multimodal_model_weights_path=MODEL_WEIGHTS_PATH,
                optimizer_params=MOCK_OPTIMIZER_PARAMS,
                scheduler_params=MOCK_SCHEDULER_PARAMS,
                training_params=MOCK_TRAINING_PARAMS,
                root_dir=ROOT_DIR,
                devices=devices,
                const_bnn_prior_parameters=MOCK_BNN_PRIOR_PARAMS,
                num_classes=NUM_CLASSES,
                apply_degradation=True,
                turbidity_range=(turb_min, turb_max),
                fixed_depth_norm=depth_norm, # PASSING THE CURRENT NORMALIZED DEPTH
            )


    logging.info("Training script execution finished for ALL degradation levels.")