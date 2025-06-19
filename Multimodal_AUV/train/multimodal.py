import logging
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import csv
from typing import Optional, Tuple
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
from Multimodal_AUV.train.checkpointing import save_model
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg') # This must be called *before* importing matplotlib.pyplot
import matplotlib.pyplot as plt

# Try to set a generic sans-serif font that is commonly available
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Helvetica', 'Verdana']


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
    channel_patch_type: Optional[str]=None,
    sss_patch_type: Optional[str]=None,
    csv_path: Optional[str]=""
) -> Tuple[float, float]:
    """
    Train a multimodal model that integrates multiple input types with Monte Carlo dropout,
    computing loss (including KL divergence), accuracy, and logging progress to CSV.

    Args:
        multimodal_model (nn.Module): The multimodal PyTorch model to train.
        dataloader (DataLoader): DataLoader providing training data batches.
        criterion (callable): Loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        epoch (int): Current epoch index, used for KL weight scaling.
        device (torch.device): Computation device (CPU/GPU).
        model_type (str): Label describing the model type for logging.
        channel_patch_type (str or None, optional): Patch type for channel data.
        sss_patch_type (str or None, optional): Patch type for SSS data.
        csv_path (str, optional): Path to CSV file for logging training metrics.

    Returns:
        Tuple[float, float]: Training loss and accuracy for the current epoch.
    """
    # Set the model to training mode
    multimodal_model.train()

    # Check if CSV file already exists
    file_exists = os.path.isfile(csv_path)
    try: 
        # Open the CSV file in append mode to log training stats
        with open(csv_path, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            # If file does not exist, write the header row
            if not file_exists:
                csv_writer.writerow(["Epoch", "Model type", "Loss", "Accuracy", "lr", "kl loss", "cross entropy loss", "SSS Patch Type", "Channel Patch Type"])

            # Initialize accumulators for tracking loss and accuracy
            total_loss = 0
            correct = 0
            total = 0

            # Define KL weight (based on epoch) for Bayesian regularization
            kl_weight = (2 ** (epoch + 1)) / (2 ** total_num_epochs)

            # Iterate through the dataloader
            for i, batch in enumerate(dataloader):
                logging.info(f"Train batch {i+1}/{len(dataloader)} - Model: {model_type}")

                # Move core tensors to device
                inputs = batch["main_image"].to(device)
                labels = batch["label"].long().to(device)
                channels_tensor = batch["channel_image"].to(device)
                sss_image = batch["sss_image"].to(device)

                # Move all patch channels and SSS patches to device dynamically
                patch_channels = {k: v.to(device) for k, v in batch.get("patch_channels", {}).items()}
                patch_sss = {k: v.to(device) for k, v in batch.get("patch_sss", {}).items()}

                # Add default full tensors with special keys
                patch_channels["patch_30_channel"] = channels_tensor
                patch_sss["patch_30_sss"] = sss_image

                # Select patches based on provided patch types (or default to full tensor)
                channel_patch = patch_channels.get(channel_patch_type, channels_tensor)  # fallback to full
                sss_patch = patch_sss.get(sss_patch_type, sss_image)                    # fallback to full

                output_ensemble = []
                kl_losses = []

                for _ in range(num_mc):
                    # Support DistributedDataParallel
                    if isinstance(multimodal_model, torch.nn.parallel.DistributedDataParallel):
                        outputs = multimodal_model.module(inputs, channel_patch, sss_patch)
                    else:
                        outputs = multimodal_model(inputs, channel_patch, sss_patch)
                    # Compute KL divergence for this pass
                    kl_main = get_kl_loss(multimodal_model)

                    # Store the output and KL loss
                    output_ensemble.append(outputs)
                    kl_losses.append(kl_main)

                # Compute mean output across all MC samples
                output = torch.mean(torch.stack(output_ensemble), dim=0)

                # Compute scaled KL loss (average over MC, scale by batch size and kl weight)
                scaled_kl = torch.mean(torch.stack(kl_losses), dim=0) / dataloader.batch_size * kl_weight

                # Compute Cross Entropy Loss
                cross_entropy_loss = criterion(output, labels)

                # Combine both losses
                loss = cross_entropy_loss + scaled_kl

                # Skip batch if loss contains NaN or Inf
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    logging.warning(f"Skipping batch {i} due to NaN/Inf loss: {loss}")
                    continue

                # Backward pass
                loss.backward()

                # Check if gradients are valid before optimizer step
                if not any(torch.any(torch.isnan(p.grad)) or torch.any(torch.isinf(p.grad)) for p in multimodal_model.parameters() if p.grad is not None):
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    logging.warning("Skipping optimizer step due to NaN/Inf gradients")

                # Accumulate loss
                total_loss += loss.item()

                # Get predicted class
                _, predicted = torch.max(output, 1)

                # Count correct predictions
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                #Log training loss
                sum_writer.add_scalar("Loss/train", loss, i)

                # Print current batch stats
                logging.info(
                                f"[Epoch {epoch} | Batch {i}] Loss: {loss.item():.4f}, "
                                f"KL: {scaled_kl.item():.4f}, Accuracy: {correct / total:.4f}"
                            )

            # Compute training metrics for this epoch
            train_accuracy = correct / total
            train_loss = total_loss / total
            lr = optimizer.param_groups[0]['lr']  # Get current learning rate

            logging.info(
                        f"Epoch {epoch + 1} complete. "
                        f"Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, LR: {lr:.6f}"
                    )
            # Extract patch sizes from type strings for logging
            sss_patch_size = sss_patch_type.replace("patch_", "").replace("_sss", "") if sss_patch_type else "none"
            channel_patch_size = channel_patch_type.replace("patch_", "").replace("_channel", "") if channel_patch_type else "none"

            # Write training results to CSV
            csv_writer.writerow([
                epoch, model_type, train_loss, train_accuracy, lr,
                scaled_kl.item(), cross_entropy_loss.item(),
                sss_patch_size, channel_patch_size
            ])

        # Save model checkpoint every 5 epochs
        if epoch % 5 ==0:
            save_model(multimodal_model, csv_path, f"{model_type}_channel_patch{channel_patch_size}_sss_patch{sss_patch_size}")

        # Free unused GPU memory
        torch.cuda.empty_cache()
    except:
          # Extract patch sizes from type strings for logging
            sss_patch_size = sss_patch_type.replace("patch_", "").replace("_sss", "") if sss_patch_type else "none"
            channel_patch_size = channel_patch_type.replace("patch_", "").replace("_channel", "") if channel_patch_type else "none"
            save_model(multimodal_model, csv_path, f"{model_type}_channel_patch{channel_patch_size}_sss_patch{sss_patch_size}")
            logging.error(f"Error at epoch {epoch}", exc_info=True)
            train_loss, train_accuracy = 0.0, 0.0
    # Return metrics
    return train_loss, train_accuracy

def evaluate_multimodal_model(
    multimodal_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    total_num_epochs: int,
    num_mc: int,
    model_type: str,
    channel_patch_type: Optional[str] = None,
    sss_patch_type: Optional[str] = None,
    csv_path: Optional[str] = ""
):
    """
    Evaluate the multimodal model using MC dropout for uncertainty estimation.

    Args:
        multimodal_model: The model.
        dataloader: Evaluation DataLoader.
        device: CPU or GPU.
        epoch: Current epoch.
        total_num_epochs: Used for scaling KL.
        num_mc: Number of MC samples.
        model_type: Descriptive name of the model.
        channel_patch_type: Patch key for channel image.
        sss_patch_type: Patch key for SSS image.
        csv_path: Output CSV file for metrics.
    """

    multimodal_model.train()  # Keep dropout active

    file_exists = os.path.isfile(csv_path)
    try: # Outer try
        with open(csv_path, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            if not file_exists:
                csv_writer.writerow([
                    "Epoch", "Model Type", "Test Loss", "Test Accuracy",
                    "Predictive Uncertainty", "Model Uncertainty",
                    "Scaled KL", "Cross Entropy Loss",
                    "Channel Patch Type", "SSS Patch Type"
                ])

            criterion = nn.CrossEntropyLoss()
            total_loss = 0
            correct = 0
            total = 0
            all_predicted = []
            all_labels = []
            all_predictive_uncertainty = []
            all_model_uncertainty = []

            kl_weight = (2 ** (epoch + 1)) / (2 ** total_num_epochs)
            epsilon = 1e-8  # for numerical stability

            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    logging.info(f"Eval, batch: {i+1}/{len(dataloader)}, model: {model_type}")

                    inputs = batch["main_image"].to(device)
                    labels = batch["label"].long().to(device)
                    channels_tensor = batch["channel_image"].to(device)
                    sss_image = batch["sss_image"].to(device)

                    patch_channels = {k: v.to(device) for k, v in batch.get("patch_channels", {}).items()}
                    patch_sss = {k: v.to(device) for k, v in batch.get("patch_sss", {}).items()}
                    patch_channels["patch_30_channel"] = channels_tensor
                    patch_sss["patch_30_sss"] = sss_image

                    channel_patch = patch_channels.get(channel_patch_type, channels_tensor)
                    sss_patch = patch_sss.get(sss_patch_type, sss_image)

                    outputs_mc = []
                    softmax_outputs_mc = []
                    kl_mc = []

                    for _ in range(num_mc):
                        outputs = multimodal_model(inputs, channel_patch, sss_patch)
                        outputs_mc.append(outputs)
                        softmax_outputs_mc.append(F.softmax(outputs, dim=1))
                        kl = get_kl_loss(multimodal_model)
                        kl_mc.append(kl)

                    outputs_stack = torch.stack(outputs_mc)  # [num_mc, batch, classes]
                    softmax_stack = torch.stack(softmax_outputs_mc)  # [num_mc, batch, classes]
                    output_mean = torch.mean(outputs_stack, dim=0) # This will be (batch, 1, classes)
                    if output_mean.ndim == 3 and output_mean.size(1) == 1:
                        output_mean = output_mean.squeeze(1) # Corrects (batch, 1, classes) to (batch, classes)
                
                    kl_mean = torch.mean(torch.stack(kl_mc), dim=0) / len(dataloader)
                    kl_scaled = kl_mean * kl_weight

                    cross_entropy_loss = criterion(output_mean, labels)
                    loss = cross_entropy_loss + kl_scaled
                    total_loss += loss.item()

                    _, predicted = torch.max(output_mean, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                    # Uncertainty calculations
                    mean_softmax = torch.mean(softmax_stack, dim=0)
                    predictive_uncertainty_batch = -torch.sum(mean_softmax * torch.log(mean_softmax + epsilon), dim=1)

                    entropy_per_mc_sample = -torch.sum(softmax_stack * torch.log(softmax_stack + epsilon), dim=2)
                    aleatoric_uncertainty_batch = torch.mean(entropy_per_mc_sample, dim=0)
                    model_uncertainty_batch = predictive_uncertainty_batch - aleatoric_uncertainty_batch

                    all_predictive_uncertainty.extend(predictive_uncertainty_batch.cpu().detach().numpy())
                    all_model_uncertainty.extend(model_uncertainty_batch.cpu().numpy())
                    all_predicted.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            test_accuracy = correct / total
            test_loss = total_loss / len(dataloader)
            predictive_uncertainty_mean = np.mean(all_predictive_uncertainty)
            model_uncertainty_mean = np.mean(all_model_uncertainty)
            fig = None # Initialize fig to None outside the inner try
            try: # Inner try for plotting
                # Confusion Matrix
                cm = confusion_matrix(all_labels, all_predicted) # Consider adding labels=list(range(num_classes))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                fig, ax = plt.subplots(figsize=(8, 8))
                disp.plot(cmap="Blues", ax=ax)
                plt.title(f"Confusion Matrix for Epoch {epoch}")

                parent_path = os.path.dirname(csv_path)
                matrix_filename = f"conf_matrix_model_{model_type}_chan_{channel_patch_type or '30'}_sss_{sss_patch_type or '30'}.png"
                plt.savefig(os.path.join(parent_path, matrix_filename))
                logging.info(f"Confusion matrix saved to: {matrix_filename}")
            except Exception as e:
                logging.warning(f"Confusion matrix not saved due to plotting error: {e}", exc_info=True)
            finally: # Ensures fig is closed regardless of success or failure
                if fig is not None:
                    plt.close(fig)

            # Log to CSV - this will now be reached even if plotting fails
            csv_writer.writerow([
                epoch + 1,
                model_type,
                test_loss,
                test_accuracy,
                predictive_uncertainty_mean,
                model_uncertainty_mean,
                kl_scaled.item(),
                cross_entropy_loss.item(),
                channel_patch_type or "patch_30_channel",
                sss_patch_type or "patch_30_sss"
            ])
            logging.info(f"Epoch {epoch + 1}: Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
                         f"Total Uncertainty: {predictive_uncertainty_mean:.4f}, "
                         f"Epistemic: {model_uncertainty_mean:.4f}")

    except Exception as e: # Catch all other exceptions in the outer block
        logging.error(f"Critical error at epoch {epoch}: {e}", exc_info=True)
        test_accuracy = 0.0
    return test_accuracy