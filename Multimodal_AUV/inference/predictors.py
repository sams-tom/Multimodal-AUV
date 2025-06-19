import logging
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

def multimodal_predict_and_save(multimodal_model: nn.Module, dataloader: DataLoader, device: torch.device, csv_path: str, num_mc_samples: int=10, sss_patch_type: Optional[str]="", channel_patch_type: Optional[str]="", model_type: str="multimodal"):
    """
    Perform prediction with Monte Carlo sampling on a multimodal model to estimate predictive
    and aleatoric uncertainty, then save predictions and uncertainties to a CSV.

    Args:
        multimodal_model (nn.Module): The multimodal model for prediction.
        dataloader (DataLoader): DataLoader with batches of input data and image names.
        device (torch.device): Device to run the model on.
        csv_path (str): File path where prediction results will be saved.
        num_mc_samples (int, optional): Number of Monte Carlo samples for uncertainty estimation.
        sss_patch_type (str, optional): SSS patch type description for logging (default empty).
        channel_patch_type (str, optional): Channel patch type description for logging (default empty).
        model_type (str, optional): Model type label for logging (default "multimodal").

    Returns:
        None: Results are saved to CSV file and printed to console.
    """
    multimodal_model.train()  # Set the model to training mode to activate dropout layers for Monte Carlo (MC) sampling
    logging.info(f"CSV will be saved to: {csv_path}")
    with open(csv_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
    
            # Define and write the header to the CSV file to clarify the meaning of each column
            header = ["Image Name", "Predicted Class", "Predictive Uncertainty", "Aleatoric Uncertainty"]
            csv_writer.writerow(header)
            logging.info(f"CSV Header written: {header}")
            logging.info(f"Length of the dataloader: {len(dataloader)}") 

            with torch.no_grad():  # Disable gradient computation to reduce memory usage and improve inference speed
                for batch_idx, (inputs, patch_30_channel, patch_30_sss, image_name) in enumerate(dataloader):
                    logging.info(f"\n--- Processing Batch {batch_idx + 1} ---")


                    # Move all inputs to the target device (CPU/GPU), maintaining consistency across computation
                    inputs = inputs.to(device)
                    channels_tensor = patch_30_channel.to(device)
                    sss_image = patch_30_sss.to(device)

                    # Log tensor shapes to confirm batch and input dimensionality align with model expectations
                    logging.debug(f"Input shape: {inputs.shape}, Channel shape: {channels_tensor.shape}, SSS shape: {sss_image.shape}")
                    logging.debug(f"Input device: {inputs.device}, Channel device: {channels_tensor.device}, SSS device: {sss_image.device}")

                    softmax_outputs_mc = []  # Store probability outputs from repeated stochastic forward passes

                    for mc_sample in range(num_mc_samples):  # Perform MC Dropout sampling for uncertainty estimation
                        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                            if isinstance(multimodal_model, torch.nn.parallel.DistributedDataParallel):
                                # Special handling if model is wrapped for distributed training
                                outputs = multimodal_model.module(inputs, channels_tensor, sss_image)
                                logging.debug(f"MC Sample {mc_sample + 1} - Output shape: {outputs.shape}")
                            else:
                                outputs = multimodal_model(inputs, channels_tensor, sss_image)
                                logging.debug(f"MC Sample {mc_sample + 1} - Output shape: {outputs.shape}")

                            # Convert raw model outputs to probabilities
                            softmax_outputs = F.softmax(outputs, dim=1)
                            softmax_outputs_mc.append(softmax_outputs)

                    # Combine the softmax outputs from all MC forward passes into one tensor
                    prob_outputs_mc = torch.stack(softmax_outputs_mc, dim=0)
                    print(f"Stacked MC probability outputs shape: {prob_outputs_mc.shape}")

                    # Calculate the model's epistemic uncertainty (how much predictions vary between passes)
                    predictive_uncertainty = torch.var(prob_outputs_mc, dim=0).mean(dim=1)
                    print(f"Predictive Uncertainty: {predictive_uncertainty.cpu().numpy()}")

                    # Estimate aleatoric uncertainty by averaging entropy across MC samples (captures data noise)
                    epsilon = 1e-7  # Prevents log(0) in entropy calculation
                    entropy_per_mc = -torch.sum(prob_outputs_mc * torch.log(prob_outputs_mc + epsilon), dim=-1)
                    aleatoric_uncertainty = torch.mean(entropy_per_mc, dim=0)
                    print(f"Aleatoric Uncertainty: {aleatoric_uncertainty.cpu().numpy()}")

                    # Use the average class probabilities over MC samples to derive final class predictions
                    mean_prob = torch.mean(prob_outputs_mc, dim=0)
                    predicted_class = torch.argmax(mean_prob, dim=1)
                    print(f"Predicted Classes: {predicted_class.cpu().numpy()}")
                    # Loop through each item in the batch and write its results to the CSV file
                    for i in range(inputs.size(0)):
                        current_image_name = image_name[i] if isinstance(image_name, (list, tuple)) else image_name
                        row_data = [
                            current_image_name,
                            predicted_class[i].item(),
                            predictive_uncertainty[i].item(),
                            aleatoric_uncertainty[i].item()
                        ]
                        csv_writer.writerow(row_data)
                        logging.debug(f"Row saved: {row_data}")
    logging.info("Completed: multimodal_predict_and_save")
