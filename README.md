# Project Overview: Multimodal AUV Bayesian Neural Networks for Underwater Environmental Understanding
This project develops and deploys **multimodal, Bayesian Neural Networks (BNNs)**, to process and interpret
habitat data collected by **Autonomous Underwater Vehicles (AUVs)**. This is to offer **scalable**, **accurate** mapping solutions
in complex underwater environments, incorporating unceratinty quantification to allow **reliable** decision making. The package
also presents a model as a retrainable foundation model for further tweaking to new datasets and scenarios.


## Problem Addressed
Accurate and scalable environmental mapping within complex underwater environments presents significant challenges due to inherent data complexities and sensor limitations. Traditional methodologies often struggle to account for the variable conditions encountered in marine settings, such as attenuation of light, turbidity, and the physical constraints of acoustic and optical sensors. These factors contribute to noisy, incomplete, and uncertain data acquisition, hindering the generation of reliable environmental characterizations.

Furthermore, conventional machine learning models typically yield point predictions without quantifying associated uncertainties. In applications requiring high-stakes decision-making, such as marine conservation, resource management, or autonomous navigation, understanding the confidence bounds of predictions is critical for robust risk assessment and operational planning. The fusion of diverse data modalities collected by Autonomous Underwater Vehicles (AUVs)—including high-resolution multibeam sonar, side-scan sonar, and optical imagery—further compounds the challenge, necessitating advanced computational approaches to effectively integrate and interpret these disparate information streams.

This project addresses these critical limitations by developing and deploying multimodal Bayesian Neural Networks (BNNs). This approach explicitly models the epistemic and aleatoric uncertainties inherent in complex underwater datasets, providing not only robust environmental classifications but also quantifiable measures of prediction confidence. By leveraging the complementary strengths of multiple sensor modalities, the framework aims to deliver enhanced accuracy, scalability, and decision-making capabilities for comprehensive underwater environmental understanding.


      
# Project Structure
```
Multimodal_AUV/
├── Multimodal_AUV/
│   ├── config/
│     ├── paths.py
│     └── __init__.py
│   ├── data/
│     ├── datasets.py
│     ├── loaders.py
│     └── __init__.py
│   ├── data_preperation/
│     ├── geospatial.py
│     ├── image_processing.py
│     ├── main_data_preparation.py
│     ├── sonar_cutting.py
│     ├── utilities.py
│     └── __init__.py
│   ├── Examples/
│     ├── Example_data_preparation.py
│     ├── Example_Inference_model.py
│     ├── Example_Retraining_model.py
│     ├── Example_training_from_scratch.py
│     └── __init__.py
│   ├── inference/
│     ├── inference_data.py
│     ├── predictiors.py
│     └── __init__.py
│   ├── models/
│     ├── base_models.py
│     ├── model_utils.py
│     └── __init__.py
│   ├── train/
│     ├── checkpointing.py
│     ├── loop_utils.py
│     ├── multimodal.py
│     ├── unitmodal.py
│     └── __init__.py
│   ├── utils/
│     ├── device.py
│     └── __init__.py
│   ├── main.py
│   └── __init__.py
└── unittests/
      ├── test_data.py
      ├── test_model.py
      ├── test_train.py
      ├── test_utils.py
      └── __init__.py
```
# Repo features
Here are some of the key capabilities of this GITHUB
* **Start to finish pipeline**: taking georeferenced imagery and sonar tiffs to then train Bayesian Neural Networks 
and make predictions from this.

* **Model to predict benthic habitat class (Northern Britain)**: Can download and run a model to evaluate bathymetric, sidescan and image "pairs"
and predict if its Sand, Mud, Rock, Gravel, Burrowed Mud (PMF), Kelp forest (PMF) or Horse Mussel reef (PMF).
 

* **Retrainable model**: Code to download and retrain a pretrained network for combining bathymetric, sidescan sonar and image
for a new dataset.

* **Training a model from scratch**: Code to take sonar and image and train a completely new model returning a csv of metrics,
the model and confusion matricies.

* **Options to optimise sonar patch sizes and to train unimodal models**: Code to find the optimal sonar patch to maximise predicitve accuracy (high compute requirements).
And to train unimodal and multimodal models to compare the beneft of multimodality.
 
# Getting started (requirements install)
1. Clone repository
2. Create a virtual environment
3. Install Dependencies
4. Prepare Data Folders (Think about what it needs to be and clarify this)
   
# Usage examples
1.Run the End-to-End Data Preparation Pipeline
## 2.Predict Benthic Habitat Class using a Pre-trained Model
Once you have your environment set up and data prepared, you can run inference using our pre-trained Multimodal AUV Bayesian Neural Network. This example demonstrates how to apply the model to new data and generate predictions with uncertainty quantification.

Prerequisites:

Ensure you have cloned this repository and installed all dependencies as described in the Installation Guide.

Your input data (images and sonar files) should be organized as expected by the CustomImageDataset (refer to Multimodal_AUV/data/datasets.py for details). The --data_dir argument should point to the root of this organized dataset.

The script will automatically download the required model weights from the Hugging Face Hub.

Inference Command Example:

To run inference on your multimodal AUV data and save the predictions to a CSV file, use the following command from the root directory of this repository:
```
Bash

python -m Multimodal_AUV.Examples.Example_Inference_model \
    --data_dir "/path/to/your/input_data/dataset" \
    --output_csv "/path/to/save/your/results/inference.csv" \
    --batch_size 4 \
    --num_mc_samples 20

```
Understanding the Arguments:

python -m Multimodal_AUV.Examples.Example_Inference_model: This executes the Example_Inference_model.py script as a Python module, which is the recommended way to run scripts within a package structure.

--data_dir "/path/to/your/input_data/dataset":

Purpose: Specifies the absolute path to the directory containing your multimodal input data (e.g., GeoTIFFs, corresponding CSVs, etc.).

Action Required: You MUST replace "/path/to/your/input_data/all_mulroy_images_and_sonar" with the actual absolute path to your dataset on your local machine.

--output_csv "/path/to/save/your/results/inference.csv":

Purpose: Defines the absolute path and filename where the inference results (predicted classes, uncertainty metrics) will be saved in CSV format.

Action Required: You MUST replace "/path/to/save/your/results/inference.csv" with your desired output path and filename. The script will create the file and any necessary parent directories if they don't exist.

--batch_size 4:

Purpose: Sets the number of samples processed at once by the model during inference.

Customization: Adjust this value based on your available GPU memory. Larger batch sizes can speed up inference but require more VRAM.

--num_mc_samples 5:

Purpose: Specifies the number of Monte Carlo (MC) samples to draw from the Bayesian Neural Network's posterior distribution. A higher number of samples leads to a more robust estimation of predictive uncertainty.

Customization: For production, you might use 100 or more samples for better uncertainty estimation. For quick testing, 5-10 samples are sufficient.

Expected Output:

Upon successful execution, a CSV file (e.g., inference.csv) will be created at the specified --output_csv path. This file will contain:

Image Name: Identifier for the input sample.

Predicted Class: The model's most likely class prediction.

Predictive Uncertainty: A measure of the total uncertainty in the prediction (combining aleatoric and epistemic).

Aleatoric Uncertainty: Uncertainty inherent in the data itself (e.g., sensor noise, ambiguous regions).

Here's how you can document your new functions in your GitHub README, following the style and detail of your existing sections:

## 3. Retrain a Pre-trained Model on a New Dataset

This example demonstrates how to fine-tune our pre-trained Multimodal AUV Bayesian Neural Network on your own custom dataset. Retraining allows you to adapt the model to specific environmental conditions or new benthic classes present in your data, leveraging the knowledge already learned by the pre-trained model.

Prerequisites:

Ensure you have cloned this repository and installed all dependencies as described in the Installation Guide.

Your input data (images and sonar files) should be organized as expected by the CustomImageDataset (refer to Multimodal_AUV/data/datasets.py for details). The --data_dir argument should point to the root of this organized dataset.

The script will automatically download the required pre-trained model weights from the Hugging Face Hub.

Retraining Command Example:

To retrain the model on your multimodal AUV data, use the following command from the root directory of this repository:

```
Bash

python -m Multimodal_AUV.Examples.Example_Retraining_model \
    --data_dir "/path/to/your/input_data/dataset" \
    --batch_size_multimodal 20 \
    --num_epochs_multimodal 20 \
    --num_mc_samples 20 \
    --learning_rate_multimodal 0.001 \
    --weight_decay_multimodal 1e-5 \
    --bathy_patch_base 30 \
    --sss_patch_base 30

```
Understanding the Arguments:

python -m Multimodal_AUV.Examples.Example_Retraining_model: This executes the Example_Retraining_model.py script as a Python module, which is the recommended way to run scripts within a package structure.

--data_dir ""/path/to/your/input_data/dataset"":

Purpose: Specifies the absolute path to the directory containing your multimodal input data for retraining (e.g., GeoTIFFs, corresponding CSVs, etc.).

Action Required: You MUST replace ""/path/to/your/input_data/dataset"" with the actual absolute path to your dataset on your local machine.

--batch_size_multimodal 20:

Purpose: Sets the number of samples processed at once by the model during retraining.

Customization: Adjust this value based on your available GPU memory. Larger batch sizes can speed up training but require more VRAM.

--num_epochs_multimodal 20:

Purpose: Defines the total number of training epochs (complete passes through the entire dataset).

Customization: Increase this value for more thorough training, especially with larger datasets or when the model is converging slowly.

--num_mc_samples 20:

Purpose: Specifies the number of Monte Carlo (MC) samples to draw from the Bayesian Neural Network's posterior distribution during training. A higher number of samples leads to a more robust estimation of predictive uncertainty.

Customization: For production, you might use 100 or more samples for better uncertainty estimation. For quicker testing or initial training, 5-10 samples are sufficient.

--learning_rate_multimodal 0.001:

Purpose: Sets the initial learning rate for the optimizer. This controls the step size at which the model's weights are updated during training.

Customization: Experiment with different learning rates (e.g., 0.01, 0.0001) to find the optimal value for your dataset.

--weight_decay_multimodal 1e-5:

Purpose: Applies L2 regularization (weight decay) to prevent overfitting by penalizing large weights.

Customization: Adjust this value to control the strength of the regularization. A higher value means stronger regularization.

--bathy_patch_base 30:

Purpose: Defines the base patch size for bathymetry data processing.

Customization: This parameter affects how bathymetry data is chunked and processed. Adjust as needed based on your data characteristics.

--sss_patch_base 30:

Purpose: Defines the base patch size for side-scan sonar (SSS) data processing.

Customization: Similar to bathy_patch_base, this affects how SSS data is chunked and processed.

## 4. Train a New Multimodal Model from Scratch
This example outlines how to train a new Multimodal AUV Bayesian Neural Network entirely from scratch using your own dataset. This is suitable when you have a large, diverse dataset and want to build a model specifically tailored to your data's unique characteristics, without relying on pre-trained weights.

Prerequisites:

Ensure you have cloned this repository and installed all dependencies as described in the Installation Guide.

Your input data (images and sonar files) should be organized as expected by the CustomImageDataset (refer to Multimodal_AUV/data/datasets.py for details). The --root_dir argument should point to the root of this organized dataset.

Training Command Example:

To train a new model from scratch on your multimodal AUV data, use the following command from the root directory of this repository:

```
Bash

python -m Multimodal_AUV.Examples.Example_training_from_scratch \
    --root_dir "/path/to/your/input_data/dataset" \
    --epochs_multimodal 20 \
    --num_mc 20 \
    --batch_size_multimodal 20 \
    --lr_multimodal 0.001
```

Understanding the Arguments:

python -m Multimodal_AUV.Examples.Example_training_from_scratch: This executes the Example_training_from_scratch.py script as a Python module, which is the recommended way to run scripts within a package structure.

--root_dir "/path/to/your/input_data/dataset":

Purpose: Specifies the absolute path to the root directory containing your multimodal input data for training (e.g., GeoTIFFs, corresponding CSVs, etc.).

Action Required: You MUST replace /home/tommorgan/Documents/data/representative_sediment_sample/ with the actual absolute path to your dataset on your local machine.

--epochs_multimodal 20:

Purpose: Defines the total number of training epochs (complete passes through the entire dataset).

Customization: Increase this value for more thorough training, especially with larger datasets. Training from scratch typically requires more epochs than retraining.

--num_mc 20:

Purpose: Specifies the number of Monte Carlo (MC) samples to draw from the Bayesian Neural Network's posterior distribution during training. A higher number of samples leads to a more robust estimation of predictive uncertainty.

Customization: For production, you might use 100 or more samples for better uncertainty estimation. For quicker testing or initial training, 5-10 samples are sufficient.

--batch_size_multimodal 20:

Purpose: Sets the number of samples processed at once by the model during training.

Customization: Adjust this value based on your available GPU memory. Larger batch sizes can speed up training but require more VRAM.

--lr_multimodal 0.001:

Purpose: Sets the initial learning rate for the optimizer. This controls the step size at which the model's weights are updated during training.

Customization: Experiment with different learning rates (e.g., 0.01, 0.0001) to find the optimal value for your dataset. Training from scratch might require more careful tuning of the learning rate.








# Configuration REWRITE ALL THIS

The project's behavior and parameters can be easily adjusted through configuration files, allowing you to adapt the pipeline and models to different datasets, training regimes, or specific requirements without modifying the source code.

**1. Main Configuration File (Recommended):**
Most overarching parameters for data preparation, training, and inference are controlled via a central configuration file. We recommend using a `.yaml` file for this purpose, located in the `Multimodal_AUV/config/` directory. This file would typically define:

* **Data Paths:** Input CSVs, GeoTIFF folders, output directories for processed data.
* **Training Parameters:** Learning rate, batch size, number of epochs, optimizers, loss functions.
* **Model Parameters:** Specific architecture choices (e.g., backbone, fusion layer dimensions), BNN prior settings.
* **Experiment Settings:** Logging paths, checkpointing frequency, evaluation metrics.

**Example (Multimodal_AUV/config/default_config.yaml):**
```yaml
# Data paths
input_csv: "path/to/your/input_data/combined_csv.csv"
geotiff_dir: "path/to/your/input_data/Irish sonar/"
output_processed_dir: "path/to/your/processed_output/"

# Data preparation
patch_size_meters: 20
geotiff_channels: ["Bathy", "SSS"]
image_dimensions: [224, 224]

# Model training
model_type: "multimodal_bnn" # or "unimodal_bnn"
num_classes: 6               # Sand, Mud, Rock, Gravel, Burrowed Mud, Kelp forest, Horse Mussel reef
batch_size: 32
learning_rate: 0.001
epochs: 50
early_stopping_patience: 10
optimizer: "Adam"
loss_function: "CrossEntropyLoss"
device: "cuda" # or "cpu"

# BNN Specifics (if applicable)
prior_mu: 0.0
prior_sigma: 1.0

# Inference
prediction_threshold: 0.5
output_prediction_dir: "path/to/save/inference_results/"
```
# Model architecture

This project leverages sophisticated **Bayesian Neural Network (BNN)** architectures designed for robust multimodal data fusion and uncertainty quantification in underwater environments. The core design principles are modularity and adaptability, allowing for both unimodal and multimodal processing.

## **1. Multimodal Fusion Architecture:**
The primary model (`multimodal.py` in `train/` and `base_models.py`) is designed to integrate information from different sensor modalities:
* **Image Encoder:** A Convolutional Neural Network (CNN) backbone (e.g., a pre-trained ResNet, specifically adapted to be Bayesian) processes the optical imagery from AUVs.
* **Sonar Encoder(s):** Separate CNN backbones process the structured sonar data (bathymetry, side-scan sonar). These are adapted to handle the specific characteristics of sonar grids (e.g., single-channel or multi-channel inputs derived from `image_processing.py`).
* **Fusion Layer:** Features extracted from each modality's encoder are concatenated or combined using a dedicated fusion layer (e.g., a fully connected network, attention mechanism). This layer learns the optimal way to combine visual and acoustic information.
* **Prediction Head:** A final set of layers (often fully connected) takes the fused features and outputs predictions for the target task (e.g., benthic habitat classification), with the Bayesian nature providing a distribution over these predictions.

### Visual representation
![image](https://github.com/user-attachments/assets/3d799daf-b876-45a6-bc93-5837dd7bd80f)

**2. Bayesian Neural Network Implementation:**
The "Bayesian" aspect is achieved by converting deterministic layers (e.g., Linear, Conv2D) into their probabilistic counterparts using `bayesian-torch`. This means:

* **Weight Distributions:** Instead of learning fixed weights, the model learns distributions over its weights, allowing it to output a distribution of predictions for a given input.
* **Uncertainty Quantification:** The variance in these output predictions provides a direct measure of the model's confidence and epistemic uncertainty, which is vital for decision-making in ambiguous underwater settings.

**3. Foundation Model Concept:**
The project aims to provide a **retrainable foundation model**. This implies:
* The architecture is general enough to be applicable across various underwater mapping tasks.
* It is pre-trained on a diverse dataset (e.g., Northern Britain benthic habitat data), providing strong initial feature representations.
* Users can then fine-tune this pre-trained model (`multimodal.py` in `train/`) on their own smaller, specific datasets to adapt it to new areas or slightly different classification schemes, significantly reducing training time and data requirements.

**4. Unimodal Models:**
The project also includes components (`unitmodal.py` in `train/` and potentially `base_models.py`) to train and evaluate models based on single modalities (e.g., image-only or sonar-only). This allows for ablation studies and comparison with the performance benefits of multimodal fusion.

### Visual representation
![image](https://github.com/user-attachments/assets/14dbb63f-864d-4fca-9bb5-fd26b91ea827)


---
# Contact
Have questions about the project, found a bug, or want to contribute? Here are a few ways to reach out:

* **GitHub Issues:** For any code-related questions, bug reports, or feature requests, please open an [Issue on this repository](https://github.com/sams-tom/multimodal-auv-bnn-project/issues). This is the preferred method for transparency and tracking.

* **Email:** For more direct or confidential inquiries, you can reach me at [phd01tm@sams.ac.uk](mailto:phd01tm@sams.ac.uk).

* **LinkedIn (Optional):** Connect with the project lead/team on LinkedIn:
    * [Tom Morgan](https://www.linkedin.com/in/tom-morgan-8a73b129b/)
      
# Citations

* **GitHub Repository (Code & Documentation):** [https://github.com/sams-tom/multimodal-auv-bnn-project](https://github.com/sams-tom/multimodal-auv-bnn-project)
* **Hugging Face Models:** [https://huggingface.co/sams-tom/multimodal-auv-bnn-models](https://huggingface.co/sams-tom/multimodal-auv-bnn-models)
* **Research Paper:** [In development]
