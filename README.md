# Project Overview: Multimodal AUV Bayesian Neural Networks for Underwater Environmental Understanding
This project develops and deploys **multimodal, Bayesian Neural Networks (BNNs)**, to process and interpret
habitat data collected by **Autonomous Underwater Vehicles (AUVS)**. This is to offer **scalable**, **accurate** mapping solutions
in complex underwater environments, incorporating unceratinty quantification to allow **reliable** decision making. The package
also presents a model as a retrainable foundation model for further tweaking to new datasets and scenarios.


## Problem Addressed
Underwater data is complex and challenging due to varying lighting, turbidity and sensor limitations. By utilising 
Multimodal Bayeisan Neural Networks, this project explicitly models these uncertainties as well as providing robust
predictions through combining data sources. 


      
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
# Features
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
2.Predict Benthic Habitat Class using a Pre-trained Model
3. Retrain a Pre-trained Model on a New Dataset
4. Train a New Multimodal Model from Scratch
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

**1. Multimodal Fusion Architecture:**
The primary model (`multimodal.py` in `train/` and `base_models.py`) is designed to integrate information from different sensor modalities:

* **Image Encoder:** A Convolutional Neural Network (CNN) backbone (e.g., a pre-trained ResNet, specifically adapted to be Bayesian) processes the optical imagery from AUVs.
* **Sonar Encoder(s):** Separate CNN backbones process the structured sonar data (bathymetry, side-scan sonar). These are adapted to handle the specific characteristics of sonar grids (e.g., single-channel or multi-channel inputs derived from `image_processing.py`).
* **Fusion Layer:** Features extracted from each modality's encoder are concatenated or combined using a dedicated fusion layer (e.g., a fully connected network, attention mechanism). This layer learns the optimal way to combine visual and acoustic information.
* **Prediction Head:** A final set of layers (often fully connected) takes the fused features and outputs predictions for the target task (e.g., benthic habitat classification), with the Bayesian nature providing a distribution over these predictions.

**2. Bayesian Neural Network Implementation:**
The "Bayesian" aspect is achieved by converting deterministic layers (e.g., Linear, Conv2D) into their probabilistic counterparts using libraries like `bayesian-torch`. This means:

* **Weight Distributions:** Instead of learning fixed weights, the model learns distributions over its weights, allowing it to output a distribution of predictions for a given input.
* **Uncertainty Quantification:** The variance in these output predictions provides a direct measure of the model's confidence and epistemic uncertainty, which is vital for decision-making in ambiguous underwater settings.

**3. Foundation Model Concept:**
The project aims to provide a **retrainable foundation model**. This implies:
* The architecture is general enough to be applicable across various underwater mapping tasks.
* It is pre-trained on a diverse dataset (e.g., Northern Britain benthic habitat data), providing strong initial feature representations.
* Users can then fine-tune this pre-trained model (`multimodal.py` in `train/`) on their own smaller, specific datasets to adapt it to new areas or slightly different classification schemes, significantly reducing training time and data requirements.

**4. Unimodal Models (Optional):**
The project also includes components (`unitmodal.py` in `train/` and potentially `base_models.py`) to train and evaluate models based on single modalities (e.g., image-only or sonar-only). This allows for ablation studies and comparison with the performance benefits of multimodal fusion.

**Visual Representation:**
*(Consider adding a high-level diagram here, illustrating the flow of data through separate encoders, the fusion point, and the Bayesian prediction head. Tools like draw.io or Lucidchart can help create simple block diagrams.)*

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
