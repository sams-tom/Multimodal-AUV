# 🌊Project Overview: Multimodal AUV Bayesian Neural Networks for Underwater Environmental Understanding🐠
This project develops and deploys **multimodal, Bayesian Neural Networks (BNNs)**, to process and interpret habitat data collected by **Autonomous Underwater Vehicles (AUVs)**. This is to offer **scalable**, **accurate** mapping solutions
in complex underwater environments,whilst incorporating unceratinty quantification to allow **reliable** decision making. The repo 
also presents a model as a retrainable foundation model for further tweaking to new datasets and scenarios.🚀


 ## 🚧 Problem Addressed 🚧
**Environmental mapping** within complex underwater environments presents significant challenges due to inherent data complexities and sensor limitations. Traditional methodologies often struggle to account for the variable conditions encountered in marine settings, such as attenuation of **light 🔦, turbidity  🌊, and the physical constraints of acoustic and optical sensors 📸** . These factors contribute to **noisy, incomplete, and uncertain data acquisition**, hindering the generation of reliable environmental characterizations.📉

Furthermore, conventional machine learning models typically yield point predictions without quantifying associated uncertainties. In applications requiring high-stakes decision-making, such as **marine conservation🌿, resource management 🐠, or autonomous navigation 🧭**, understanding the **confidence bounds** of predictions is critical for robust risk assessment and operational planning. The fusion of diverse data modalities collected by Autonomous Underwater Vehicles (AUVs), including high-resolution **multibeam sonar 📡, side-scan sonar 🛰️, and optical imagery 📷**, further compounds the challenge, necessitating advanced computational approaches to effectively integrate and interpret these disparate information streams.

This project addresses these critical limitations by developing and deploying **multimodal Bayesian Neural Networks (BNNs)**. This approach explicitly models and quantifies the **epistemic and aleatoric uncertainties** inherent in complex underwater datasets, providing not only robust environmental classifications but also **quantifiable measures of prediction confidence**. By leveraging the **complementary strengths of multiple sensor modalities**, the framework aims to deliver enhanced accuracy, scalability, and decision-making capabilities for comprehensive underwater environmental understanding. ✨


      
# Project Structure 🏗️
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
# Module features 🚀
Here are some of the key capabilities of this GITHUB
* **End-to-End Pipeline**:The repo offers a complete pipeline, allowing you to turn raw georeferenced imagery📸 and sonar tiffs 📡into **valid predictions with quantified uncertainty** by training Bayesian Neural Networks.

* **Model to predict benthic habitat class (Northern Britain)**: Can download and run a model to evaluate bathymetric, sidescan and image "pairs"
and predict specific  benthic habitat classes found in Northern Britain: **Sand 🏖️, Mud 🏞️, Rock 🪨, Gravel ⚫, Burrowed Mud (PMF) 🕳️, Kelp forest (PMF) 🌳, or Horse Mussel reef (PMF) 🐚**.
 
* **Retrainable foundation model**: Code to download and retrain a **pretrained network** for combining bathymetric, sidescan sonar and image for a new datasets, adapting the model to your specific needs with reduced computational requirements. 🔄

* **Training a model from scratch**: Code to take sonar and image and train a **completely new model** returning a  CSV of metrics 📊, the model itself 🧠, and confusion matrices 📈.

* **Options to optimise sonar patch sizes and to train unimodal models**: Code to find the **optimal sonar patch** to maximise predicitve accuracy (high compute requirements! ⚡) and to train unimodal and multimodal models to **compare the benefits of multimodality**. 🔬
 
# Getting started
This section guides you through setting up the project, installing dependencies, and preparing your data for processing and model training/inference.

   
1. **Create and Activate Conda Environment**:
   We recommend using Conda to manage the project's dependencies for a consistent and isolated     environment.
   
   Create the Conda environment:
   ```
   Bash
   
   conda create -n multimodal_auv python=3.9 # Or your specific Python version (e.g., 3.10,     3.11)
   ```
   Activate the environment:
   ```
   Bash
   
   conda activate multimodal_auv
   ```
   You should see (multimodal_auv) at the beginning of your terminal prompt, indicating the  environment is active.

3. **Install Dependencies**:
   With your Conda environment active, install all necessary Python packages listed in the  requirements.txt file.
   ```
   Bash
   
   pip install Multimodal-AUV
   ```
Important Note on GPU Support:
In order to train quickly this project utilises PyTorch with CUDA for GPU acceleration. However, the requirements.txt file does not includ PyTorch (torch, torchvision, torchaudio) and NVIDIA CUDA runtime dependencies as these need to be downloaded to fit with your local CUDA toolkit or GPU driver setup. Navigate to this webpage: https://pytorch.org/get-started/locally/ select your requirements and then copy the command and run that locally.

For example, for CUDA 11.8, Python on  windows:
```
Bash
# Then, install PyTorch with CUDA via Conda
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```

To import for a script simply call at the top of the script:
```
Bash
import Multimodal_AUV
```

4. **Prepare Data Folders**:
   
   Your project requires specific data structures for input and output. If you run the examples below this will be structored correctly. Please organize your data as follows, and update the paths in your config.yaml file accordingly.

Recommended Folder Structure:
```
Multimodal_AUV/
├── data/
│   ├── individual_data_point/
│   │   ├── auv_image.jpg/  # Image from camera
│   │   ├── local_side_scan_image.jpg/  # Cut out of sonar local to camera
│   │   ├── local_bathy_image.jpg/  # Cut out of sonar local to camera
│   │   └── LABEL.txt/    # Where the Label is in the title replacing LABEL
│   ├── individual_data_point/
│..........
│   └── individual_data_point/
│   ├── processed_output/    # Output folder for processed AUV data (e.g., aligned images, extracted features)
│   ├── model_checkpoints/   # Directory to save trained model weights/checkpoints
│   └── inference_results/   # Directory to save inference output (e.g., prediction CSVs, classified maps)
├── config.yaml              # Your main configuration file
├── Multimodal_AUV/
│   └── ...                  # Your Python source code
├── your_runner_script.py    # (Optional) Script to run commands based on config.yaml
├── requirements.txt         # List of Python dependencies
└── README.md
```
## Clarifying Data Folder Contents:

* ```data/```: Folder containing folders of paired data. Your training scripts' ```--root_dir``` would typically point here.
* ```data/individual_data_point/```: Example of folder within folder holding required data files
* ```data/individual_data_point/auv_image.jpg```: The individual image for prediction
* ```data/individual_data_point/local_side_scan_image.jpg```: The individual side scan image local to the camera image for prediction
* ```data/individual_data_point/local_bathy_image.jpg```: The individual bathymetric image local to the camera image for prediction
* ```data/individual_data_point/LABEL.txt```: The label to predict. **N.B.** Not required if youre not training/retraining a model.
## NOTE : Sidescna files must have SSS in name and bathymetric files must be called "patch_30m_combined_bathy"
## Example root directory
![image](https://github.com/user-attachments/assets/e2e3c0be-e38e-41c5-b21b-affc50149cdc)

## Example interal data directory
![image](https://github.com/user-attachments/assets/c157bcec-c5dc-4f2c-8c36-ddeed2c5a3ce)

## Understanding the arguments
* ```data/processed_output/```: Stores intermediate or final processed data, often generated by preliminary scripts.

* ```data/model_checkpoints/```: Dedicated location for saving trained model weights and checkpoints.

* ```data/inference_results/```: Stores outputs generated by your inference models (e.g., prediction CSVs, classified maps).

### Action Required:

* **Create these directories manually** within your cloned repository if they don't exist. **Note**: If you run the below code including the example of data preparation the correct structure will be created automatically.

* **Update** ```config.yaml```: Open your ```config.yaml``` file and set the ```data_root_dir```, ```output_base_dir```, and other relevant paths within ```training_from_scratch```, ```retraining_model```, ```inference_model```, and ```raw_data_processing``` sections to match the paths you've created.
   
# Usage examples
## 1. Run the End-to-End Data Preparation Pipeline ⚙️
To preprocess your AUV sonar and optical image data, execute the following command from your terminal:

```bash
multimodal-auv-data-prep --raw_optical_images_folder "/home/tommorgan/Documents/data/Newfolder/" --geotiff_folder "/home/tommorgan/Documents/data/Newfolder/sonar/" --output_folder "/home/tommorgan/Documents/data/test/" --window_size_meters 30 --image_enhancement_method "AverageSubtraction"  --exiftool_path '/usr/bin/exiftool'

```
To do this in a script run:
```
Bash
# Example for run_auv_preprocessing
import os
from Multimodal_AUV import run_auv_preprocessing

raw_optical_images_folder = "/your/data/dir/"
geotiff_folder = "/your/data/dir//sonar"
output_folder = "/your/output/folder/processed_auv_data_output"
exiftool_path = "/usr/bin/exiftool" #for linux Or "C:/exiftool/exiftool.exe" for Windows. Must point at your exiftool.exe download

run_auv_preprocessing(
    raw_optical_images_folder=raw_optical_images_folder,
    geotiff_folder=geotiff_folder,
    output_folder=output_folder,
    exiftool_path=exiftool_path,
    window_size_meters=20.0,
    image_enhancement_method="AverageSubtraction",
    skip_bathy_combine=False
)
print("Preprocessing function called.")
```

### Understanding the Arguments:

* **```python Example_data_preparation.py```**: This invokes the main preprocessing script.

* **```--raw_optical_images_folder```**: ```"/path/to/your/raw/optical_images"```
     
     **Purpose**: Specifies the absolute path to the directory containing a collection of folders with your original, unprocessed JPG optical image files from the AUV. This should be as its downloaded from your datasource. The structure should have folders inside (at least one) containing images with metadata accessible by Exiftool and organised in this structure: 
          ```<comment>
              <altitude>1.52</altitude>
              <depth>25.78</depth>
              <heading>123.45</heading>
              <pitch>2.10</pitch>
              <roll>-0.75</roll>
              <surge>0.15</surge>
              <sway>-0.05</sway>
              <lat>56.12345</lat>
              <lon>-3.98765</lon>
          </comment>``` 
     If not you will have to rewrite the metadata part of the function or organise your own data function.
     
     **Action Required**: You MUST replace ```/path/to/your/raw/optical_images``` with the actual, full path to your raw optical images folder on your local machine.

* **```--geotiff_folder```**: ```"/path/to/your/auv_geotiffs"```

  **Purpose**: Defines the absolute path to the directory containing all your GeoTIFF files, which typically include bathymetry and side-scan sonar data. The bathymetry tiffs must have "bathy" in the file name, the side-scan must have "SSS" in the file name. 
  
  **Action Required**: You MUST replace ```/path/to/your/auv_geotiffs``` with the actual, full path to your GeoTIFFs folder.

  Example Structure:
  
  ```/path/to/your/auv_geotiffs/
  ├── bathymetry.tif
  ├── side_scan.tif
  └── ...```

* **```--output_folder```**: ```"/path/to/your/processed_auv_data"```
  
 **Purpose**: Designates the root directory where all the processed and organized output data will be saved. This is where the processed optical images, sonar patches, and the main coords.csv file will reside.
  
  **Action Required**: You MUST replace ```/path/to/your/processed_auv_data``` with your desired output directory.

* **```--exiftool_path```** ```"C:/exiftool/"```

  **Purpose**: Provides the absolute path to the directory where the exiftool.exe executable is located. This is essential for extracting GPS and timestamp information from your optical images.
  
  **Action Required**: You MUST download and unpack exiftool and then replace
```"C:/exiftool/exiftool.exe "``` with the correct path to your ExifTool installation, it MUST point at the .exe itself. For Linux/macOS, this might be /usr/bin/ or /usr/local/bin/ if installed globally.

* **```--window_size_meters 30.0```**

  **Purpose**: Sets the desired side length (in meters) for the square patches that will be extracted from your GeoTIFF files (e.g., a 30.0 value means a 30m x 30m sonar patch).
  
  **Customization**: Adjust this value based on the scale of features you want to capture in your sonar data for machine learning and the typical coverage of your optical images. 30 meters has been found optimal in most scenarios

* **```--image_enhancement_method```** ```"AverageSubtraction"```

  **Purpose**: Specifies the method to be used for enhancing the optical images. This can improve the visual quality and potentially the feature extraction for machine learning.
  
  **Customization**: Choose between "AverageSubtraction" (a simpler method) or "CLAHE" (Contrast Limited Adaptive Histogram Equalization, often more effective for underwater images). The default is AverageSubtraction.

* **```--skip_bathy_combine (Optional flag)```**

  **Purpose**: If this flag is present, the post-processing step that attempts to combine multiple bathymetry channels into a single representation will be skipped.
  
  **Usage**: Include this flag in your command if you do not want this channel combination to occur. For example: python your_script_name.py ... --skip_bathy_combine (no value needed, just the flag).

### Output Data Structure

Upon successful execution, your ```--output_folder``` will contain a structured dataset. Here's an example of the typical output:
   ```
   /path/to/your/processed_auv_data/
   ├── coords.csv
   ├── image_0001/
   │   ├── image_0001_processed.jpg  # Enhanced optical image
   │   ├── bathymetry_patch.tif      # Extracted bathymetry patch
   │   ├── side_scan_patch.tif       # Extracted side-scan sonar patch
   │   └── (other_geotiff_name)_patch.tif
   ├── image_0002/
   │   ├── image_0002_processed.jpg
   │   ├── bathymetry_patch.tif
   │   └── ...
   └── ...
   ```

* **coords.csv**: A primary metadata file containing entries for each processed optical image, including its filename, geographical coordinates (latitude, longitude), timestamp, and the relative path to its corresponding processed image and sonar patches within the output structure.

* **image_XXXX/ subfolders**: Each subfolder is named after the processed optical image and contains the processed optical image itself.

* **GeoTIFF patches** : Individual GeoTIFF files representing the extracted square patches from each of your input GeoTIFFs (e.g., bathymetry, side-scan sonar) for that specific location.


## 2.Predict Benthic Habitat Class using a Pre-trained Model 🐠

Once you have your environment set up and data prepared, you can run inference using our pre-trained Multimodal AUV Bayesian Neural Network (Found here: https://huggingface.co/sams-tom/multimodal-auv-bathy-bnn-classifier/tree/main/multimodal-bnn) . This example demonstrates how to apply the model to new data and generate predictions with uncertainty quantification.

### Prerequisites:

* Ensure you have cloned this repository and installed all dependencies as described in the Installation Guide.

* Your input data (images and sonar files) should be organized as expected by the CustomImageDataset (refer to ```Multimodal_AUV/data/datasets.py``` or the above example (1.) for details). The ```--data_dir``` argument should point to the root of this organized dataset.

* The script will **automatically** download the required model weights from the Hugging Face Hub.

Inference Command Example:

```
Bash

multimodal-auv-inference --data_dir "/home/tommorgan/Documents/data/all_mulroy_images_and_sonar" --output_csv "/home/tommorgan/Documents/data/test/csv.csv" --batch_size 4 --num_mc_samples 10


```
To do this in a script run:
```
Bash
from Multimodal_AUV import run_auv_inference

inference_data_dir = "/your/data/dir"
inference_output_csv = "/output/dir/inference_results.csv"

run_auv_inference(
    data_directory=inference_data_dir,
    batch_size=4,
    output_csv=inference_output_csv,
    num_mc_samples=5,
    num_classes=7 # IMPORTANT: This must equal 7 to fit the downloaded model
)
print("Inference function called. Check results in:", inference_output_csv)
````
### Understanding the Arguments:

* **```python -m Multimodal_AUV.Examples.Example_Inference_model```**: This executes the ```Example_Inference_model.py``` script as a Python module, which is the recommended way to run scripts within a package structure.

* **```--data_dir``` ```"/path/to/your/input_data/dataset"```**:

  **Purpose**: Specifies the absolute path to the directory containing your multimodal input data (e.g., GeoTIFFs, corresponding CSVs, etc.).
  
  **Action Required** : You MUST replace ```"/path/to/your/input_data/all_mulroy_images_and_sonar"``` with the actual absolute path to your dataset on your local machine.

* **```--output_csv``` ```"/path/to/save/your/results/inference.csv"```**:

  **Purpose**: Defines the absolute path and filename where the inference results (predicted classes, uncertainty metrics) will be saved in CSV format.
  
  **Action Required**: You MUST replace ```"/path/to/save/your/results/inference.csv"``` with your desired output path and filename. The script will create the file and any necessary parent directories if they don't exist.

* **```--batch_size 4:```**

  **Purpose**: Sets the number of samples processed at once by the model during inference.
  
  **Customization**: Adjust this value based on your available GPU memory. Larger batch sizes can speed up inference but require more VRAM.

* **```--num_mc_samples 5```**:

  **Purpose**: Specifies the number of Monte Carlo (MC) samples to draw from the Bayesian Neural Network's posterior distribution. A higher number of samples leads to a more robust estimation of predictive uncertainty.
  
  **Customization**: For production, you might use 100 or more samples for better uncertainty estimation. For quick testing, 5-10 samples are sufficient.

### Expected Output:

Upon successful execution, a CSV file (e.g., inference.csv) will be created at the specified --output_csv path. This file will contain:

* **Image Name**: Identifier for the input sample.

* **Predicted Class**: The model's most likely class prediction.

* **Predictive Uncertainty**: A measure of the total uncertainty in the prediction (combining aleatoric and epistemic).

* **Aleatoric Uncertainty**: Uncertainty inherent in the data itself (e.g., sensor noise, ambiguous regions).


## 3. Retrain a Pre-trained Model on a New Dataset 🔄

This example demonstrates how to fine-tune our pre-trained Multimodal AUV Bayesian Neural Network (Found here: https://huggingface.co/sams-tom/multimodal-auv-bathy-bnn-classifier/tree/main/multimodal-bnn )  on your own custom dataset. Retraining allows you to adapt the model to specific environmental conditions or new benthic classes present in your data, leveraging the knowledge already learned by the pre-trained model.

### Prerequisites:

* Ensure you have cloned this repository and installed all dependencies as described in the Installation Guide.

* Your input data (images and sonar files) should be organized as expected by the CustomImageDataset (refer to ```Multimodal_AUV/data/datasets.py``` or Example.data preparataion above (1) for details). The ```--data_dir``` argument should point to the root of this organized dataset.

* The script will automatically download the required pre-trained model weights from the Hugging Face Hub.

Retraining Command Example:

```
Bash

multimodal-auv-retrain --data_dir "home/tommorgan/Documents/data/representative_sediment_sample/" --batch_size_multimodal 4 --num_epochs_multimodal 5 --num_mc_samples 5 --learning_rate_multimodal 1e-5 --weight_decay_multimodal 1e-5 --bathy_patch_base 30 --sss_patch_base 30


```
To run this as a script:
```
Bash
import torch
from Multimodal_AUV import run_auv_inference

training_root_dir = "/your/data/dir/"
num_classes_for_training = 7 # IMPORTANT: Adjust to your actual number of classes

devices = [torch.device("cuda:0")] if torch.cuda.is_available() else [torch.device("cpu")]


 const_bnn_prior_parameters = {
     "prior_mu": 0.0,
     "prior_sigma": 1.0,
     "posterior_mu_init": 0.0,
     "posterior_rho_init": -3.0,
     "type": "Reparameterization",
     "moped_enable": True,
     "moped_delta": 0.1,
 }

 optimizer_params = {
     "image_model": {"lr": 1e-5}, # Example fixed LR for unimodal
     "bathy_model": {"lr": 0.01}, # Example fixed LR for unimodal
     "sss_model": {"lr": 1e-5},   # Example fixed LR for unimodal
     "multimodal_model": {"lr": args.lr_multimodal}
 }

 scheduler_params = {
     "image_model": {"step_size": 7, "gamma": 0.1},
     "bathy_model": {"step_size": 5, "gamma": 0.5},
     "sss_model": {"step_size": 7, "gamma": 0.7},
     "multimodal_model": {"step_size": 7, "gamma": 0.752} # You might want to make this configurable too
 }

 training_params = {
     "num_epochs_unimodal": 30, # Example fixed value, consider making it an arg
     "num_epochs_multimodal": args.epochs_multimodal,
     "num_mc": args.num_mc,
     "bathy_patch_base": f"patch_{args.bathy_patch_base}_bathy", # Format as string
     "sss_patch_base": f"patch_{args.sss_patch_base}_sss",     # Format as string
     "bathy_patch_types": ["patch_2_bathy", "patch_5_bathy", "patch_10_bathy", "patch_30_bathy", "patch_50_bathy"],
     "sss_patch_types": ["patch_2_sss", "patch_5_sss", "patch_10_sss", "patch_30_sss", "patch_50_sss"],
     "batch_size_unimodal" : args.batch_size_unimodal,
     "batch_size_multimodal" : args.batch_size_multimodal
 }
 run_auv_training(
                optimizer_params=optimizer_params,
                scheduler_params=scheduler_params,
                training_params=training_params,
                root_dir=training_root_dir,
                devices=training_devices,
                const_bnn_prior_parameters=bnn_prior_parameters,
                num_classes=num_classes_for_training
            )
print("Retraining function called.")
```
### Understanding the Arguments:

* **```python -m Multimodal_AUV.Examples.Example_Retraining_model```**: This executes the ```Example_Retraining_model.py``` script as a Python module, which is the recommended way to run scripts within a package structure.

* **```--data_dir``` ```""/path/to/your/input_data/dataset""```**:

   **Purpose**: Specifies the absolute path to the directory containing your multimodal input data for retraining (e.g., GeoTIFFs, corresponding CSVs, etc.).
   
   **Action Required**: You MUST replace ```""/path/to/your/input_data/dataset""``` with the actual absolute path to your dataset on your local machine.

* **```--batch_size_multimodal 20```**:

  **Purpose**: Sets the number of samples processed at once by the model during retraining.
  
  **Customization**: Adjust this value based on your available GPU memory. Larger batch sizes can speed up training but require more VRAM.

* **```--num_epochs_multimodal 20```**:

  **Purpose**: Defines the total number of training epochs (complete passes through the entire dataset).
  
  **Customization**: Increase this value for more thorough training, especially with larger datasets or when the model is converging slowly.

*  **```num_mc_samples 20```**: 

    **Purpose**: Specifies the number of Monte Carlo (MC) samples to draw from the Bayesian Neural Network's posterior distribution during training. A higher number of samples leads to a more robust estimation of predictive uncertainty.
    
    **Customization**: For production, you might use 100 or more samples for better uncertainty estimation. For quicker testing or initial training, 5-10 samples are sufficient.

* **```--learning_rate_multimodal 0.001```**:

  **Purpose**: Sets the initial learning rate for the optimizer. This controls the step size at which the model's weights are updated during training.
  
  **Customization**: Experiment with different learning rates (e.g., 0.01, 0.0001) to find the optimal value for your dataset.

* **```--weight_decay_multimodal 1e-5```**:

  **Purpose**: Applies L2 regularization (weight decay) to prevent overfitting by penalizing large weights.
  
  **Customization**: Adjust this value to control the strength of the regularization. A higher value means stronger regularization.

* **```--bathy_patch_base 30```**:

  **Purpose**: Defines the base patch size for bathymetry data processing.
  
  **Customization**: This parameter affects how bathymetry data is chunked and processed. Adjust as needed based on your data characteristics.

* **```--sss_patch_base 30```**:

  **Purpose**: Defines the base patch size for side-scan sonar (SSS) data processing.
  
  **Customization**: Similar to bathy_patch_base, this affects how SSS data is chunked and processed.

## 4. Train a New Multimodal Model from Scratch 🧠

This example outlines how to train a new Multimodal AUV Bayesian Neural Network entirely from scratch using your own dataset. This is suitable when you have a large, diverse dataset and want to build a model specifically tailored to your data's unique characteristics, without relying on pre-trained weights.

### Prerequisites:

* Ensure you have cloned this repository and installed all dependencies as described in the Installation Guide.

* Your input data (images and sonar files) should be organized as expected by the CustomImageDataset (refer to ```Multimodal_AUV/data/datasets.py``` or example.data_preparation above (1) for details). The ```--root_dir``` argument should point to the root of this organized dataset.

Training Command Example:

```
Bash

multimodal-auv-train-scratch --root_dir "home/tommorgan/Documents/data/representative_sediment_sample/" --batch_size_multimodal 4 --epochs_multimodal 5 --num_mc 5 --lr_multimodal 1e-5 

```
To run this as a script:
```
Bash
import torch
from Multimodal_AUV import run_AUV_training_from_scratch

training_root_dir = "/your/data/dir/"
num_classes_for_training = 7 # IMPORTANT: Adjust to your actual number of classes

devices = [torch.device("cuda:0")] if torch.cuda.is_available() else [torch.device("cpu")]


 const_bnn_prior_parameters = {
     "prior_mu": 0.0,
     "prior_sigma": 1.0,
     "posterior_mu_init": 0.0,
     "posterior_rho_init": -3.0,
     "type": "Reparameterization",
     "moped_enable": True,
     "moped_delta": 0.1,
 }

 optimizer_params = {
     "image_model": {"lr": 1e-5}, # Example fixed LR for unimodal
     "bathy_model": {"lr": 0.01}, # Example fixed LR for unimodal
     "sss_model": {"lr": 1e-5},   # Example fixed LR for unimodal
     "multimodal_model": {"lr": args.lr_multimodal}
 }

 scheduler_params = {
     "image_model": {"step_size": 7, "gamma": 0.1},
     "bathy_model": {"step_size": 5, "gamma": 0.5},
     "sss_model": {"step_size": 7, "gamma": 0.7},
     "multimodal_model": {"step_size": 7, "gamma": 0.752} # You might want to make this configurable too
 }

 training_params = {
     "num_epochs_unimodal": 30, # Example fixed value, consider making it an arg
     "num_epochs_multimodal": args.epochs_multimodal,
     "num_mc": args.num_mc,
     "bathy_patch_base": f"patch_{args.bathy_patch_base}_bathy", # Format as string
     "sss_patch_base": f"patch_{args.sss_patch_base}_sss",     # Format as string
     "bathy_patch_types": ["patch_2_bathy", "patch_5_bathy", "patch_10_bathy", "patch_30_bathy", "patch_50_bathy"],
     "sss_patch_types": ["patch_2_sss", "patch_5_sss", "patch_10_sss", "patch_30_sss", "patch_50_sss"],
     "batch_size_unimodal" : args.batch_size_unimodal,
     "batch_size_multimodal" : args.batch_size_multimodal
 }
run_AUV_training_from_scratch(
    const_bnn_prior_parameters=const_bnn_prior_parameters,
    optimizer_params=optimizer_params,
    scheduler_params=scheduler_params,
    training_params=training_params,
    root_dir=training_root_dir,
    devices=devices,
    num_classes=num_classes_for_training
)
print("Training from scratch function called.")
```
### Understanding the Arguments:

* **```python -m Multimodal_AUV.Examples.Example_training_from_scratch```**: This executes the ```Example_training_from_scratch.py``` script as a Python module, which is the recommended way to run scripts within a package structure.

* **```--root_dir```** "/path/to/your/input_data/dataset":

  **Purpose**: Specifies the absolute path to the root directory containing your multimodal input data for training (e.g., GeoTIFFs, corresponding CSVs, etc.).
  
  **Action Required**: You MUST replace ```/home/tommorgan/Documents/data/representative_sediment_sample/``` with the actual absolute path to your dataset on your local machine.

* **```--epochs_multimodal```** 20:

  **Purpose**: Defines the total number of training epochs (complete passes through the entire dataset).
  
  **Customization**: Increase this value for more thorough training, especially with larger datasets. Training from scratch typically requires more epochs than retraining.

* **```--num_mc```** 20:

  **Purpose**: Specifies the number of Monte Carlo (MC) samples to draw from the Bayesian Neural Network's posterior distribution during training. A higher number of samples leads to a more robust estimation of predictive uncertainty.
  
  **Customization**: For production, you might use 100 or more samples for better uncertainty estimation. For quicker testing or initial training, 5-10 samples are sufficient.

* **```--batch_size_multimodal```** 20:

  **Purpose**: Sets the number of samples processed at once by the model during training.
  
  **Customization**: Adjust this value based on your available GPU memory. Larger batch sizes can speed up training but require more VRAM.

* **```--lr_multimodal```** 0.001:

  **Purpose**: Sets the initial learning rate for the optimizer. This controls the step size at which the model's weights are updated during training.
  
  **Customization**: Experiment with different learning rates (e.g., 0.01, 0.0001) to find the optimal value for your dataset. Training from scratch might require more careful tuning of the learning rate.

# Running tests ✅ 

To ensure the integrity and correctness of the codebase, you can run the provided unit tests. Navigate to the root directory of the repository and execute:

```bash
cd ..
pytest unittests/
```

# ⚙️ Configuration ⚙️

All core parameters for data processing, model training, and inference are controlled via **YAML configuration files**. This approach ensures reproducibility 🔁, simplifies experimentation 🧪, and facilitates seamless collaboration 🤝.

**Key Configuration Areas**:
The configuration is organized to cover various stages of the AUV data processing and model lifecycle:

### Data Management:  📊

Input/Output Paths: Define locations for raw data  (e.g., optical images 📸, GeoTIFFs 🗺️), processed outputs, and inference results.

Data Preparation Parameters: Specify settings like patch sizes forbathymetry 📏 and SSS, image dimensions 🖼️,, and relevant GeoTIFF channels.

### Model Training & Retraining: 🧠

Core Training Parameters: Control fundamental aspects like learning rate 📉, batch size 📦, number of epochs ⏳, and optimization algorithms.

Model Architecture: Configure choices such as model type (e.g., multimodal_bnn, unimodal_bnn), number of output classes, and specific layer dimensions.

Bayesian Neural Network (BNN) Settings: Parameters for BNN priors, if applicable.

### Inference:  🔮

Prediction Control: Define thresholds for classification and output formats for results.

### Configuration Examples and Usage:
Below are examples reflecting the arguments used by various scripts within the project. These can be integrated into a single, comprehensive config.yaml file, or broken down into separate files for specific tasks.

```
YAML

#Configuration File 

#General Project Settings (can be shared across scripts)
global_settings:
  data_root_dir: "/path/to/your/input_data/dataset"
  output_base_dir: "/path/to/your/project_outputs"
  num_mc_samples: 20 # Common for BNN inference/evaluation
  multimodal_batch_size: 20 # Common batch size for multimodal models

#--- Individual Script Configurations ---

#Configuration for Example_training_from_scratch
training_from_scratch:
  epochs_multimodal: 20
  lr_multimodal: 0.001
  # root_dir and batch_size_multimodal can inherit from global_settings or be overridden here

#Configuration for Example_Retraining_model
retraining_model:
  num_epochs_multimodal: 20 # Renamed from 'epochs_multimodal' in original script
  learning_rate_multimodal: 0.001 # Renamed from 'lr_multimodal'
  weight_decay_multimodal: 1e-5
  bathy_patch_base: 30
  sss_patch_base: 30
  # data_dir, batch_size_multimodal, num_mc_samples can inherit from global_settings or be overridden

#Configuration for Example_Inference_model
inference_model:
  output_csv: "%(output_base_dir)s/inference_results/inference.csv" # Example using global var
  batch_size: 4 # Specific batch size for inference

#Configuration for your_script_name.py (e.g., for raw data processing)
raw_data_processing:
  raw_optical_images_folder: "%(data_root_dir)s/raw_auv_images"
  geotiff_folder: "%(data_root_dir)s/auv_geotiffs"
  output_folder: "%(output_base_dir)s/processed_auv_data"
  exiftool_path: "C:/exiftool/" # Note: This might need to be OS-specific or relative
  window_size_meters: 30.0
  image_enhancement_method: "AverageSubtraction"
```

# 🧠 Model Architecture 🏗️

This project leverages sophisticated ** Multimodal Bayesian Neural Network (BNN)** architectures designed for robust data fusion and uncertainty quantification in underwater environments. The core design principles are **modularity** and **adaptability** , allowing for both unimodal and multimodal processing. ✨

## **1. Multimodal Fusion Architecture:**  🤝
The primary model (used in 2.Predict Benthic Habitat Class using a Pre-trained Model 🐠, 3. Retrain a Pre-trained Model on a New Dataset 🔄, 4. Train a New Multimodal Model from Scratch 🧠) is designed to integrate information from different sensor modalities:
* **Image Encoder:** A Convolutional Neural Network (CNN) backbone (e.g., a pre-trained ResNet, specifically adapted to be Bayesian) processes the optical imagery from AUVs. 📸
* **Bathymetric Sonar Encoder(s):** A Convolutional Neural Network (CNN) backbone (e.g., a pre-trained ResNet, specifically adapted to be Bayesian) processes the bathymetric sonar from AUVs. 📏
* * **Side scan sonar Sonar Encoder(s):** A Convolutional Neural Network (CNN) backbone (e.g., a pre-trained ResNet, specifically adapted to be Bayesian) processes the Side scan sonar from AUVs. 📡
* **Fusion Layer:** Features extracted from each modality's encoder are concatenated or combined using a dedicated fusion layer (e.g., a fully connected network, attention mechanism). This layer learns the optimal way to combine visual and acoustic information.  🔗
* **Prediction Head:** A final set of layers (often fully connected) takes the fused features and outputs predictions for the target task (e.g., benthic habitat classification  🐠), with the Bayesian nature providing a distribution over these predictions.

### Diagram of the Multimodal Network:  🖼️
![image](https://github.com/user-attachments/assets/3d799daf-b876-45a6-bc93-5837dd7bd80f)

**2. Bayesian Neural Network Implementation:** 💡
The "Bayesian" aspect is achieved by converting deterministic layers (e.g., Linear, Conv2D) into their probabilistic counterparts using `bayesian-torch`. This means:

* **Weight Distributions:** Instead of learning fixed weights, the model learns **distributions over its weights**, allowing it to output a distribution of predictions for a given input.📊
* **Uncertainty Quantification:** The variance in these output predictions provides a direct measure of the model's confidence and **epistemic uncertainty**, which is vital for decision-making in ambiguous underwater settings. 🌊

**3. Foundation Model Concept:** 🚀
In addition, this project aims to provide a **retrainable foundation model**:
* The architecture is general enough to be applicable across various underwater mapping tasks. 🌐
* It is pre-trained on a diverse dataset (e.g., Northern Britain benthic habitat data), providing strong initial feature representations.💪
* Users can then **fine-tune** this pre-trained model (3. Retrain a Pre-trained Model on a New Dataset 🔄) on their own smaller, specific datasets to adapt it to new areas or different classification schemes, significantly reducing training time and data requirements. ⏱️

**4. Unimodal Models:**  🎯
The project also includes components (`unitmodal.py` in `train/` and potentially `base_models.py`) to train and evaluate models based on **single modalities** (e.g.,  image-only 📸 or sonar-only 📡). This allows for ablation studies and comparison with the performance benefits of multimodal fusion.

### Diagram of the Unimodal Networks: 🖼️
![image](https://github.com/user-attachments/assets/14dbb63f-864d-4fca-9bb5-fd26b91ea827)


---
# Contact
Have questions about the project, found a bug, or want to contribute? Here are a few ways to reach out:

* **GitHub Issues:** For any code-related questions, bug reports, or feature requests, please open an [Issue on this repository](https://github.com/sams-tom/multimodal-auv-bnn-project/issues). This is the preferred method for transparency and tracking.

* **Email:** For more direct or confidential inquiries, you can reach me at [phd01tm@sams.ac.uk](mailto:phd01tm@sams.ac.uk).

* **LinkedIn:** Connect with the project lead/team on LinkedIn:
    * [Tom Morgan](https://www.linkedin.com/in/tom-morgan-8a73b129b/)
      
# Citations

* **GitHub Repository (Code & Documentation):** [https://github.com/sams-tom/multimodal-auv-bnn-project](https://github.com/sams-tom/multimodal-auv-bnn-project)
* **Hugging Face Models:** [https://huggingface.co/sams-tom/multimodal-auv-bnn-models](https://huggingface.co/sams-tom/multimodal-auv-bnn-models)
* **Research Paper:** [In development]
