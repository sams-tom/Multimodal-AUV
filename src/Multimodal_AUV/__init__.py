# File: src/Multimodal_AUV/__init__.py

# --- Expose your core, reusable functions at the top level ---
# Use RELATIVE IMPORTS when importing within your own package
from .functions.functions import ( # Notice the leading dot '.'
    run_auv_inference,
    run_auv_retraining,
    run_auv_preprocessing,
    run_AUV_training_from_scratch
)

# --- Package Metadata (02/07/2025)) ---
__version__ = "0.1.0"
__author__ = "Tom Morgan"
__email__ = "phd01tm@sams.ac.uk"

# --- Define what gets imported with `from Multimodal_AUV import *` (Optional but good practice) ---
# Explicitly list the public API for convenience imports.
__all__ = [
    "run_auv_inference",
    "run_auv_retraining",
    "run_auv_preprocessing",
    "run_AUV_training_from_scratch"
]

# --- Basic Logging Setup for the Package ---
import logging
# This ensures that if a user of your package configures logging,
# messages from your package will go to their configured handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())