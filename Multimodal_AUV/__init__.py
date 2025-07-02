
# my_auv_module/__init__.py

# --- Expose your core, reusable functions at the top level ---
# These are the functions other Python code will likely want to import and use.
from .functions.functions import (
    run_auv_inference,
    run_auv_training,
    run_auv_preprocessing,
    run_AUV_training_from_scratch
)

# --- Package Metadata (02/07/2025)) ---
__version__ = "0.1.0"
__author__ = "Tom Morgan"
__email__ = "phd01tm@sams.ac.uk"

# --- Define what gets imported with `from my_auv_module import *` (Optional) ---
# Explicitly list the public API for convenience imports.
__all__ = [
    "run_auv_inference",
    "run_auv_training",
    "run_auv_preprocessing",
    "run_AUV_training_from_scratch"
]

# --- Basic Logging Setup for the Package  ---
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())