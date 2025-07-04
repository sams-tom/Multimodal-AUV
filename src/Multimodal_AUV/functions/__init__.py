
# Import the functions you want to expose from functions.py
from .functions import (
    run_auv_inference,
    run_auv_retraining, 
    run_auv_preprocessing,
    run_AUV_training_from_scratch
)

__all__ = [
    "run_auv_inference",
    "cli_main",
    "run_auv_retraining",
    "run_auv_preprocessing",
    "run_AUV_training_from_scratch"
]

# You might want to set up a logger for this sub-package as well
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())