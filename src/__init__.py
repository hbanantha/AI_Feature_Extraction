"""
AI Feature Extraction from Drone Images
SVAMITVA Scheme Orthophoto Analysis

This package provides tools for extracting features from drone imagery:

• Building footprint extraction with roof classification
  (RCC, Tiled, Tin, Others)

• Road feature extraction

• Waterbody extraction

• Infrastructure detection
  (distribution transformers, overhead tanks, wells)

The system is optimized for low-resource environments
"""

__version__ = "1.0.0"
__author__ = "AI Feature Extraction Team"

# Package modules
from . import preprocessing
from . import models
from . import training
from . import inference