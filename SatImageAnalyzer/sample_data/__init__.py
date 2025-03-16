# sample_data/__init__.py
# Package initialization for the sample data module

# This package contains sample satellite imagery for demonstration purposes

"""
Sample Satellite Imagery

This package provides sample satellite imagery for users to test the 
Satellite Imagery Analysis Tool without having to upload their own data.

The sample images are stored in this directory and metadata about them
is provided in the metadata.py file.
"""

import os
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Ensure the sample data directory exists
try:
    os.makedirs('sample_data', exist_ok=True)
    logger.info("Sample data directory checked/created")
except Exception as e:
    logger.error(f"Error creating sample data directory: {str(e)}")

# Note: The actual sample image files should be placed in this directory
# but are not included in the code generation as they would be binary files.
# In a real application, you would need to provide these files separately.
