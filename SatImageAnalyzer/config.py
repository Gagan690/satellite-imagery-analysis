# config.py
# Configuration settings for the Satellite Imagery Analysis tool

import os

class Config:
    """Base configuration class for the application"""
    # Supported image formats
    SUPPORTED_FORMATS = ['tif', 'tiff', 'jp2', 'png', 'jpg', 'jpeg']
    
    # Temporary directory for uploaded files
    UPLOAD_FOLDER = 'uploads'
    
    # Results directory for analysis outputs
    RESULTS_FOLDER = 'results'
    
    # Maximum image dimensions to process (to prevent memory issues)
    MAX_IMAGE_DIMENSIONS = (10000, 10000)
    
    # Available image processing techniques
    PROCESSING_TECHNIQUES = [
        'histogram_equalization',
        'gaussian_blur',
        'median_filter',
        'edge_detection',
        'ndvi_calculation',
        'rgb_composite',
        'supervised_classification',
        'unsupervised_classification',
        'band_combination',
        'pansharpening'
    ]
    
    # Available feature extraction methods
    FEATURE_EXTRACTION_METHODS = [
        'vegetation_indices', 
        'urban_detection',
        'water_detection',
        'change_detection',
        'object_detection',
        'texture_analysis'
    ]
    
    # Default visualization settings
    DEFAULT_COLORMAP = 'viridis'
    
    # Satellite sensors supported
    SUPPORTED_SENSORS = [
        'Landsat',
        'Sentinel',
        'MODIS',
        'WorldView',
        'PlanetScope',
        'QuickBird',
        'SPOT'
    ]

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = False
    TESTING = True
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

# Get configuration based on environment
def get_config():
    env = os.environ.get('FLASK_ENV', 'development')
    
    configs = {
        'development': DevelopmentConfig,
        'testing': TestingConfig,
        'production': ProductionConfig
    }
    
    return configs.get(env, DevelopmentConfig)
