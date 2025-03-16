"""
Satellite Image Utilities Module
--------------------------------

This module provides utility functions used across the satellite imagery analysis package.
It includes tools for file handling, coordinate conversion, and miscellaneous helper functions.

Key functions:
- File utility functions
- Coordinate conversion functions
- Progress tracking
- Error handling

Dependencies:
- numpy: For numerical operations
- pyproj: For coordinate transformations
"""

import os
import uuid
import shutil
import tempfile
import numpy as np
from datetime import datetime
import logging
import math
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def create_temp_directory():
    """
    Create a temporary directory for processing files.
    
    Returns:
    --------
    str
        Path to the created temporary directory
    """
    temp_dir = os.path.join(tempfile.gettempdir(), f"sat_analysis_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    logger.debug(f"Created temporary directory: {temp_dir}")
    return temp_dir

def clean_temp_directory(temp_dir):
    """
    Clean up a temporary directory.
    
    Parameters:
    -----------
    temp_dir : str
        Path to the temporary directory to clean
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        logger.debug(f"Cleaned temporary directory: {temp_dir}")

def filename_to_safe_string(filename):
    """
    Convert a filename to a safe string for use in output filenames.
    
    Parameters:
    -----------
    filename : str
        Original filename
        
    Returns:
    --------
    str
        Safe string version of the filename
    """
    # Remove extension
    base_name = os.path.splitext(filename)[0]
    
    # Replace unsafe characters
    safe_string = "".join(c if c.isalnum() else "_" for c in base_name)
    
    # Ensure the string is not empty
    if not safe_string:
        safe_string = "unnamed"
    
    return safe_string

def get_file_extension(file_path):
    """
    Get the file extension from a file path.
    
    Parameters:
    -----------
    file_path : str
        Path to the file
        
    Returns:
    --------
    str
        File extension (lowercase, without dot)
    """
    return os.path.splitext(file_path)[1].lower()[1:]

def is_supported_image_format(file_path):
    """
    Check if a file is in a supported image format.
    
    Parameters:
    -----------
    file_path : str
        Path to the file
        
    Returns:
    --------
    bool
        True if the file format is supported, False otherwise
    """
    supported_formats = {'tif', 'tiff', 'jp2', 'img', 'hdf', 'nc', 'png', 'jpg', 'jpeg'}
    return get_file_extension(file_path) in supported_formats

def generate_output_filename(input_filename, suffix, output_dir=None, extension=None):
    """
    Generate an output filename based on an input filename.
    
    Parameters:
    -----------
    input_filename : str
        Original filename
    suffix : str
        Suffix to append to the filename
    output_dir : str or None
        Directory to place the output file (if None, use same directory as input)
    extension : str or None
        File extension to use (if None, use same extension as input)
        
    Returns:
    --------
    str
        Generated output filename
    """
    # Split input filename into directory, basename, and extension
    input_dir, input_basename = os.path.split(input_filename)
    basename, input_ext = os.path.splitext(input_basename)
    
    # Determine output directory
    if output_dir is None:
        output_dir = input_dir
    
    # Determine output extension
    if extension is None:
        output_ext = input_ext
    else:
        output_ext = f".{extension}" if not extension.startswith('.') else extension
    
    # Generate output filename
    output_basename = f"{basename}_{suffix}{output_ext}"
    output_path = os.path.join(output_dir, output_basename)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    return output_path

def ensure_directory_exists(directory_path):
    """
    Ensure a directory exists, creating it if necessary.
    
    Parameters:
    -----------
    directory_path : str
        Path to the directory
        
    Returns:
    --------
    str
        Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)
    return directory_path

def calculate_area(mask, pixel_size):
    """
    Calculate the area of a binary mask given the pixel size.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        Binary mask (0 or 1)
    pixel_size : float or tuple
        Size of each pixel in square meters. If tuple, contains (x_size, y_size).
        
    Returns:
    --------
    float
        Area in square meters
    """
    # Count the number of pixels in the mask
    pixel_count = np.sum(mask)
    
    # Calculate area based on pixel size
    if isinstance(pixel_size, tuple):
        x_size, y_size = pixel_size
        area = pixel_count * x_size * y_size
    else:
        area = pixel_count * pixel_size * pixel_size
    
    return area

def lat_lon_to_pixel(lat, lon, geotransform):
    """
    Convert latitude and longitude to pixel coordinates.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude
    geotransform : tuple
        GDAL-style geotransform parameters
        
    Returns:
    --------
    tuple
        (x, y) pixel coordinates
    """
    # Extract geotransform parameters
    origin_x, pixel_width, _, origin_y, _, pixel_height = geotransform
    
    # Calculate pixel coordinates
    x = int((lon - origin_x) / pixel_width)
    y = int((lat - origin_y) / pixel_height)
    
    return x, y

def pixel_to_lat_lon(x, y, geotransform):
    """
    Convert pixel coordinates to latitude and longitude.
    
    Parameters:
    -----------
    x : int
        X pixel coordinate
    y : int
        Y pixel coordinate
    geotransform : tuple
        GDAL-style geotransform parameters
        
    Returns:
    --------
    tuple
        (lat, lon) coordinates
    """
    # Extract geotransform parameters
    origin_x, pixel_width, _, origin_y, _, pixel_height = geotransform
    
    # Calculate lat/lon coordinates
    lon = origin_x + pixel_width * x
    lat = origin_y + pixel_height * y
    
    return lat, lon

def calculate_ndvi_thresholds(ndvi_values, method='otsu'):
    """
    Calculate thresholds for NDVI classification.
    
    Parameters:
    -----------
    ndvi_values : numpy.ndarray
        Array of NDVI values
    method : str
        Method for threshold calculation:
        - 'otsu': Otsu's method for binary thresholding
        - 'percentile': Use percentiles for multi-level thresholding
        - 'kmeans': Use k-means clustering for thresholding
        
    Returns:
    --------
    list
        List of threshold values
    """
    # Remove NaN values
    valid_ndvi = ndvi_values[~np.isnan(ndvi_values)]
    
    if method == 'otsu':
        try:
            from skimage.filters import threshold_otsu
            # Calculate single threshold using Otsu's method
            threshold = threshold_otsu(valid_ndvi)
            return [threshold]
        except ImportError:
            logger.warning("scikit-image not available. Using percentile method instead.")
            method = 'percentile'
    
    if method == 'percentile':
        # Calculate thresholds based on percentiles
        thresholds = [
            np.percentile(valid_ndvi, 20),  # Low vegetation
            np.percentile(valid_ndvi, 50),  # Moderate vegetation
            np.percentile(valid_ndvi, 80)   # High vegetation
        ]
        return thresholds
    
    if method == 'kmeans':
        try:
            from sklearn.cluster import KMeans
            # Reshape for k-means
            X = valid_ndvi.reshape(-1, 1)
            # Apply k-means with 4 clusters
            kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
            # Get cluster centers and sort them
            centers = sorted(kmeans.cluster_centers_.flatten().tolist())
            # Use the boundaries between clusters as thresholds
            thresholds = [(centers[i] + centers[i+1]) / 2 for i in range(len(centers)-1)]
            return thresholds
        except ImportError:
            logger.warning("scikit-learn not available. Using percentile method instead.")
            method = 'percentile'
            return calculate_ndvi_thresholds(ndvi_values, method=method)
    
    # Default fallback
    return [0.2, 0.4, 0.6]

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth using the Haversine formula.
    
    Parameters:
    -----------
    lat1 : float
        Latitude of the first point in degrees
    lon1 : float
        Longitude of the first point in degrees
    lat2 : float
        Latitude of the second point in degrees
    lon2 : float
        Longitude of the second point in degrees
        
    Returns:
    --------
    float
        Distance in meters
    """
    # Earth radius in meters
    R = 6371000
    
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance

def create_timestamp_string():
    """
    Create a timestamp string for use in filenames.
    
    Returns:
    --------
    str
        Formatted timestamp string
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def get_sample_data_path():
    """
    Get the path to the sample data directory.
    
    Returns:
    --------
    str
        Path to the sample data directory
    """
    # Get the directory of the current script
    base_dir = Path(__file__).resolve().parent.parent
    sample_data_dir = os.path.join(base_dir, 'static', 'sample_data')
    
    # Create the directory if it doesn't exist
    os.makedirs(sample_data_dir, exist_ok=True)
    
    return sample_data_dir

def sanitize_dict_for_json(d):
    """
    Sanitize a dictionary to ensure it can be serialized to JSON.
    
    Parameters:
    -----------
    d : dict
        Dictionary to sanitize
        
    Returns:
    --------
    dict
        Sanitized dictionary
    """
    result = {}
    for k, v in d.items():
        # Skip any non-string keys or keys starting with underscore
        if not isinstance(k, str) or k.startswith('_'):
            continue
            
        if isinstance(v, dict):
            # Recursively sanitize nested dictionaries
            result[k] = sanitize_dict_for_json(v)
        elif isinstance(v, (list, tuple)):
            # Sanitize lists
            result[k] = [sanitize_value_for_json(item) for item in v]
        else:
            # Sanitize the value
            result[k] = sanitize_value_for_json(v)
    
    return result

def sanitize_value_for_json(v):
    """
    Sanitize a value to ensure it can be serialized to JSON.
    
    Parameters:
    -----------
    v : any
        Value to sanitize
        
    Returns:
    --------
    any
        Sanitized value
    """
    if isinstance(v, (str, int, float, bool, type(None))):
        # These types are JSON-serializable
        return v
    elif isinstance(v, (np.integer, np.int32, np.int64)):
        # Convert NumPy integers to Python integers
        return int(v)
    elif isinstance(v, (np.float32, np.float64)):
        # Convert NumPy floats to Python floats
        return float(v)
    elif isinstance(v, np.ndarray):
        # Convert NumPy arrays to lists
        return v.tolist()
    elif isinstance(v, (datetime, np.datetime64)):
        # Convert datetime objects to ISO format strings
        return v.isoformat()
    elif isinstance(v, dict):
        # Recursively sanitize nested dictionaries
        return sanitize_dict_for_json(v)
    elif isinstance(v, (list, tuple)):
        # Recursively sanitize lists
        return [sanitize_value_for_json(item) for item in v]
    else:
        # Convert other types to strings
        return str(v)
