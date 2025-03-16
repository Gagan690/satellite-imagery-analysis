# utils/image_processing.py
# Module containing functions for processing satellite imagery

import os
import numpy as np
import cv2
import logging
import rasterio
from rasterio.plot import reshape_as_image
from scipy import ndimage
from skimage import exposure, feature, filters, segmentation, color
from sklearn.cluster import KMeans

# Configure logger
logger = logging.getLogger(__name__)

def load_image(file_path):
    """
    Load a satellite image from a file path.
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        tuple: (image_data, metadata)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file {file_path} does not exist")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext in ['.tif', '.tiff']:
            # Use rasterio for GeoTIFF files
            with rasterio.open(file_path) as src:
                metadata = {
                    'driver': src.driver,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'crs': src.crs,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'nodata': src.nodata
                }
                
                # Read all bands
                image_data = src.read()
                
                # Reshape to (height, width, bands) for easier handling
                if src.count > 1:
                    image_data = reshape_as_image(image_data)
                else:
                    image_data = image_data[0]
                
                return image_data, metadata
                
        elif file_ext in ['.jp2']:
            # Use rasterio for JPEG2000 files
            try:
                with rasterio.open(file_path) as src:
                    metadata = {
                        'driver': src.driver,
                        'width': src.width,
                        'height': src.height,
                        'count': src.count,
                        'crs': src.crs,
                        'transform': src.transform,
                        'bounds': src.bounds,
                        'nodata': src.nodata
                    }
                    
                    # Read all bands
                    image_data = src.read()
                    
                    # Reshape to (height, width, bands) for easier handling
                    if src.count > 1:
                        image_data = reshape_as_image(image_data)
                    else:
                        image_data = image_data[0]
                    
                    return image_data, metadata
            except Exception as e:
                logger.error(f"Failed to open JPEG2000 file with rasterio: {str(e)}")
                # Fall back to OpenCV if rasterio fails
                image_data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if image_data is None:
                    raise ValueError(f"Unable to open {file_path} with OpenCV or rasterio")
                    
                metadata = {
                    'width': image_data.shape[1],
                    'height': image_data.shape[0],
                    'count': 1 if len(image_data.shape) == 2 else image_data.shape[2],
                    'driver': 'JP2OpenJPEG'
                }
                
                return image_data, metadata
            
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            # Use OpenCV for standard image formats
            image_data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image_data is None:
                raise ValueError(f"Unable to open {file_path} with OpenCV")
                
            metadata = {
                'width': image_data.shape[1],
                'height': image_data.shape[0],
                'count': 1 if len(image_data.shape) == 2 else image_data.shape[2],
                'driver': file_ext[1:]
            }
            
            return image_data, metadata
            
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    except Exception as e:
        logger.error(f"Error loading image {file_path}: {str(e)}")
        raise
    
def save_image(image_data, output_path, metadata=None):
    """
    Save processed image data to a file.
    
    Args:
        image_data (numpy.ndarray): Image data to save
        output_path (str): Path where the image should be saved
        metadata (dict, optional): Metadata to include in the saved file
        
    Returns:
        bool: True if successful, False otherwise
        
    Raises:
        ValueError: If there's an issue saving the file
    """
    try:
        file_ext = os.path.splitext(output_path)[1].lower()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if file_ext in ['.tif', '.tiff']:
            # Save as GeoTIFF with rasterio
            if metadata and 'count' in metadata:
                count = metadata['count']
            else:
                count = 1 if len(image_data.shape) == 2 else image_data.shape[2]
                
            height, width = image_data.shape[0], image_data.shape[1]
            
            # Prepare image data in the format expected by rasterio
            if len(image_data.shape) == 3:
                # Convert from (height, width, bands) to (bands, height, width)
                image_data = np.transpose(image_data, (2, 0, 1))
            else:
                # Add band dimension for 2D images
                image_data = image_data[np.newaxis, :, :]
            
            # Set up rasterio profile
            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': count,
                'dtype': image_data.dtype
            }
            
            # Add geospatial metadata if available
            if metadata:
                if 'crs' in metadata:
                    profile['crs'] = metadata['crs']
                if 'transform' in metadata:
                    profile['transform'] = metadata['transform']
                if 'nodata' in metadata:
                    profile['nodata'] = metadata['nodata']
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(image_data)
            
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            # Save as standard image format with OpenCV
            if len(image_data.shape) == 3 and image_data.shape[2] > 3:
                # If more than 3 bands, save only the first 3 (RGB)
                logger.warning(f"Image has {image_data.shape[2]} bands, saving only the first 3 as RGB")
                image_data = image_data[:, :, :3]
                
            # Normalize data to 0-255 if it's floating point
            if image_data.dtype == np.float32 or image_data.dtype == np.float64:
                image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255
                image_data = image_data.astype(np.uint8)
                
            cv2.imwrite(output_path, image_data)
            
        else:
            raise ValueError(f"Unsupported output format: {file_ext}")
            
        logger.info(f"Image successfully saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {str(e)}")
        raise ValueError(f"Failed to save image: {str(e)}")

def histogram_equalization(image):
    """
    Apply histogram equalization to enhance image contrast.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Check if the image has multiple channels
    if len(image.shape) == 3:
        # Apply to each channel separately
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[2]):
            result[:, :, i] = exposure.equalize_hist(image[:, :, i])
    else:
        # Single channel image
        result = exposure.equalize_hist(image)
    
    logger.debug("Applied histogram equalization")
    return result

def gaussian_blur(image, sigma=1.0):
    """
    Apply Gaussian blur for noise reduction.
    
    Args:
        image (numpy.ndarray): Input image
        sigma (float): Standard deviation for Gaussian kernel
        
    Returns:
        numpy.ndarray: Blurred image
    """
    return filters.gaussian(image, sigma=sigma, preserve_range=True)

def median_filter(image, size=3):
    """
    Apply median filter for noise reduction while preserving edges.
    
    Args:
        image (numpy.ndarray): Input image
        size (int): Size of the median filter kernel
        
    Returns:
        numpy.ndarray: Filtered image
    """
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = ndimage.median_filter(image[:, :, i], size=size)
    else:
        result = ndimage.median_filter(image, size=size)
    
    return result

def edge_detection(image, method='canny'):
    """
    Detect edges in the image using various methods.
    
    Args:
        image (numpy.ndarray): Input image
        method (str): Edge detection method ('canny', 'sobel', 'roberts', 'prewitt')
        
    Returns:
        numpy.ndarray: Edge map
    """
    # Convert to grayscale if it's a multi-channel image
    if len(image.shape) == 3:
        if image.shape[2] >= 3:
            gray = color.rgb2gray(image[:, :, :3])
        else:
            gray = image[:, :, 0]
    else:
        gray = image
    
    # Apply appropriate edge detection algorithm
    if method == 'canny':
        return feature.canny(gray)
    elif method == 'sobel':
        return filters.sobel(gray)
    elif method == 'roberts':
        return filters.roberts(gray)
    elif method == 'prewitt':
        return filters.prewitt(gray)
    else:
        raise ValueError(f"Unsupported edge detection method: {method}")

def calculate_ndvi(image, red_band=2, nir_band=3):
    """
    Calculate Normalized Difference Vegetation Index.
    
    Args:
        image (numpy.ndarray): Multi-band satellite image
        red_band (int): Index of the red band (0-based)
        nir_band (int): Index of the near-infrared band (0-based)
        
    Returns:
        numpy.ndarray: NDVI values (-1 to 1)
        
    Raises:
        ValueError: If the image doesn't have enough bands
    """
    if len(image.shape) < 3 or image.shape[2] <= max(red_band, nir_band):
        raise ValueError(f"Image doesn't have the required bands. Shape: {image.shape}")
    
    # Extract red and NIR bands
    red = image[:, :, red_band].astype(np.float32)
    nir = image[:, :, nir_band].astype(np.float32)
    
    # Avoid division by zero
    denominator = nir + red
    ndvi = np.zeros_like(red)
    valid_mask = denominator > 0
    ndvi[valid_mask] = (nir[valid_mask] - red[valid_mask]) / denominator[valid_mask]
    
    logger.debug("NDVI calculation completed")
    return ndvi

def unsupervised_classification(image, n_clusters=5):
    """
    Perform unsupervised classification using K-Means clustering.
    
    Args:
        image (numpy.ndarray): Input multi-band image
        n_clusters (int): Number of clusters to create
        
    Returns:
        numpy.ndarray: Classification map
    """
    # Reshape image for clustering
    h, w, bands = image.shape
    reshaped_image = image.reshape((h * w, bands))
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reshaped_image)
    
    # Reshape back to image dimensions
    classified_image = labels.reshape((h, w))
    
    logger.debug(f"Unsupervised classification completed with {n_clusters} clusters")
    return classified_image

def create_composite(image, band_indices=[0, 1, 2]):
    """
    Create a composite image from specified bands.
    
    Args:
        image (numpy.ndarray): Multi-band satellite image
        band_indices (list): Indices of bands to use for RGB composite
        
    Returns:
        numpy.ndarray: RGB composite image
        
    Raises:
        ValueError: If the image doesn't have enough bands
    """
    if len(image.shape) < 3 or image.shape[2] <= max(band_indices):
        raise ValueError(f"Image doesn't have the required bands. Shape: {image.shape}")
    
    # Extract bands for the composite
    composite = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    
    for i, band_idx in enumerate(band_indices[:3]):
        composite[:, :, i] = image[:, :, band_idx]
    
    # Normalize each band to 0-1 range
    for i in range(3):
        if composite[:, :, i].max() > 0:
            composite[:, :, i] = (composite[:, :, i] - composite[:, :, i].min()) / (composite[:, :, i].max() - composite[:, :, i].min())
    
    logger.debug(f"Created band composite using bands {band_indices}")
    return composite

def contrast_stretch(image, percentiles=(2, 98)):
    """
    Apply contrast stretching to enhance the image.
    
    Args:
        image (numpy.ndarray): Input image
        percentiles (tuple): Lower and upper percentiles for stretching
        
    Returns:
        numpy.ndarray: Contrast-enhanced image
    """
    if len(image.shape) == 3:
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[2]):
            p_low, p_high = np.percentile(image[:, :, i], percentiles)
            result[:, :, i] = exposure.rescale_intensity(image[:, :, i], in_range=(p_low, p_high))
    else:
        p_low, p_high = np.percentile(image, percentiles)
        result = exposure.rescale_intensity(image, in_range=(p_low, p_high))
    
    logger.debug(f"Applied contrast stretching with percentiles {percentiles}")
    return result

def segment_image(image, method='watershed', n_segments=100):
    """
    Segment the image into regions.
    
    Args:
        image (numpy.ndarray): Input image
        method (str): Segmentation method ('watershed', 'slic', 'quickshift')
        n_segments (int): Number of segments (for applicable methods)
        
    Returns:
        numpy.ndarray: Segmented image with labels
    """
    # Convert to 3-channel image if it's grayscale
    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 1:
        image_rgb = np.stack([image[:, :, 0]] * 3, axis=-1)
    else:
        # Use first 3 bands or copy bands if less than 3
        if image.shape[2] >= 3:
            image_rgb = image[:, :, :3]
        else:
            image_rgb = np.zeros((image.shape[0], image.shape[1], 3))
            for i in range(min(3, image.shape[2])):
                image_rgb[:, :, i] = image[:, :, i]
    
    # Normalize to 0-1 range
    if image_rgb.max() > 1.0:
        image_rgb = image_rgb / 255.0
        
    # Apply segmentation
    if method == 'watershed':
        gradient = filters.sobel(color.rgb2gray(image_rgb))
        segments = segmentation.watershed(gradient, markers=n_segments, compactness=0.001)
    elif method == 'slic':
        segments = segmentation.slic(image_rgb, n_segments=n_segments, compactness=10)
    elif method == 'quickshift':
        segments = segmentation.quickshift(image_rgb, kernel_size=3, max_dist=6, ratio=0.5)
    else:
        raise ValueError(f"Unsupported segmentation method: {method}")
        
    logger.debug(f"Image segmentation completed using {method}")
    return segments
