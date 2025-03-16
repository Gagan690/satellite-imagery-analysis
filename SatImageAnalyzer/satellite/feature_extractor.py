"""
Satellite Image Feature Extractor Module
----------------------------------------

This module provides functions for extracting features and information from
satellite imagery, including vegetation indices, water bodies, urban areas,
and more.

Key functions:
- NDVI calculation (Normalized Difference Vegetation Index)
- Water body extraction
- Urban area detection
- Land cover classification
- Change detection

Dependencies:
- NumPy: For numerical operations
- SciPy: For scientific computing algorithms
- scikit-learn: For machine learning algorithms
- scikit-image: For image processing functions
"""

import numpy as np
from scipy import ndimage
import cv2
import logging
from skimage import filters, morphology, segmentation, color

# Set up logging
logger = logging.getLogger(__name__)

def calculate_ndvi(img_data):
    """
    Calculate the Normalized Difference Vegetation Index (NDVI).
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Input satellite image as NumPy array. Expected to have at least
        red and near-infrared bands.
        
    Returns:
    --------
    numpy.ndarray
        NDVI values ranging from -1 to 1
        
    Notes:
    ------
    NDVI = (NIR - Red) / (NIR + Red)
    
    For Landsat 8: NIR = Band 5, Red = Band 4
    For Sentinel-2: NIR = Band 8, Red = Band 4
    
    This function assumes the image has at least 4 bands with red and NIR present.
    If not, it falls back to a simulated NDVI using available bands.
    """
    # Check if the image has enough bands for NDVI calculation
    if img_data.ndim < 3 or img_data.shape[2] < 4:
        logger.warning("Image doesn't have enough bands for true NDVI calculation. "
                      "Attempting to simulate NDVI with available bands.")
        
        if img_data.ndim == 3 and img_data.shape[2] >= 3:
            # For RGB image, try to simulate NDVI using red and green bands
            # This is just an approximation for visualization purposes
            red_band = img_data[:, :, 0].astype(np.float32)
            green_band = img_data[:, :, 1].astype(np.float32)
            
            # Normalize if needed
            if red_band.max() > 1.0 or green_band.max() > 1.0:
                red_band = red_band / 255.0
                green_band = green_band / 255.0
            
            # Use green band as a crude substitute for NIR
            pseudo_ndvi = (green_band - red_band) / (green_band + red_band + 1e-10)
            
            logger.info("Calculated pseudo-NDVI using red and green bands")
            return pseudo_ndvi
            
        elif img_data.ndim == 2:
            # For single band image, we can't calculate NDVI
            logger.error("Cannot calculate NDVI from single-band image")
            return np.zeros_like(img_data, dtype=np.float32)
    
    # Extract NIR and Red bands
    # Assuming standard channel order (R, G, B, NIR, ...)
    red_band = img_data[:, :, 0].astype(np.float32)
    nir_band = img_data[:, :, 3].astype(np.float32)
    
    # Normalize if needed
    if red_band.max() > 1.0 or nir_band.max() > 1.0:
        red_band = red_band / 255.0
        nir_band = nir_band / 255.0
    
    # Calculate NDVI
    # Add small constant to avoid division by zero
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)
    
    # Clip values to valid NDVI range (-1 to 1)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    
    logger.info("Calculated NDVI from red and NIR bands")
    return ndvi

def calculate_ndwi(img_data):
    """
    Calculate the Normalized Difference Water Index (NDWI).
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Input satellite image as NumPy array. Expected to have at least
        green and near-infrared bands.
        
    Returns:
    --------
    numpy.ndarray
        NDWI values ranging from -1 to 1
        
    Notes:
    ------
    NDWI = (Green - NIR) / (Green + NIR)
    
    For Landsat 8: Green = Band 3, NIR = Band 5
    For Sentinel-2: Green = Band 3, NIR = Band 8
    
    This function assumes the image has at least 4 bands with green and NIR present.
    If not, it falls back to a simulated NDWI using available bands.
    """
    # Check if the image has enough bands for NDWI calculation
    if img_data.ndim < 3 or img_data.shape[2] < 4:
        logger.warning("Image doesn't have enough bands for true NDWI calculation. "
                      "Attempting to simulate NDWI with available bands.")
        
        if img_data.ndim == 3 and img_data.shape[2] >= 3:
            # For RGB image, try to simulate NDWI using blue and green bands
            # This is just an approximation for visualization purposes
            green_band = img_data[:, :, 1].astype(np.float32)
            blue_band = img_data[:, :, 2].astype(np.float32)
            
            # Normalize if needed
            if green_band.max() > 1.0 or blue_band.max() > 1.0:
                green_band = green_band / 255.0
                blue_band = blue_band / 255.0
            
            # Use blue band as a crude substitute for NIR
            pseudo_ndwi = (green_band - blue_band) / (green_band + blue_band + 1e-10)
            
            logger.info("Calculated pseudo-NDWI using green and blue bands")
            return pseudo_ndwi
            
        elif img_data.ndim == 2:
            # For single band image, we can't calculate NDWI
            logger.error("Cannot calculate NDWI from single-band image")
            return np.zeros_like(img_data, dtype=np.float32)
    
    # Extract Green and NIR bands
    # Assuming standard channel order (R, G, B, NIR, ...)
    green_band = img_data[:, :, 1].astype(np.float32)
    nir_band = img_data[:, :, 3].astype(np.float32)
    
    # Normalize if needed
    if green_band.max() > 1.0 or nir_band.max() > 1.0:
        green_band = green_band / 255.0
        nir_band = nir_band / 255.0
    
    # Calculate NDWI
    # Add small constant to avoid division by zero
    ndwi = (green_band - nir_band) / (green_band + nir_band + 1e-10)
    
    # Clip values to valid NDWI range (-1 to 1)
    ndwi = np.clip(ndwi, -1.0, 1.0)
    
    logger.info("Calculated NDWI from green and NIR bands")
    return ndwi

def extract_water_bodies(img_data, method='ndwi', threshold=0.3):
    """
    Extract water bodies from satellite imagery.
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Input satellite image as NumPy array
    method : str
        Method for water extraction. Options:
        - 'ndwi': Normalized Difference Water Index
        - 'blue': Using blue channel thresholding
        - 'nir': Using NIR channel thresholding
    threshold : float
        Threshold value for water classification
        
    Returns:
    --------
    numpy.ndarray
        Binary mask of water bodies (1 for water, 0 for non-water)
    """
    water_mask = None
    
    if method == 'ndwi':
        # Use NDWI for water detection
        ndwi = calculate_ndwi(img_data)
        
        # Threshold NDWI to get water mask
        water_mask = ndwi > threshold
    
    elif method == 'blue':
        # Use blue band for water detection
        if img_data.ndim == 3 and img_data.shape[2] >= 3:
            blue_band = img_data[:, :, 2].astype(np.float32)
            
            # Normalize if needed
            if blue_band.max() > 1.0:
                blue_band = blue_band / 255.0
            
            # Apply Otsu's thresholding
            thresh = filters.threshold_otsu(blue_band)
            water_mask = blue_band > thresh
        else:
            logger.error("Cannot extract water bodies using blue band from this image")
            return np.zeros((img_data.shape[0], img_data.shape[1]), dtype=np.uint8)
    
    elif method == 'nir':
        # Use NIR band for water detection (water absorbs NIR, so it appears dark)
        if img_data.ndim == 3 and img_data.shape[2] >= 4:
            nir_band = img_data[:, :, 3].astype(np.float32)
            
            # Normalize if needed
            if nir_band.max() > 1.0:
                nir_band = nir_band / 255.0
            
            # Apply Otsu's thresholding
            thresh = filters.threshold_otsu(nir_band)
            water_mask = nir_band < thresh  # Water is dark in NIR
        else:
            logger.error("Cannot extract water bodies using NIR band from this image")
            return np.zeros((img_data.shape[0], img_data.shape[1]), dtype=np.uint8)
    
    else:
        logger.warning(f"Unknown water extraction method: {method}. Using NDWI.")
        ndwi = calculate_ndwi(img_data)
        water_mask = ndwi > threshold
    
    # Apply morphological operations to clean up the water mask
    water_mask = morphology.remove_small_objects(water_mask, min_size=100)
    water_mask = morphology.remove_small_holes(water_mask, area_threshold=100)
    
    # Convert to uint8 for compatibility
    water_mask = water_mask.astype(np.uint8)
    
    logger.info(f"Extracted water bodies using {method} method")
    return water_mask

def extract_urban_areas(img_data, method='texture'):
    """
    Extract urban areas from satellite imagery.
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Input satellite image as NumPy array
    method : str
        Method for urban area extraction. Options:
        - 'texture': Using texture analysis
        - 'ndbi': Normalized Difference Built-up Index
        - 'color': Using color-based segmentation
        
    Returns:
    --------
    numpy.ndarray
        Binary mask of urban areas (1 for urban, 0 for non-urban)
    """
    urban_mask = None
    
    if method == 'texture':
        # Use texture analysis for urban area detection
        # Convert to grayscale if needed
        if img_data.ndim == 3:
            if img_data.shape[2] >= 3:
                if img_data.dtype != np.uint8:
                    # Scale to 0-255
                    rgb = np.zeros((img_data.shape[0], img_data.shape[1], 3), dtype=np.uint8)
                    for i in range(3):
                        band = img_data[:, :, i]
                        min_val = band.min()
                        max_val = band.max()
                        if max_val > min_val:
                            rgb[:, :, i] = ((band - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                else:
                    rgb = img_data[:, :, :3]
                
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_data[:, :, 0].astype(np.uint8)
        else:
            gray = img_data.astype(np.uint8)
        
        # Compute edge density as a texture measure
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply Gaussian blur to get edge density
        edge_density = cv2.GaussianBlur(edges, (15, 15), 0)
        
        # Threshold edge density to get urban mask
        _, urban_mask = cv2.threshold(edge_density, 10, 1, cv2.THRESH_BINARY)
    
    elif method == 'ndbi':
        # Use NDBI (Normalized Difference Built-up Index) for urban detection
        # NDBI = (SWIR - NIR) / (SWIR + NIR)
        # Since not all images have SWIR, we may need to approximate
        if img_data.ndim == 3 and img_data.shape[2] >= 4:
            # Assuming standard channel order (R, G, B, NIR, ...)
            nir_band = img_data[:, :, 3].astype(np.float32)
            
            # If SWIR is available (usually band 5 or 6), use it
            if img_data.shape[2] >= 5:
                swir_band = img_data[:, :, 4].astype(np.float32)
            else:
                # If SWIR is not available, use red band as an approximation
                swir_band = img_data[:, :, 0].astype(np.float32)
            
            # Normalize if needed
            if nir_band.max() > 1.0 or swir_band.max() > 1.0:
                nir_band = nir_band / 255.0
                swir_band = swir_band / 255.0
            
            # Calculate NDBI
            ndbi = (swir_band - nir_band) / (swir_band + nir_band + 1e-10)
            
            # Threshold NDBI to get urban mask
            urban_mask = ndbi > 0.0
            
        else:
            logger.warning("Not enough bands for NDBI calculation. Falling back to texture method.")
            return extract_urban_areas(img_data, method='texture')
    
    elif method == 'color':
        # Use color-based segmentation for urban detection
        if img_data.ndim == 3 and img_data.shape[2] >= 3:
            # Convert to proper format for segmentation
            if img_data.dtype != np.uint8:
                # Scale to 0-255
                rgb = np.zeros((img_data.shape[0], img_data.shape[1], 3), dtype=np.uint8)
                for i in range(3):
                    band = img_data[:, :, i]
                    min_val = band.min()
                    max_val = band.max()
                    if max_val > min_val:
                        rgb[:, :, i] = ((band - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
            else:
                rgb = img_data[:, :, :3]
            
            # Convert to LAB color space
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            
            # Apply k-means clustering to segment the image
            pixels = lab.reshape(-1, 3)
            pixels = np.float32(pixels)
            
            # Define criteria and apply kmeans
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            k = 5  # Number of clusters
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert back to uint8 and reshape
            labels = labels.reshape(img_data.shape[0], img_data.shape[1])
            
            # Identify urban clusters based on their color characteristics
            # (This is a simplified approach - in practice, you would need to
            # analyze each cluster to determine which ones represent urban areas)
            urban_mask = np.zeros_like(labels, dtype=np.uint8)
            
            # As a simple heuristic, we'll consider the darker clusters as potentially urban
            for i in range(k):
                center = centers[i]
                # If the cluster center has low brightness, consider it urban
                if center[0] < 100:  # Low L value in LAB
                    urban_mask[labels == i] = 1
            
        else:
            logger.warning("Not enough bands for color-based segmentation. Falling back to texture method.")
            return extract_urban_areas(img_data, method='texture')
    
    else:
        logger.warning(f"Unknown urban extraction method: {method}. Using texture analysis.")
        return extract_urban_areas(img_data, method='texture')
    
    # Apply morphological operations to clean up the urban mask
    urban_mask = morphology.remove_small_objects(urban_mask.astype(bool), min_size=100)
    urban_mask = morphology.remove_small_holes(urban_mask.astype(bool), area_threshold=100)
    
    # Convert to uint8 for compatibility
    urban_mask = urban_mask.astype(np.uint8)
    
    logger.info(f"Extracted urban areas using {method} method")
    return urban_mask

def classify_landcover(img_data, num_classes=5):
    """
    Perform basic land cover classification on satellite imagery.
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Input satellite image as NumPy array
    num_classes : int
        Number of land cover classes to identify
        
    Returns:
    --------
    numpy.ndarray
        Land cover classification map
        
    Notes:
    ------
    This function uses a simple unsupervised K-means clustering approach.
    For more accurate land cover classification, supervised methods with
    training data should be used.
    """
    # Ensure the image is in the right format for classification
    if img_data.ndim < 3:
        logger.error("Image must have at least 3 dimensions for classification")
        return np.zeros_like(img_data, dtype=np.uint8)
    
    # Prepare data for clustering
    # Reshape to (pixels, features)
    h, w = img_data.shape[:2]
    n_bands = img_data.shape[2] if img_data.ndim == 3 else 1
    
    # Normalize data
    img_norm = np.zeros_like(img_data, dtype=np.float32)
    for i in range(n_bands):
        band = img_data[:, :, i].astype(np.float32)
        min_val = band.min()
        max_val = band.max()
        if max_val > min_val:
            img_norm[:, :, i] = (band - min_val) / (max_val - min_val)
        else:
            img_norm[:, :, i] = 0
    
    # Reshape for clustering
    pixels = img_norm.reshape(-1, n_bands)
    
    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_classes, None, criteria, 
                                   10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Reshape labels back to image dimensions
    classified = labels.reshape(h, w)
    
    logger.info(f"Classified land cover into {num_classes} classes")
    return classified.astype(np.uint8)

def detect_changes(img1, img2, method='difference'):
    """
    Detect changes between two satellite images.
    
    Parameters:
    -----------
    img1 : numpy.ndarray
        First satellite image
    img2 : numpy.ndarray
        Second satellite image
    method : str
        Change detection method to use. Options:
        - 'difference': Simple image differencing
        - 'ratio': Image ratioing
        - 'ndvi_diff': NDVI differencing
        
    Returns:
    --------
    numpy.ndarray
        Change map highlighting areas of change
        
    Notes:
    ------
    The two input images should be co-registered (aligned) and have the
    same dimensions and number of bands for accurate change detection.
    """
    # Check if images have compatible dimensions
    if img1.shape != img2.shape:
        logger.error("Images must have the same dimensions for change detection")
        return None
    
    change_map = None
    
    if method == 'difference':
        # Simple image differencing
        # Convert to float for calculations
        img1_float = img1.astype(np.float32)
        img2_float = img2.astype(np.float32)
        
        # Normalize if needed
        if img1_float.max() > 1.0 or img2_float.max() > 1.0:
            img1_float = img1_float / 255.0
            img2_float = img2_float / 255.0
        
        # Compute absolute difference
        if img1.ndim == 3:
            # For multi-band images, compute the mean difference across all bands
            diff = np.mean(np.abs(img1_float - img2_float), axis=2)
        else:
            # For single-band images
            diff = np.abs(img1_float - img2_float)
        
        # Normalize difference to 0-1 range
        if diff.max() > 0:
            diff = diff / diff.max()
        
        # Apply threshold to get binary change map
        change_map = diff > 0.2
    
    elif method == 'ratio':
        # Image ratioing
        # Convert to float for calculations
        img1_float = img1.astype(np.float32)
        img2_float = img2.astype(np.float32)
        
        # Add small constant to avoid division by zero
        img1_float = img1_float + 1e-10
        img2_float = img2_float + 1e-10
        
        # Compute ratio
        if img1.ndim == 3:
            # For multi-band images, compute the mean ratio across all bands
            ratio = np.mean(img1_float / img2_float, axis=2)
        else:
            # For single-band images
            ratio = img1_float / img2_float
        
        # Apply log transform to make distribution more symmetric
        log_ratio = np.log(ratio)
        
        # Normalize to 0-1 range
        min_val = log_ratio.min()
        max_val = log_ratio.max()
        if max_val > min_val:
            log_ratio_norm = (log_ratio - min_val) / (max_val - min_val)
        else:
            log_ratio_norm = np.zeros_like(log_ratio)
        
        # Apply threshold to get binary change map
        change_map = (log_ratio_norm < 0.4) | (log_ratio_norm > 0.6)
    
    elif method == 'ndvi_diff':
        # NDVI differencing
        ndvi1 = calculate_ndvi(img1)
        ndvi2 = calculate_ndvi(img2)
        
        # Compute absolute difference in NDVI
        ndvi_diff = np.abs(ndvi1 - ndvi2)
        
        # Apply threshold to get binary change map
        change_map = ndvi_diff > 0.2
    
    else:
        logger.warning(f"Unknown change detection method: {method}. Using difference method.")
        return detect_changes(img1, img2, method='difference')
    
    # Apply morphological operations to clean up the change map
    change_map = morphology.remove_small_objects(change_map, min_size=100)
    
    # Convert to uint8 for compatibility
    change_map = change_map.astype(np.uint8)
    
    logger.info(f"Detected changes using {method} method")
    return change_map

def segment_image(img_data, method='watershed', num_segments=100):
    """
    Segment satellite imagery into meaningful regions.
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Input satellite image as NumPy array
    method : str
        Segmentation method to use. Options:
        - 'watershed': Watershed segmentation
        - 'felzenszwalb': Felzenszwalb segmentation
        - 'slic': SLIC superpixel segmentation
    num_segments : int
        Approximate number of segments to create (for methods that support it)
        
    Returns:
    --------
    numpy.ndarray
        Segmentation map with segment labels
    """
    # Ensure the image is in the right format for segmentation
    if img_data.ndim == 3 and img_data.shape[2] >= 3:
        # Convert to proper format for segmentation
        if img_data.dtype != np.uint8:
            # Scale to 0-255
            rgb = np.zeros((img_data.shape[0], img_data.shape[1], 3), dtype=np.uint8)
            for i in range(3):
                band = img_data[:, :, i]
                min_val = band.min()
                max_val = band.max()
                if max_val > min_val:
                    rgb[:, :, i] = ((band - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
        else:
            rgb = img_data[:, :, :3]
    else:
        # For single-band images, create a 3-band image by duplication
        if img_data.ndim == 2:
            band = img_data
        else:  # img_data.ndim == 3 and img_data.shape[2] < 3
            band = img_data[:, :, 0]
            
        if band.dtype != np.uint8:
            # Scale to 0-255
            min_val = band.min()
            max_val = band.max()
            if max_val > min_val:
                band_8bit = ((band - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
            else:
                band_8bit = np.zeros_like(band, dtype=np.uint8)
        else:
            band_8bit = band
            
        rgb = np.stack([band_8bit] * 3, axis=2)
    
    # Apply segmentation based on selected method
    if method == 'watershed':
        # Watershed segmentation
        # Convert to grayscale for gradient calculation
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Compute gradient as segmentation marker
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
        
        # Apply threshold to gradient
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Compute markers
        # Apply distance transform
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        
        # Threshold distance transform to get markers
        _, markers = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 1, 0)
        markers = cv2.connectedComponents(markers.astype(np.uint8))[1]
        
        # Apply watershed algorithm
        cv2.watershed(rgb, markers.astype(np.int32))
        
        # Convert to uint8 for consistency
        segments = (markers + 1).astype(np.uint8)
        
    elif method == 'felzenszwalb':
        try:
            # Felzenszwalb segmentation from scikit-image
            from skimage.segmentation import felzenszwalb
            
            # Apply Felzenszwalb segmentation
            segments = felzenszwalb(rgb, scale=100, sigma=0.5, min_size=50)
            
            # Convert to uint8 for consistency (if possible)
            if segments.max() < 256:
                segments = segments.astype(np.uint8)
                
        except ImportError:
            logger.warning("scikit-image's felzenszwalb segmentation not available. Using watershed instead.")
            return segment_image(img_data, method='watershed', num_segments=num_segments)
    
    elif method == 'slic':
        try:
            # SLIC superpixel segmentation from scikit-image
            from skimage.segmentation import slic
            
            # Apply SLIC segmentation
            segments = slic(rgb, n_segments=num_segments, compactness=10, sigma=1)
            
            # Convert to uint8 for consistency (if possible)
            if segments.max() < 256:
                segments = segments.astype(np.uint8)
                
        except ImportError:
            logger.warning("scikit-image's SLIC segmentation not available. Using watershed instead.")
            return segment_image(img_data, method='watershed', num_segments=num_segments)
    
    else:
        logger.warning(f"Unknown segmentation method: {method}. Using watershed.")
        return segment_image(img_data, method='watershed', num_segments=num_segments)
    
    logger.info(f"Segmented image using {method} method")
    return segments

def extract_textures(img_data, window_size=11):
    """
    Extract texture features from satellite imagery.
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Input satellite image as NumPy array
    window_size : int
        Size of the window for texture calculation
        
    Returns:
    --------
    dict
        Dictionary of texture feature maps
        
    Notes:
    ------
    This function calculates several texture features based on Haralick textures:
    - Contrast
    - Homogeneity
    - Energy
    - Correlation
    """
    # Ensure the image is in the right format for texture calculation
    if img_data.ndim == 3:
        # Convert to grayscale
        if img_data.shape[2] >= 3:
            if img_data.dtype != np.uint8:
                # Scale to 0-255
                rgb = np.zeros((img_data.shape[0], img_data.shape[1], 3), dtype=np.uint8)
                for i in range(3):
                    band = img_data[:, :, i]
                    min_val = band.min()
                    max_val = band.max()
                    if max_val > min_val:
                        rgb[:, :, i] = ((band - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
            else:
                rgb = img_data[:, :, :3]
            
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_data[:, :, 0].astype(np.uint8)
    else:
        # Single band image
        if img_data.dtype != np.uint8:
            min_val = img_data.min()
            max_val = img_data.max()
            if max_val > min_val:
                gray = ((img_data - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
            else:
                gray = np.zeros_like(img_data, dtype=np.uint8)
        else:
            gray = img_data
    
    # Initialize texture feature maps
    h, w = gray.shape
    contrast = np.zeros((h, w), dtype=np.float32)
    homogeneity = np.zeros((h, w), dtype=np.float32)
    energy = np.zeros((h, w), dtype=np.float32)
    correlation = np.zeros((h, w), dtype=np.float32)
    
    # Calculate texture features using GLCM
    half_window = window_size // 2
    
    # Pad the image to handle edge pixels
    padded = cv2.copyMakeBorder(gray, half_window, half_window, half_window, half_window, 
                              cv2.BORDER_REFLECT)
    
    try:
        # Try to use mahotas for GLCM calculation if available
        import mahotas.features
        
        logger.info("Using mahotas for texture feature extraction")
        
        # Calculate Haralick features for each pixel
        for i in range(h):
            for j in range(w):
                # Extract window
                window = padded[i:i+window_size, j:j+window_size]
                
                # Calculate Haralick features
                haralick = mahotas.features.haralick(window, return_mean=True)
                
                # Assign texture features
                # Haralick features are:
                # 0: Angular Second Moment (Energy)
                # 1: Contrast
                # 2: Correlation
                # 3: Sum of Squares: Variance
                # 4: Inverse Difference Moment (Homogeneity)
                # ...
                
                energy[i, j] = haralick[0]
                contrast[i, j] = haralick[1]
                correlation[i, j] = haralick[2]
                homogeneity[i, j] = haralick[4]
    
    except ImportError:
        logger.warning("mahotas not available. Using alternative texture calculation methods.")
        
        # Use simpler texture measures as alternatives
        # Variance as a simple texture measure
        variance = ndimage.generic_filter(gray.astype(np.float32), np.var, size=window_size)
        
        # Gradient magnitude as an alternative to contrast
        dx = ndimage.sobel(gray, axis=0)
        dy = ndimage.sobel(gray, axis=1)
        gradient_mag = np.sqrt(dx**2 + dy**2)
        
        # Local entropy as another texture measure
        try:
            from skimage.filters.rank import entropy
            from skimage.morphology import disk
            
            entropy_map = entropy(gray, disk(half_window))
        except ImportError:
            # Fallback if skimage is not available
            entropy_map = ndimage.generic_filter(gray, lambda x: -np.sum(x/np.sum(x) * np.log2(x/np.sum(x) + 1e-10)), 
                                              size=window_size)
        
        # Normalize and assign to texture maps
        contrast = gradient_mag / gradient_mag.max() if gradient_mag.max() > 0 else gradient_mag
        energy = 1 - variance / variance.max() if variance.max() > 0 else variance
        correlation = np.zeros_like(contrast)  # We don't have a simple alternative
        homogeneity = 1 - entropy_map / entropy_map.max() if entropy_map.max() > 0 else entropy_map
    
    # Normalize texture features to 0-1 range
    def normalize(arr):
        min_val = arr.min()
        max_val = arr.max()
        if max_val > min_val:
            return (arr - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(arr)
    
    contrast = normalize(contrast)
    homogeneity = normalize(homogeneity)
    energy = normalize(energy)
    correlation = normalize(correlation)
    
    # Create dictionary of texture features
    textures = {
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation
    }
    
    logger.info("Extracted texture features")
    return textures
