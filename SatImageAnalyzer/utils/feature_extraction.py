# utils/feature_extraction.py
# Module for extracting features from satellite imagery

import numpy as np
import cv2
import logging
from skimage import feature, measure, segmentation
from scipy import ndimage
from sklearn.cluster import DBSCAN
import utils.image_processing as ip

# Configure logger
logger = logging.getLogger(__name__)

def extract_vegetation_indices(image, index_type='ndvi', red_band=2, nir_band=3):
    """
    Calculate various vegetation indices from satellite imagery.
    
    Args:
        image (numpy.ndarray): Multi-band satellite image
        index_type (str): Type of vegetation index to calculate
            Options: 'ndvi', 'evi', 'savi', 'ndwi', 'all'
        red_band (int): Index of the red band (0-based)
        nir_band (int): Index of the near-infrared band (0-based)
        
    Returns:
        dict or numpy.ndarray: Calculated vegetation index/indices
        
    Raises:
        ValueError: If the image doesn't have enough bands
    """
    if len(image.shape) < 3 or image.shape[2] <= max(red_band, nir_band):
        raise ValueError(f"Image doesn't have the required bands. Shape: {image.shape}")
    
    # Extract bands
    red = image[:, :, red_band].astype(np.float32)
    nir = image[:, :, nir_band].astype(np.float32)
    
    # Create a dictionary to store indices
    indices = {}
    
    # Calculate NDVI (Normalized Difference Vegetation Index)
    if index_type in ['ndvi', 'all']:
        # Avoid division by zero
        denominator = nir + red
        ndvi = np.zeros_like(red)
        valid_mask = denominator > 0
        ndvi[valid_mask] = (nir[valid_mask] - red[valid_mask]) / denominator[valid_mask]
        indices['ndvi'] = ndvi
    
    # Calculate EVI (Enhanced Vegetation Index)
    if index_type in ['evi', 'all'] and image.shape[2] > 3:  # Need blue band
        blue_band = 0  # Assuming blue is the first band
        blue = image[:, :, blue_band].astype(np.float32)
        
        # EVI = 2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1)
        denominator = nir + 6 * red - 7.5 * blue + 1
        evi = np.zeros_like(red)
        valid_mask = denominator > 0
        evi[valid_mask] = 2.5 * (nir[valid_mask] - red[valid_mask]) / denominator[valid_mask]
        indices['evi'] = evi
    
    # Calculate SAVI (Soil Adjusted Vegetation Index)
    if index_type in ['savi', 'all']:
        # SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        # where L is a soil brightness correction factor (usually 0.5)
        L = 0.5
        denominator = nir + red + L
        savi = np.zeros_like(red)
        valid_mask = denominator > 0
        savi[valid_mask] = ((nir[valid_mask] - red[valid_mask]) / denominator[valid_mask]) * (1 + L)
        indices['savi'] = savi
    
    # Calculate NDWI (Normalized Difference Water Index)
    if index_type in ['ndwi', 'all'] and image.shape[2] > 3:  # Need green band
        green_band = 1  # Assuming green is the second band
        green = image[:, :, green_band].astype(np.float32)
        
        # NDWI = (Green - NIR) / (Green + NIR)
        denominator = green + nir
        ndwi = np.zeros_like(green)
        valid_mask = denominator > 0
        ndwi[valid_mask] = (green[valid_mask] - nir[valid_mask]) / denominator[valid_mask]
        indices['ndwi'] = ndwi
    
    logger.debug(f"Calculated vegetation indices: {list(indices.keys())}")
    
    if index_type != 'all':
        return indices.get(index_type)
    return indices

def detect_urban_areas(image, method='ndbi'):
    """
    Detect urban areas in satellite imagery.
    
    Args:
        image (numpy.ndarray): Multi-band satellite image
        method (str): Detection method to use
            Options: 'ndbi', 'texture', 'clustering'
            
    Returns:
        numpy.ndarray: Binary mask of urban areas
    """
    if method == 'ndbi' and image.shape[2] >= 5:  # Need SWIR band
        # Calculate NDBI (Normalized Difference Built-up Index)
        # NDBI = (SWIR - NIR) / (SWIR + NIR)
        # Assuming SWIR is band 5 and NIR is band 4
        swir_band = 4
        nir_band = 3
        
        swir = image[:, :, swir_band].astype(np.float32)
        nir = image[:, :, nir_band].astype(np.float32)
        
        # Calculate NDBI
        denominator = swir + nir
        ndbi = np.zeros_like(swir)
        valid_mask = denominator > 0
        ndbi[valid_mask] = (swir[valid_mask] - nir[valid_mask]) / denominator[valid_mask]
        
        # Threshold to get urban areas (typically NDBI > 0 for urban)
        urban_mask = ndbi > 0
        
    elif method == 'texture':
        # Use texture analysis to detect urban areas
        # Urban areas typically have high texture variance
        
        # Convert to grayscale if it's a multi-band image
        if len(image.shape) == 3:
            if image.shape[2] >= 3:
                # Use first three bands as RGB
                gray = cv2.cvtColor(image[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image[:, :, 0].astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        # Calculate texture (variance in local neighborhood)
        texture = ndimage.generic_filter(gray, np.var, size=7)
        
        # Normalize and threshold
        texture = (texture - texture.min()) / (texture.max() - texture.min())
        urban_mask = texture > 0.3
        
    elif method == 'clustering':
        # Use clustering to identify urban areas
        # First calculate multiple features
        if len(image.shape) == 3 and image.shape[2] >= 3:
            # Calculate texture
            gray = cv2.cvtColor(image[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            texture = ndimage.generic_filter(gray, np.var, size=7)
            texture = (texture - texture.min()) / (texture.max() - texture.min())
            
            # Use band ratios as features
            band_ratio1 = image[:, :, 0] / (image[:, :, 1] + 0.01)
            band_ratio2 = image[:, :, 2] / (image[:, :, 1] + 0.01)
            
            # Stack features
            features = np.stack([
                image[:, :, 0], 
                image[:, :, 1], 
                image[:, :, 2], 
                texture,
                band_ratio1,
                band_ratio2
            ], axis=-1)
            
            # Reshape for clustering
            h, w, num_features = features.shape
            features_flat = features.reshape((h * w, num_features))
            
            # Apply clustering
            db = DBSCAN(eps=0.3, min_samples=10).fit(features_flat)
            labels = db.labels_.reshape((h, w))
            
            # Identify urban clusters (this requires some domain knowledge)
            # For simplicity, we'll use the clusters with the highest average texture
            unique_labels = np.unique(labels)
            max_texture_label = -1
            max_texture_value = -1
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                mask = labels == label
                avg_texture = np.mean(texture[mask])
                if avg_texture > max_texture_value:
                    max_texture_value = avg_texture
                    max_texture_label = label
            
            urban_mask = labels == max_texture_label
        else:
            raise ValueError("Clustering method requires at least 3 bands")
    else:
        raise ValueError(f"Unsupported urban detection method: {method}")
    
    logger.debug(f"Urban area detection completed using {method}")
    return urban_mask

def detect_water(image, method='ndwi'):
    """
    Detect water bodies in satellite imagery.
    
    Args:
        image (numpy.ndarray): Multi-band satellite image
        method (str): Detection method
            Options: 'ndwi', 'threshold', 'modified_ndwi'
            
    Returns:
        numpy.ndarray: Binary mask of water areas
    """
    if method == 'ndwi' and image.shape[2] >= 4:
        # NDWI = (Green - NIR) / (Green + NIR)
        # Assuming green is band 1 and NIR is band 3
        green_band = 1
        nir_band = 3
        
        green = image[:, :, green_band].astype(np.float32)
        nir = image[:, :, nir_band].astype(np.float32)
        
        # Calculate NDWI
        denominator = green + nir
        ndwi = np.zeros_like(green)
        valid_mask = denominator > 0
        ndwi[valid_mask] = (green[valid_mask] - nir[valid_mask]) / denominator[valid_mask]
        
        # Threshold NDWI to get water (typically NDWI > 0 indicates water)
        water_mask = ndwi > 0
        
    elif method == 'modified_ndwi' and image.shape[2] >= 6:
        # Modified NDWI = (Green - SWIR) / (Green + SWIR)
        # Assuming green is band 1 and SWIR is band 5
        green_band = 1
        swir_band = 5
        
        green = image[:, :, green_band].astype(np.float32)
        swir = image[:, :, swir_band].astype(np.float32)
        
        # Calculate Modified NDWI
        denominator = green + swir
        mndwi = np.zeros_like(green)
        valid_mask = denominator > 0
        mndwi[valid_mask] = (green[valid_mask] - swir[valid_mask]) / denominator[valid_mask]
        
        # Threshold Modified NDWI to get water
        water_mask = mndwi > 0
        
    elif method == 'threshold':
        # Simple thresholding on blue band
        if len(image.shape) == 3 and image.shape[2] >= 3:
            blue_band = 0  # Assuming blue is the first band
            blue = image[:, :, blue_band].astype(np.float32)
            
            # Normalize to 0-1 range
            blue_norm = (blue - blue.min()) / (blue.max() - blue.min())
            
            # Threshold (water typically has high blue values)
            water_mask = blue_norm > 0.7
            
            # Clean up with morphological operations
            water_mask = ndimage.binary_opening(water_mask, structure=np.ones((3, 3)))
        else:
            raise ValueError("Threshold method requires at least 3 bands")
    else:
        raise ValueError(f"Unsupported water detection method: {method}")
    
    logger.debug(f"Water detection completed using {method}")
    return water_mask

def detect_change(image1, image2, method='difference'):
    """
    Detect changes between two satellite images.
    
    Args:
        image1 (numpy.ndarray): First satellite image
        image2 (numpy.ndarray): Second satellite image (same dimensions as image1)
        method (str): Change detection method
            Options: 'difference', 'ratio', 'ndvi_diff', 'cvaps'
            
    Returns:
        numpy.ndarray: Change mask or change magnitude image
        
    Raises:
        ValueError: If images have different shapes
    """
    # Check if images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for change detection")
    
    if method == 'difference':
        # Simple absolute difference between images
        if len(image1.shape) == 3:
            # Multi-band images - calculate difference for each band
            diff = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
            
            # Average across bands
            change_magnitude = np.mean(diff, axis=2)
            
            # Normalize to 0-1
            if change_magnitude.max() > 0:
                change_magnitude = change_magnitude / change_magnitude.max()
            
            # Threshold to get binary change mask
            change_mask = change_magnitude > 0.2
        else:
            # Single-band images
            diff = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
            
            # Normalize and threshold
            if diff.max() > 0:
                diff = diff / diff.max()
            change_mask = diff > 0.2
            
    elif method == 'ratio':
        # Ratio-based change detection
        if len(image1.shape) == 3:
            # Calculate ratio for each band
            epsilon = 1e-10  # Small value to avoid division by zero
            ratio = image1.astype(np.float32) / (image2.astype(np.float32) + epsilon)
            
            # Average across bands
            ratio_avg = np.mean(ratio, axis=2)
            
            # Log-ratio (log values closer to 0 indicate no change)
            log_ratio = np.abs(np.log(ratio_avg + epsilon))
            
            # Normalize and threshold
            if log_ratio.max() > 0:
                log_ratio = log_ratio / log_ratio.max()
            change_mask = log_ratio > 0.2
        else:
            # Single-band images
            epsilon = 1e-10
            ratio = image1.astype(np.float32) / (image2.astype(np.float32) + epsilon)
            log_ratio = np.abs(np.log(ratio + epsilon))
            
            if log_ratio.max() > 0:
                log_ratio = log_ratio / log_ratio.max()
            change_mask = log_ratio > 0.2
            
    elif method == 'ndvi_diff':
        # NDVI difference-based change detection
        # Calculate NDVI for both images
        ndvi1 = ip.calculate_ndvi(image1)
        ndvi2 = ip.calculate_ndvi(image2)
        
        # Calculate absolute difference in NDVI
        ndvi_diff = np.abs(ndvi1 - ndvi2)
        
        # Threshold to get change mask
        change_mask = ndvi_diff > 0.2
        
    elif method == 'cvaps':
        # Change Vector Analysis in Posterior Space (simplified)
        # Convert to grayscale if multi-band
        if len(image1.shape) == 3:
            if image1.shape[2] >= 3:
                gray1 = cv2.cvtColor(image1[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(image2[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray1 = image1[:, :, 0].astype(np.uint8)
                gray2 = image2[:, :, 0].astype(np.uint8)
        else:
            gray1 = image1.astype(np.uint8)
            gray2 = image2.astype(np.uint8)
        
        # Calculate gradient magnitude for both images
        gradient1_x = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
        gradient1_y = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
        gradient1_mag = np.sqrt(gradient1_x**2 + gradient1_y**2)
        
        gradient2_x = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
        gradient2_y = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
        gradient2_mag = np.sqrt(gradient2_x**2 + gradient2_y**2)
        
        # Calculate gradient difference
        gradient_diff = np.abs(gradient1_mag - gradient2_mag)
        
        # Normalize and threshold
        if gradient_diff.max() > 0:
            gradient_diff = gradient_diff / gradient_diff.max()
        change_mask = gradient_diff > 0.3
    else:
        raise ValueError(f"Unsupported change detection method: {method}")
    
    logger.debug(f"Change detection completed using {method}")
    return change_mask

def extract_texture_features(image, method='glcm'):
    """
    Extract texture features from the image.
    
    Args:
        image (numpy.ndarray): Input image
        method (str): Texture feature extraction method
            Options: 'glcm', 'lbp', 'gabor'
            
    Returns:
        dict: Dictionary of texture features
    """
    # Convert to grayscale if it's a multi-band image
    if len(image.shape) == 3:
        if image.shape[2] >= 3:
            gray = cv2.cvtColor(image[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image[:, :, 0].astype(np.uint8)
    else:
        gray = image.astype(np.uint8)
    
    texture_features = {}
    
    if method == 'glcm':
        # Gray Level Co-occurrence Matrix features
        # Parameters for GLCM
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Calculate GLCM
        glcm = feature.graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
        
        # Extract properties
        texture_features['contrast'] = feature.graycoprops(glcm, 'contrast')[0, 0]
        texture_features['dissimilarity'] = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
        texture_features['homogeneity'] = feature.graycoprops(glcm, 'homogeneity')[0, 0]
        texture_features['energy'] = feature.graycoprops(glcm, 'energy')[0, 0]
        texture_features['correlation'] = feature.graycoprops(glcm, 'correlation')[0, 0]
        texture_features['ASM'] = feature.graycoprops(glcm, 'ASM')[0, 0]
        
    elif method == 'lbp':
        # Local Binary Pattern features
        radius = 3
        n_points = 8 * radius
        
        lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Calculate histogram of LBP
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        texture_features['lbp_histogram'] = hist
        
    elif method == 'gabor':
        # Gabor filter features
        # Generate Gabor filter kernels
        gabor_responses = []
        
        # Define orientations and frequencies
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        frequencies = [0.1, 0.2, 0.3, 0.4]
        
        for theta in orientations:
            for frequency in frequencies:
                # Apply Gabor filter
                filt_real, filt_imag = filters.gabor(gray, frequency=frequency, theta=theta, n_stds=3)
                
                # Calculate response magnitude
                gabor_mag = np.sqrt(filt_real**2 + filt_imag**2)
                
                # Store mean and standard deviation as features
                mean_mag = np.mean(gabor_mag)
                std_mag = np.std(gabor_mag)
                
                feature_name = f'gabor_mean_{frequency:.1f}_{theta:.2f}'
                texture_features[feature_name] = mean_mag
                
                feature_name = f'gabor_std_{frequency:.1f}_{theta:.2f}'
                texture_features[feature_name] = std_mag
    else:
        raise ValueError(f"Unsupported texture feature extraction method: {method}")
    
    logger.debug(f"Texture feature extraction completed using {method}")
    return texture_features

def extract_shape_features(image, mask=None):
    """
    Extract shape features from regions in the image.
    
    Args:
        image (numpy.ndarray): Input image
        mask (numpy.ndarray, optional): Binary mask defining regions of interest
            
    Returns:
        list: List of dictionaries containing shape features for each region
    """
    # Create segmentation mask if not provided
    if mask is None:
        # Convert to grayscale if it's a multi-band image
        if len(image.shape) == 3:
            if image.shape[2] >= 3:
                gray = cv2.cvtColor(image[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image[:, :, 0].astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        # Adaptive thresholding
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 2)
    
    # Label connected regions
    labeled_mask, num_labels = ndimage.label(mask)
    
    # Initialize list to store features
    shape_features = []
    
    # Measure properties of labeled regions
    props = measure.regionprops(labeled_mask)
    
    for prop in props:
        # Extract useful shape metrics
        features = {
            'area': prop.area,
            'perimeter': prop.perimeter,
            'eccentricity': prop.eccentricity,
            'equivalent_diameter': prop.equivalent_diameter,
            'major_axis_length': prop.major_axis_length,
            'minor_axis_length': prop.minor_axis_length,
            'orientation': prop.orientation,
            'solidity': prop.solidity,
            'centroid': prop.centroid
        }
        
        # Calculate shape factor (circularity)
        features['shape_factor'] = 4 * np.pi * features['area'] / (features['perimeter'] ** 2)
        
        shape_features.append(features)
    
    logger.debug(f"Extracted shape features for {len(shape_features)} regions")
    return shape_features

def object_detection(image, method='contour', min_size=100):
    """
    Detect objects in satellite imagery.
    
    Args:
        image (numpy.ndarray): Input image
        method (str): Object detection method
            Options: 'contour', 'blob', 'watershed'
        min_size (int): Minimum object size in pixels
            
    Returns:
        list: List of detected objects with coordinates
    """
    # Convert to grayscale if it's a multi-band image
    if len(image.shape) == 3:
        if image.shape[2] >= 3:
            gray = cv2.cvtColor(image[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image[:, :, 0].astype(np.uint8)
    else:
        gray = image.astype(np.uint8)
    
    detected_objects = []
    
    if method == 'contour':
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_size:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Get contour centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2
                
                # Store object information
                obj = {
                    'centroid': (cx, cy),
                    'bounding_box': (x, y, w, h),
                    'area': area,
                    'contour': contour.tolist()  # Convert to list for JSON serialization
                }
                detected_objects.append(obj)
                
    elif method == 'blob':
        # Set up the SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200
        
        # Filter by area
        params.filterByArea = True
        params.minArea = min_size
        
        # Filter by circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        
        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        
        # Filter by inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(gray)
        
        # Process each keypoint
        for keypoint in keypoints:
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
            r = int(keypoint.size / 2)
            
            obj = {
                'centroid': (x, y),
                'radius': r,
                'bounding_box': (x - r, y - r, 2 * r, 2 * r),
                'area': np.pi * r * r
            }
            detected_objects.append(obj)
            
    elif method == 'watershed':
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that background is not 0, but 1
        markers = markers + 1
        
        # Mark the unknown region with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        if len(image.shape) == 3 and image.shape[2] >= 3:
            markers = cv2.watershed(image[:, :, :3].astype(np.uint8), markers)
        else:
            # Convert grayscale to BGR for watershed
            img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(img_color, markers)
        
        # Process regions
        for label in np.unique(markers):
            # Skip background and watershed boundary
            if label <= 1:
                continue
                
            # Create a mask for the current label
            mask = np.zeros_like(gray)
            mask[markers == label] = 255
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_size:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Get contour centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = x + w // 2, y + h // 2
                    
                    # Store object information
                    obj = {
                        'centroid': (cx, cy),
                        'bounding_box': (x, y, w, h),
                        'area': area,
                        'contour': contour.tolist()
                    }
                    detected_objects.append(obj)
    else:
        raise ValueError(f"Unsupported object detection method: {method}")
    
    logger.debug(f"Detected {len(detected_objects)} objects using {method} method")
    return detected_objects
