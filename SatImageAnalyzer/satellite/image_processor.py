"""
Satellite Image Processor Module
--------------------------------

This module provides functions for preprocessing and enhancing satellite imagery
to improve its quality and make it more suitable for analysis.

Key functions:
- Atmospheric correction
- Radiometric calibration
- Noise reduction
- Contrast enhancement
- Image registration/alignment
- Cloud detection and masking

Dependencies:
- OpenCV (cv2): For image processing operations
- NumPy: For numerical operations
- SciPy: For scientific computing algorithms
"""

import numpy as np
import cv2
from scipy import ndimage
import logging

# Set up logging
logger = logging.getLogger(__name__)

def enhance_image(img_data, method='adaptive_equalization'):
    """
    Enhance the contrast and brightness of satellite imagery.
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Input satellite image as NumPy array
    method : str
        Enhancement method to use. Options:
        - 'histogram_equalization': Simple histogram equalization
        - 'adaptive_equalization': Contrast Limited Adaptive Histogram Equalization (CLAHE)
        - 'gamma_correction': Gamma correction
        
    Returns:
    --------
    numpy.ndarray
        Enhanced image
        
    Notes:
    ------
    Different enhancement methods work better for different types of imagery.
    CLAHE typically works well for most satellite imagery as it enhances local contrast.
    """
    # Make a copy to avoid modifying original
    enhanced = img_data.copy()
    
    # Check image dimensions and type
    if enhanced.ndim == 2:  # Single band
        # Convert to 8-bit if needed
        if enhanced.dtype != np.uint8:
            # Scale to 0-255 range
            min_val = enhanced.min()
            max_val = enhanced.max()
            if max_val > min_val:  # Avoid division by zero
                enhanced = ((enhanced - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
            else:
                enhanced = np.zeros_like(enhanced, dtype=np.uint8)
        
        if method == 'histogram_equalization':
            # Apply histogram equalization
            enhanced = cv2.equalizeHist(enhanced)
            
        elif method == 'adaptive_equalization':
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced)
            
        elif method == 'gamma_correction':
            # Apply gamma correction with gamma=1.2
            gamma = 1.2
            lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(np.uint8)
            enhanced = cv2.LUT(enhanced, lookup_table)
            
        else:
            logger.warning(f"Unknown enhancement method: {method}. Using original image.")
    
    elif enhanced.ndim == 3:  # Multi-band image
        # Determine number of bands
        if enhanced.shape[2] > 3:
            logger.info(f"Image has {enhanced.shape[2]} bands, processing first 3 for visualization.")
            visualization_bands = enhanced[:, :, :3].copy()
        else:
            visualization_bands = enhanced.copy()
        
        # Convert to 8-bit if needed
        if visualization_bands.dtype != np.uint8:
            # Scale each band to 0-255 range
            for i in range(visualization_bands.shape[2]):
                band = visualization_bands[:, :, i]
                min_val = band.min()
                max_val = band.max()
                if max_val > min_val:  # Avoid division by zero
                    visualization_bands[:, :, i] = ((band - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                else:
                    visualization_bands[:, :, i] = np.zeros_like(band, dtype=np.uint8)
        
        if method == 'histogram_equalization':
            # Apply histogram equalization to each band
            for i in range(visualization_bands.shape[2]):
                visualization_bands[:, :, i] = cv2.equalizeHist(visualization_bands[:, :, i])
                
        elif method == 'adaptive_equalization':
            # Apply CLAHE to each band
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            for i in range(visualization_bands.shape[2]):
                visualization_bands[:, :, i] = clahe.apply(visualization_bands[:, :, i])
                
        elif method == 'gamma_correction':
            # Apply gamma correction with gamma=1.2
            gamma = 1.2
            lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(np.uint8)
            for i in range(visualization_bands.shape[2]):
                visualization_bands[:, :, i] = cv2.LUT(visualization_bands[:, :, i], lookup_table)
                
        else:
            logger.warning(f"Unknown enhancement method: {method}. Using original image.")
        
        # If original had more than 3 bands, keep those bands unchanged
        if enhanced.shape[2] > 3:
            enhanced[:, :, :3] = visualization_bands
        else:
            enhanced = visualization_bands
    
    else:
        logger.error(f"Unexpected image dimensions: {enhanced.ndim}. Cannot enhance.")
        return img_data
    
    logger.info(f"Enhanced image using {method} method")
    return enhanced

def remove_noise(img_data, method='gaussian'):
    """
    Apply noise reduction to satellite imagery.
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Input satellite image as NumPy array
    method : str
        Noise reduction method to use. Options:
        - 'gaussian': Gaussian blur
        - 'median': Median filter
        - 'bilateral': Bilateral filter (edge-preserving)
        
    Returns:
    --------
    numpy.ndarray
        Noise-reduced image
    """
    # Make a copy to avoid modifying original
    denoised = img_data.copy()
    
    # Convert to 8-bit if needed for OpenCV compatibility
    original_dtype = denoised.dtype
    if original_dtype != np.uint8 and denoised.dtype != np.float32:
        # Scale to 0-255 range
        if denoised.ndim == 2 or denoised.ndim == 3:  # Single band or multi-band
            scaled = np.zeros_like(denoised, dtype=np.uint8)
            
            if denoised.ndim == 2:  # Single band
                min_val = denoised.min()
                max_val = denoised.max()
                if max_val > min_val:  # Avoid division by zero
                    scaled = ((denoised - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                else:
                    scaled = np.zeros_like(denoised, dtype=np.uint8)
            else:  # Multi-band
                for i in range(denoised.shape[2]):
                    band = denoised[:, :, i]
                    min_val = band.min()
                    max_val = band.max()
                    if max_val > min_val:  # Avoid division by zero
                        scaled[:, :, i] = ((band - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                    else:
                        scaled[:, :, i] = np.zeros_like(band, dtype=np.uint8)
            
            denoised = scaled
    
    # Apply noise reduction based on method
    if method == 'gaussian':
        if denoised.ndim == 2:  # Single band
            denoised = cv2.GaussianBlur(denoised, (5, 5), 0)
        else:  # Multi-band
            for i in range(denoised.shape[2]):
                denoised[:, :, i] = cv2.GaussianBlur(denoised[:, :, i], (5, 5), 0)
    
    elif method == 'median':
        if denoised.ndim == 2:  # Single band
            denoised = cv2.medianBlur(denoised, 5)
        else:  # Multi-band
            for i in range(denoised.shape[2]):
                denoised[:, :, i] = cv2.medianBlur(denoised[:, :, i], 5)
    
    elif method == 'bilateral':
        if denoised.ndim == 2:  # Single band
            denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        else:  # Multi-band
            for i in range(denoised.shape[2]):
                denoised[:, :, i] = cv2.bilateralFilter(denoised[:, :, i], 9, 75, 75)
    
    else:
        logger.warning(f"Unknown noise reduction method: {method}. Using original image.")
        return img_data
    
    # Convert back to original data type if needed
    if original_dtype != np.uint8:
        # This is a simplification - proper conversion would depend on the specific case
        denoised = denoised.astype(original_dtype)
    
    logger.info(f"Applied {method} noise reduction")
    return denoised

def detect_edges(img_data, method='canny', low_threshold=50, high_threshold=150):
    """
    Detect edges in satellite imagery.
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Input satellite image as NumPy array
    method : str
        Edge detection method to use. Options:
        - 'canny': Canny edge detector
        - 'sobel': Sobel operator
    low_threshold : int
        Lower threshold for the hysteresis procedure (for Canny)
    high_threshold : int
        Higher threshold for the hysteresis procedure (for Canny)
        
    Returns:
    --------
    numpy.ndarray
        Binary edge map
    """
    # Handle multi-band images
    if img_data.ndim == 3:
        # Convert to grayscale for edge detection
        if img_data.shape[2] >= 3:  # RGB or more bands
            # Use the first three bands as RGB
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
            
            # Convert RGB to grayscale
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:  # Just use the first band
            gray = img_data[:, :, 0].astype(np.uint8)
    else:  # Already grayscale
        gray = img_data.astype(np.uint8)
    
    # Apply edge detection
    if method == 'canny':
        edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    elif method == 'sobel':
        # Apply Sobel operator in x and y directions
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute the magnitude of gradients
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize and threshold
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    
    else:
        logger.warning(f"Unknown edge detection method: {method}. Using Canny edge detector.")
        edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    logger.info(f"Applied {method} edge detection")
    return edges

def register_images(ref_img, target_img, method='feature'):
    """
    Register (align) a target image to a reference image.
    
    Parameters:
    -----------
    ref_img : numpy.ndarray
        Reference image
    target_img : numpy.ndarray
        Target image to align with the reference
    method : str
        Registration method to use. Options:
        - 'feature': Feature-based registration using ORB features
        - 'ecc': Enhanced Correlation Coefficient Maximization
        
    Returns:
    --------
    numpy.ndarray
        Registered target image aligned with the reference
    """
    # Ensure images are in the right format for registration
    if ref_img.ndim == 3 and ref_img.shape[2] > 1:
        # Use first band or convert to grayscale for feature detection
        if ref_img.dtype != np.uint8:
            ref_gray = ((ref_img[:, :, 0] - ref_img[:, :, 0].min()) * 255 / 
                        (ref_img[:, :, 0].max() - ref_img[:, :, 0].min())).astype(np.uint8)
        else:
            ref_gray = cv2.cvtColor(ref_img[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        ref_gray = ref_img.astype(np.uint8)
    
    if target_img.ndim == 3 and target_img.shape[2] > 1:
        if target_img.dtype != np.uint8:
            target_gray = ((target_img[:, :, 0] - target_img[:, :, 0].min()) * 255 / 
                          (target_img[:, :, 0].max() - target_img[:, :, 0].min())).astype(np.uint8)
        else:
            target_gray = cv2.cvtColor(target_img[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        target_gray = target_img.astype(np.uint8)
    
    # Create output placeholders
    height, width = ref_img.shape[:2]
    registered_img = np.zeros_like(target_img)
    
    if method == 'feature':
        # Feature-based registration using ORB features
        try:
            # Initialize the ORB detector
            orb = cv2.ORB_create(nfeatures=500)
            
            # Find keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(ref_gray, None)
            kp2, des2 = orb.detectAndCompute(target_gray, None)
            
            # If no features found, return original image
            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                logger.warning("Not enough features found for registration. Returning original image.")
                return target_img
            
            # Create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            # Match descriptors
            matches = bf.match(des1, des2)
            
            # Sort them in order of their distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Use only top matches
            good_matches = matches[:min(50, len(matches))]
            
            # Extract location of good matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            # Use homography to warp image
            if H is not None:
                registered_img = cv2.warpPerspective(target_img, H, (width, height))
            else:
                logger.warning("Homography estimation failed. Returning original image.")
                return target_img
                
        except Exception as e:
            logger.error(f"Feature-based registration failed: {str(e)}. Returning original image.")
            return target_img
    
    elif method == 'ecc':
        # Enhanced Correlation Coefficient Maximization
        try:
            # Define the motion model
            warp_mode = cv2.MOTION_HOMOGRAPHY
            
            # Define 3x3 warp matrix
            warp_matrix = np.eye(3, 3, dtype=np.float32)
            
            # Specify the number of iterations
            number_of_iterations = 5000
            
            # Specify the threshold of the increment in the correlation coefficient
            termination_eps = 1e-8
            
            # Define termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                       number_of_iterations, termination_eps)
            
            # Run the ECC algorithm
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, target_gray, warp_matrix, warp_mode, criteria)
            
            # Warp the target image
            registered_img = cv2.warpPerspective(
                target_img, warp_matrix, (width, height), 
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                
        except Exception as e:
            logger.error(f"ECC registration failed: {str(e)}. Returning original image.")
            return target_img
    
    else:
        logger.warning(f"Unknown registration method: {method}. Returning original image.")
        return target_img
    
    logger.info(f"Applied {method} image registration")
    return registered_img

def detect_clouds(img_data):
    """
    Detect clouds in satellite imagery.
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Input satellite image as NumPy array
        
    Returns:
    --------
    numpy.ndarray
        Binary cloud mask (1 for cloud, 0 for non-cloud)
        
    Notes:
    ------
    This is a simplified cloud detection algorithm based on brightness
    thresholding. For more accurate cloud detection, specialized algorithms
    like Fmask (Function of mask) should be used, which would require
    additional spectral bands and metadata.
    """
    # Simple cloud detection based on brightness thresholding
    # This is a simplified approach - production systems would use more sophisticated methods
    
    # If multiband image, use the blue band (usually band 2) and NIR
    if img_data.ndim == 3 and img_data.shape[2] >= 4:
        # Assuming band order: R, G, B, NIR, ...
        blue_band = img_data[:, :, 2].astype(np.float32)
        nir_band = img_data[:, :, 3].astype(np.float32)
        
        # Normalize bands to 0-1 range if needed
        if blue_band.max() > 1.0:
            blue_band = blue_band / 255.0
        if nir_band.max() > 1.0:
            nir_band = nir_band / 255.0
        
        # Cloud detection using blue/NIR ratio and thresholding
        # Clouds are bright in the blue band and relatively dark in NIR
        blue_nir_ratio = np.divide(blue_band, nir_band, 
                                  out=np.ones_like(blue_band), 
                                  where=nir_band != 0)
        
        # Apply threshold to ratio
        cloud_mask = (blue_nir_ratio > 1.0) & (blue_band > 0.7)
        
    elif img_data.ndim == 3 and img_data.shape[2] >= 3:
        # If we only have RGB, use blue band and brightness
        blue_band = img_data[:, :, 2].astype(np.float32)
        
        # Normalize to 0-1 range if needed
        if blue_band.max() > 1.0:
            blue_band = blue_band / 255.0
        
        # Calculate overall brightness
        if img_data.shape[2] >= 3:
            # Use RGB bands
            red_band = img_data[:, :, 0].astype(np.float32)
            green_band = img_data[:, :, 1].astype(np.float32)
            
            if red_band.max() > 1.0:
                red_band = red_band / 255.0
            if green_band.max() > 1.0:
                green_band = green_band / 255.0
            
            brightness = (red_band + green_band + blue_band) / 3.0
        else:
            brightness = blue_band
        
        # Clouds are typically bright
        cloud_mask = (blue_band > 0.8) & (brightness > 0.7)
        
    else:  # Single band
        # For single band, use simple brightness thresholding
        img_float = img_data.astype(np.float32)
        
        # Normalize to 0-1 range if needed
        if img_float.max() > 1.0:
            img_float = img_float / 255.0
        
        # Simple threshold for bright areas
        cloud_mask = img_float > 0.8
    
    # Apply morphological operations to clean up the mask
    cloud_mask = ndimage.binary_opening(cloud_mask, structure=np.ones((3, 3)))
    cloud_mask = ndimage.binary_closing(cloud_mask, structure=np.ones((5, 5)))
    
    # Convert to uint8 for compatibility
    cloud_mask = cloud_mask.astype(np.uint8)
    
    logger.info("Generated cloud mask")
    return cloud_mask

def pansharpening(ms_img, pan_img):
    """
    Pansharpen a multispectral image using a higher-resolution panchromatic band.
    
    Parameters:
    -----------
    ms_img : numpy.ndarray
        Multispectral image (lower resolution)
    pan_img : numpy.ndarray
        Panchromatic image (higher resolution)
        
    Returns:
    --------
    numpy.ndarray
        Pansharpened multispectral image
        
    Notes:
    ------
    This function implements a simple Brovey transform for pansharpening.
    The input images should be registered and resampled to the same dimensions.
    """
    # Check if images have the right dimensions
    if ms_img.ndim != 3:
        logger.error("Multispectral image must have 3 dimensions (H, W, Bands)")
        return ms_img
    
    if pan_img.ndim != 2:
        logger.error("Panchromatic image must have 2 dimensions (H, W)")
        return ms_img
    
    # Check if images have the same spatial dimensions
    if ms_img.shape[0] != pan_img.shape[0] or ms_img.shape[1] != pan_img.shape[1]:
        logger.error("Multispectral and panchromatic images must have the same dimensions")
        return ms_img
    
    # Convert to float for calculations
    ms_float = ms_img.astype(np.float32)
    pan_float = pan_img.astype(np.float32)
    
    # Normalize to 0-1 range if needed
    if ms_float.max() > 1.0:
        ms_float = ms_float / 255.0
    if pan_float.max() > 1.0:
        pan_float = pan_float / 255.0
    
    # Implement Brovey transform
    # Create an output array with the same shape as multispectral
    pansharpened = np.zeros_like(ms_float)
    
    # Calculate the intensity of multispectral image
    # Typically the mean of all bands or a weighted combination
    intensity = np.mean(ms_float, axis=2)
    
    # Avoid division by zero
    intensity[intensity == 0] = 1.0e-10
    
    # Apply the transform for each band
    for i in range(ms_float.shape[2]):
        pansharpened[:, :, i] = (ms_float[:, :, i] / intensity) * pan_float
    
    # Clip values to valid range
    pansharpened = np.clip(pansharpened, 0.0, 1.0)
    
    # Convert back to original data type
    if ms_img.dtype == np.uint8:
        pansharpened = (pansharpened * 255).astype(np.uint8)
    else:
        pansharpened = pansharpened.astype(ms_img.dtype)
    
    logger.info("Applied pansharpening using Brovey transform")
    return pansharpened
