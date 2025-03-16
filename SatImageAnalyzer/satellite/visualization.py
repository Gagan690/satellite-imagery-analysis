"""
Satellite Image Visualization Module
-----------------------------------

This module provides functions for visualizing satellite imagery and analysis results.
It includes tools for creating composite RGB images, colormapping single-band data,
and generating comparison visualizations.

Key functions:
- Creating RGB composites from multi-band satellite data
- Visualizing indices like NDVI with appropriate color scales
- Generating classification maps
- Creating comparative visualizations

Dependencies:
- Matplotlib: For plotting and color mapping
- NumPy: For array operations
- OpenCV (cv2): For image processing operations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Patch
import matplotlib.cm as cm
import cv2
import os
import logging
from io import BytesIO
from base64 import b64encode

# Set up logging
logger = logging.getLogger(__name__)

def create_rgb_composite(img_data, rgb_bands=(0, 1, 2), stretch='linear', percentile=(2, 98)):
    """
    Create an RGB composite image from multi-band satellite data.
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Input satellite image as NumPy array
    rgb_bands : tuple
        Indices of the bands to use for RGB channels
    stretch : str
        Type of stretch to apply:
        - 'linear': Linear stretch between min and max
        - 'percentile': Linear stretch between specified percentiles
        - 'equalize': Histogram equalization
    percentile : tuple
        Lower and upper percentiles for percentile stretch
        
    Returns:
    --------
    numpy.ndarray
        RGB composite image (0-255 uint8)
        
    Notes:
    ------
    This function creates a 3-band RGB composite from a multi-band satellite
    image by selecting specified bands and applying a contrast stretch.
    """
    # Check if image has enough bands
    if img_data.ndim < 3 or img_data.shape[2] <= max(rgb_bands):
        logger.error(f"Image does not have enough bands. Required: {max(rgb_bands)+1}, Found: {img_data.shape[2] if img_data.ndim == 3 else 1}")
        # Fall back to single band visualization if possible
        if img_data.ndim == 2:
            # Single band grayscale
            logger.info("Falling back to grayscale visualization")
            gray = img_data.copy()
            # Apply stretch
            if stretch == 'linear':
                min_val = gray.min()
                max_val = gray.max()
                if max_val > min_val:
                    gray = ((gray - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                else:
                    gray = np.zeros_like(gray, dtype=np.uint8)
            elif stretch == 'percentile':
                p_low, p_high = np.percentile(gray, percentile)
                gray = np.clip(gray, p_low, p_high)
                gray = ((gray - p_low) * 255 / (p_high - p_low)).astype(np.uint8)
            elif stretch == 'equalize':
                if gray.dtype != np.uint8:
                    # Scale to 0-255 for histogram equalization
                    min_val = gray.min()
                    max_val = gray.max()
                    if max_val > min_val:
                        gray = ((gray - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                    else:
                        gray = np.zeros_like(gray, dtype=np.uint8)
                gray = cv2.equalizeHist(gray)
            
            # Convert grayscale to RGB
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            return rgb
        elif img_data.ndim == 3 and img_data.shape[2] == 1:
            # Single band as 3D array
            logger.info("Falling back to grayscale visualization")
            gray = img_data[:, :, 0].copy()
            # Apply stretch
            if stretch == 'linear':
                min_val = gray.min()
                max_val = gray.max()
                if max_val > min_val:
                    gray = ((gray - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                else:
                    gray = np.zeros_like(gray, dtype=np.uint8)
            elif stretch == 'percentile':
                p_low, p_high = np.percentile(gray, percentile)
                gray = np.clip(gray, p_low, p_high)
                gray = ((gray - p_low) * 255 / (p_high - p_low)).astype(np.uint8)
            elif stretch == 'equalize':
                if gray.dtype != np.uint8:
                    # Scale to 0-255 for histogram equalization
                    min_val = gray.min()
                    max_val = gray.max()
                    if max_val > min_val:
                        gray = ((gray - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                    else:
                        gray = np.zeros_like(gray, dtype=np.uint8)
                gray = cv2.equalizeHist(gray)
            
            # Convert grayscale to RGB
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            return rgb
        else:
            # Return empty RGB image if everything fails
            logger.error("Cannot create RGB composite from the provided image")
            return np.zeros((img_data.shape[0], img_data.shape[1], 3), dtype=np.uint8)
    
    # Extract RGB bands
    r_band = img_data[:, :, rgb_bands[0]].astype(np.float32)
    g_band = img_data[:, :, rgb_bands[1]].astype(np.float32)
    b_band = img_data[:, :, rgb_bands[2]].astype(np.float32)
    
    # Apply stretch
    rgb = np.zeros((img_data.shape[0], img_data.shape[1], 3), dtype=np.uint8)
    
    if stretch == 'linear':
        # Linear stretch each band
        for i, band in enumerate([r_band, g_band, b_band]):
            min_val = band.min()
            max_val = band.max()
            if max_val > min_val:
                rgb[:, :, i] = ((band - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
    
    elif stretch == 'percentile':
        # Percentile stretch each band
        for i, band in enumerate([r_band, g_band, b_band]):
            p_low, p_high = np.percentile(band, percentile)
            band_clipped = np.clip(band, p_low, p_high)
            rgb[:, :, i] = ((band_clipped - p_low) * 255 / (p_high - p_low)).astype(np.uint8)
    
    elif stretch == 'equalize':
        # Histogram equalization each band
        for i, band in enumerate([r_band, g_band, b_band]):
            # Convert to uint8 for histogram equalization
            if band.max() > 255 or band.min() < 0 or band.dtype != np.uint8:
                min_val = band.min()
                max_val = band.max()
                if max_val > min_val:
                    band_8bit = ((band - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                else:
                    band_8bit = np.zeros_like(band, dtype=np.uint8)
            else:
                band_8bit = band.astype(np.uint8)
            
            rgb[:, :, i] = cv2.equalizeHist(band_8bit)
    
    else:
        logger.warning(f"Unknown stretch method: {stretch}. Using linear stretch.")
        # Fall back to linear stretch
        for i, band in enumerate([r_band, g_band, b_band]):
            min_val = band.min()
            max_val = band.max()
            if max_val > min_val:
                rgb[:, :, i] = ((band - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
    
    logger.info(f"Created RGB composite from bands {rgb_bands} with {stretch} stretch")
    return rgb

def visualize_ndvi(ndvi_data, colormap='RdYlGn', vmin=-1, vmax=1):
    """
    Create a visualization of NDVI data with an appropriate colormap.
    
    Parameters:
    -----------
    ndvi_data : numpy.ndarray
        NDVI data as NumPy array, typically with values from -1 to 1
    colormap : str
        Name of the matplotlib colormap to use
    vmin : float
        Minimum value for colormap scaling
    vmax : float
        Maximum value for colormap scaling
        
    Returns:
    --------
    numpy.ndarray
        Colorized NDVI image (0-255 uint8 RGB)
        
    Notes:
    ------
    This function applies a colormap to NDVI data to create a visualization
    that highlights vegetation patterns. The RdYlGn (Red-Yellow-Green) colormap
    is often used for NDVI, with red representing low values (no vegetation)
    and green representing high values (dense vegetation).
    """
    # Convert NaN values to a valid number within range
    ndvi_valid = np.copy(ndvi_data)
    ndvi_valid[np.isnan(ndvi_valid)] = vmin
    
    # Clip values to specified range
    ndvi_clipped = np.clip(ndvi_valid, vmin, vmax)
    
    # Create a normalized colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(colormap)
    
    # Apply colormap
    ndvi_colored = cmap(norm(ndvi_clipped))
    
    # Convert to 8-bit RGB
    ndvi_rgb = (ndvi_colored[:, :, :3] * 255).astype(np.uint8)
    
    logger.info(f"Created NDVI visualization with {colormap} colormap")
    return ndvi_rgb

def visualize_classification(class_data, class_names=None, colormap='tab20'):
    """
    Create a visualization of classification results with an appropriate colormap.
    
    Parameters:
    -----------
    class_data : numpy.ndarray
        Classification data as NumPy array, with integer class labels
    class_names : list or None
        List of class names for the legend
    colormap : str
        Name of the matplotlib colormap to use
        
    Returns:
    --------
    numpy.ndarray
        Colorized classification image (0-255 uint8 RGB)
        
    Notes:
    ------
    This function applies a discrete colormap to classification data to
    create a visualization that distinguishes different classes.
    """
    # Determine number of classes
    num_classes = np.max(class_data) + 1
    
    # Create a discrete colormap
    cmap = plt.get_cmap(colormap, num_classes)
    
    # Normalize data to 0-1 range for colormap
    norm = colors.Normalize(vmin=0, vmax=num_classes-1)
    
    # Apply colormap
    class_colored = cmap(norm(class_data))
    
    # Convert to 8-bit RGB
    class_rgb = (class_colored[:, :, :3] * 255).astype(np.uint8)
    
    logger.info(f"Created classification visualization with {colormap} colormap")
    return class_rgb

def create_legend_image(class_data, class_names, colormap='tab20'):
    """
    Create a legend image for classification results.
    
    Parameters:
    -----------
    class_data : numpy.ndarray
        Classification data as NumPy array, with integer class labels
    class_names : list
        List of class names for the legend
    colormap : str
        Name of the matplotlib colormap to use
        
    Returns:
    --------
    bytes
        PNG image of the legend as base64 encoded string
        
    Notes:
    ------
    This function creates a standalone legend image that can be used
    alongside the classification visualization.
    """
    # Determine number of classes
    num_classes = np.max(class_data) + 1
    
    # Create a discrete colormap
    cmap = plt.get_cmap(colormap, num_classes)
    
    # Create a figure for the legend
    plt.figure(figsize=(3, num_classes * 0.5))
    
    # Create legend patches
    legend_elements = []
    for i in range(num_classes):
        color = cmap(i / (num_classes - 1) if num_classes > 1 else 0)
        if i < len(class_names):
            label = class_names[i]
        else:
            label = f"Class {i}"
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label))
    
    # Create the legend
    plt.legend(handles=legend_elements, loc='center')
    plt.axis('off')
    
    # Save the legend as a PNG
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Encode the image as base64
    buf.seek(0)
    img_str = f"data:image/png;base64,{b64encode(buf.read()).decode('utf-8')}"
    
    return img_str

def visualize_binary_mask(mask, background='original', bg_img=None):
    """
    Create a visualization of a binary mask overlaid on an image.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        Binary mask (0 or 1) as NumPy array
    background : str
        Type of background to use:
        - 'original': Use the original image as background
        - 'white': Use a white background
        - 'black': Use a black background
    bg_img : numpy.ndarray or None
        Original image to use as background if background='original'
        
    Returns:
    --------
    numpy.ndarray
        Visualization of the binary mask (0-255 uint8 RGB)
        
    Notes:
    ------
    This function creates a visualization of a binary mask by coloring
    the masked areas (1) in a semi-transparent color, and the background (0)
    either as the original image or a solid color.
    """
    # Ensure mask is binary
    mask_binary = mask > 0
    
    # Create output image
    h, w = mask.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    if background == 'original' and bg_img is not None:
        # Use original image as background
        if bg_img.ndim == 2:
            # Convert grayscale to RGB
            bg_rgb = cv2.cvtColor(bg_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif bg_img.ndim == 3 and bg_img.shape[2] >= 3:
            # Use RGB channels
            bg_rgb = bg_img[:, :, :3].copy()
            
            # Convert to uint8 if needed
            if bg_rgb.dtype != np.uint8:
                # Scale to 0-255
                for i in range(3):
                    band = bg_rgb[:, :, i]
                    min_val = band.min()
                    max_val = band.max()
                    if max_val > min_val:
                        bg_rgb[:, :, i] = ((band - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                    else:
                        bg_rgb[:, :, i] = np.zeros_like(band, dtype=np.uint8)
        else:
            # Fallback to white background
            logger.warning("Background image has incompatible dimensions. Using white background.")
            bg_rgb = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Copy background
        vis = bg_rgb.copy()
        
        # Overlay mask in red color (semi-transparent)
        red_mask = np.zeros_like(vis)
        red_mask[mask_binary] = [255, 0, 0]  # Red color
        
        # Blend original image with mask
        alpha = 0.5  # Transparency factor
        vis = cv2.addWeighted(vis, 1, red_mask, alpha, 0)
        
    elif background == 'white':
        # White background with colored mask
        vis.fill(255)  # White background
        vis[mask_binary] = [255, 0, 0]  # Red mask
        
    elif background == 'black':
        # Black background with colored mask
        vis.fill(0)  # Black background
        vis[mask_binary] = [255, 0, 0]  # Red mask
        
    else:
        logger.warning(f"Unknown background type: {background}. Using white background.")
        vis.fill(255)  # White background
        vis[mask_binary] = [255, 0, 0]  # Red mask
    
    logger.info(f"Created binary mask visualization with {background} background")
    return vis

def create_comparison_visualization(images, titles=None, cmaps=None):
    """
    Create a comparison visualization of multiple images.
    
    Parameters:
    -----------
    images : list
        List of images to compare
    titles : list or None
        List of titles for each image
    cmaps : list or None
        List of colormaps for each image (for grayscale images)
        
    Returns:
    --------
    numpy.ndarray
        Comparison visualization (0-255 uint8 RGB)
        
    Notes:
    ------
    This function creates a side-by-side comparison of multiple images,
    with optional titles and colormaps.
    """
    # Validate inputs
    if not images:
        logger.error("No images provided for comparison")
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    num_images = len(images)
    
    if titles is None:
        titles = [f"Image {i+1}" for i in range(num_images)]
    else:
        # Ensure we have the right number of titles
        titles = titles[:num_images]
        while len(titles) < num_images:
            titles.append(f"Image {len(titles)+1}")
    
    if cmaps is None:
        cmaps = [None] * num_images
    else:
        # Ensure we have the right number of cmaps
        cmaps = cmaps[:num_images]
        while len(cmaps) < num_images:
            cmaps.append(None)
    
    # Determine layout based on number of images
    if num_images <= 3:
        rows, cols = 1, num_images
    elif num_images <= 6:
        rows, cols = 2, (num_images + 1) // 2
    else:
        rows, cols = 3, (num_images + 2) // 3
    
    # Create a figure for the comparison
    plt.figure(figsize=(cols * 5, rows * 5))
    
    # Display each image
    for i, (img, title, cmap) in enumerate(zip(images, titles, cmaps)):
        plt.subplot(rows, cols, i + 1)
        
        # Convert image to proper format for display
        if img.ndim == 2:
            # Grayscale image
            plt.imshow(img, cmap=cmap)
        elif img.ndim == 3 and img.shape[2] == 1:
            # Single-channel image as 3D array
            plt.imshow(img[:, :, 0], cmap=cmap)
        elif img.ndim == 3 and img.shape[2] == 3:
            # RGB image
            plt.imshow(img)
        elif img.ndim == 3 and img.shape[2] > 3:
            # Multi-band image, display as RGB
            rgb = create_rgb_composite(img)
            plt.imshow(rgb)
        
        plt.title(title)
        plt.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert plot to image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    
    # Convert to numpy array
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    logger.info(f"Created comparison visualization with {num_images} images")
    return img

def save_comparison(img1, img2, output_path):
    """
    Create and save a side-by-side comparison of two images.
    
    Parameters:
    -----------
    img1 : numpy.ndarray
        First image
    img2 : numpy.ndarray
        Second image
    output_path : str
        Path to save the comparison image
        
    Returns:
    --------
    str
        Path to the saved comparison image
    """
    # Create a comparison visualization
    comparison = create_comparison_visualization([img1, img2], 
                                              titles=["Original", "Processed"],
                                              cmaps=['gray', 'gray'])
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the comparison image
    cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    logger.info(f"Saved comparison visualization to {output_path}")
    return output_path

def save_ndvi_visualization(ndvi_data, output_path):
    """
    Create and save a visualization of NDVI data.
    
    Parameters:
    -----------
    ndvi_data : numpy.ndarray
        NDVI data as NumPy array
    output_path : str
        Path to save the NDVI visualization
        
    Returns:
    --------
    str
        Path to the saved NDVI visualization
    """
    # Create an NDVI visualization
    ndvi_vis = visualize_ndvi(ndvi_data)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the NDVI visualization
    cv2.imwrite(output_path, cv2.cvtColor(ndvi_vis, cv2.COLOR_RGB2BGR))
    
    logger.info(f"Saved NDVI visualization to {output_path}")
    return output_path

def save_binary_mask(mask_data, output_path):
    """
    Create and save a visualization of a binary mask.
    
    Parameters:
    -----------
    mask_data : numpy.ndarray
        Binary mask data as NumPy array
    output_path : str
        Path to save the binary mask visualization
        
    Returns:
    --------
    str
        Path to the saved binary mask visualization
    """
    # Create a binary mask visualization
    mask_vis = visualize_binary_mask(mask_data, background='white')
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the binary mask visualization
    cv2.imwrite(output_path, cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))
    
    logger.info(f"Saved binary mask visualization to {output_path}")
    return output_path

def create_heatmap(data, colormap='jet', alpha=0.7, background=None):
    """
    Create a heatmap visualization of data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data to visualize as a heatmap
    colormap : str
        Name of the matplotlib colormap to use
    alpha : float
        Transparency of the heatmap (0-1)
    background : numpy.ndarray or None
        Background image to overlay the heatmap on
        
    Returns:
    --------
    numpy.ndarray
        Heatmap visualization (0-255 uint8 RGB)
    """
    # Normalize data to 0-1 range
    norm_data = data.copy().astype(np.float32)
    min_val = norm_data.min()
    max_val = norm_data.max()
    
    if max_val > min_val:
        norm_data = (norm_data - min_val) / (max_val - min_val)
    else:
        norm_data.fill(0)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(norm_data)
    
    # Convert to uint8
    heatmap_rgb = (heatmap[:, :, :3] * 255).astype(np.uint8)
    
    # If background is provided, overlay the heatmap
    if background is not None:
        # Ensure background is RGB
        if background.ndim == 2:
            bg_rgb = cv2.cvtColor(background.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif background.ndim == 3 and background.shape[2] >= 3:
            bg_rgb = background[:, :, :3].copy()
            
            # Convert to uint8 if needed
            if bg_rgb.dtype != np.uint8:
                # Scale to 0-255
                for i in range(3):
                    band = bg_rgb[:, :, i]
                    min_val = band.min()
                    max_val = band.max()
                    if max_val > min_val:
                        bg_rgb[:, :, i] = ((band - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                    else:
                        bg_rgb[:, :, i] = np.zeros_like(band, dtype=np.uint8)
        else:
            # Fallback to black background
            logger.warning("Background image has incompatible dimensions. Using black background.")
            bg_rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
        
        # Create alpha channel for blending
        alpha_channel = (heatmap[:, :, 3] * alpha * 255).astype(np.uint8)
        
        # Blend background with heatmap
        result = bg_rgb.copy()
        for c in range(3):
            result[:, :, c] = (bg_rgb[:, :, c] * (255 - alpha_channel) + 
                             heatmap_rgb[:, :, c] * alpha_channel) // 255
        
        return result
    else:
        return heatmap_rgb

def create_spectral_plot(img_data, x, y, output_path=None):
    """
    Create a spectral profile plot for a pixel in the image.
    
    Parameters:
    -----------
    img_data : numpy.ndarray
        Multispectral image data
    x : int
        X coordinate of the pixel
    y : int
        Y coordinate of the pixel
    output_path : str or None
        Path to save the plot (if None, return as image)
        
    Returns:
    --------
    numpy.ndarray or str
        Spectral plot as image or path to saved plot
    """
    # Check if coordinates are valid
    h, w = img_data.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        logger.error(f"Invalid coordinates ({x}, {y}) for image of shape {img_data.shape}")
        return None
    
    # Check if image has multiple bands
    if img_data.ndim < 3:
        logger.error("Image must have multiple bands for spectral plot")
        return None
    
    # Extract the spectral profile
    n_bands = img_data.shape[2]
    band_values = []
    
    for i in range(n_bands):
        band_values.append(float(img_data[y, x, i]))
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_bands + 1), band_values, 'o-', linewidth=2, markersize=8)
    plt.grid(True)
    plt.xlabel('Band Number')
    plt.ylabel('Pixel Value')
    plt.title(f'Spectral Profile at Pixel ({x}, {y})')
    plt.tight_layout()
    
    if output_path:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save the plot
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved spectral plot to {output_path}")
        return output_path
    else:
        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert to numpy array
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        logger.info("Created spectral plot as image")
        return img
