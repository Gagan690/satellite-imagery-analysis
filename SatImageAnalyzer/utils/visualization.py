# utils/visualization.py
# Module for visualizing satellite imagery and analysis results

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle, Circle, Polygon
import logging
import io
import base64
from skimage import color, segmentation

# Configure logger
logger = logging.getLogger(__name__)

def normalize_image(image):
    """
    Normalize image data for display.
    
    Args:
        image (numpy.ndarray): Input image data
        
    Returns:
        numpy.ndarray: Normalized image (0-1 range)
    """
    # Check if the image is already normalized
    if image.min() >= 0 and image.max() <= 1:
        return image
        
    # Handle different data types
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    else:
        # General case for float arrays
        image_min = image.min()
        image_max = image.max()
        
        if image_max > image_min:
            return (image - image_min) / (image_max - image_min)
        else:
            return np.zeros_like(image, dtype=np.float32)

def create_composite_visualization(image, bands=[0, 1, 2], stretch=True, clip_percentile=2):
    """
    Create an RGB composite image for visualization.
    
    Args:
        image (numpy.ndarray): Multi-band satellite image
        bands (list): Indices of bands to use for RGB channels
        stretch (bool): Apply contrast stretching
        clip_percentile (float): Percentile for contrast stretching
        
    Returns:
        numpy.ndarray: RGB composite image for visualization
    """
    if len(image.shape) < 3 or image.shape[2] <= max(bands):
        raise ValueError(f"Image doesn't have the required bands. Shape: {image.shape}")
    
    # Extract specified bands
    if len(bands) == 3:
        rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
        for i, band_idx in enumerate(bands):
            rgb[:, :, i] = image[:, :, band_idx].astype(np.float32)
    else:
        # Grayscale if only one band specified
        rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
        for i in range(3):
            rgb[:, :, i] = image[:, :, bands[0]].astype(np.float32)
    
    # Apply contrast stretching if requested
    if stretch:
        for i in range(3):
            if np.any(rgb[:, :, i]):  # Only stretch if band has data
                p_low, p_high = np.percentile(rgb[:, :, i], [clip_percentile, 100 - clip_percentile])
                rgb[:, :, i] = np.clip(rgb[:, :, i], p_low, p_high)
                
                # Normalize to 0-1 range
                min_val = np.min(rgb[:, :, i])
                max_val = np.max(rgb[:, :, i])
                
                if max_val > min_val:
                    rgb[:, :, i] = (rgb[:, :, i] - min_val) / (max_val - min_val)
                else:
                    rgb[:, :, i] = np.zeros_like(rgb[:, :, i])
    else:
        # Simple normalization
        rgb = normalize_image(rgb)
    
    logger.debug(f"Created composite visualization using bands {bands}")
    return rgb

def visualize_ndvi(ndvi, colormap='RdYlGn', title='NDVI', figure_size=(10, 8)):
    """
    Create a visualization of NDVI data.
    
    Args:
        ndvi (numpy.ndarray): NDVI data (-1 to 1 range)
        colormap (str): Matplotlib colormap name
        title (str): Plot title
        figure_size (tuple): Figure size in inches
        
    Returns:
        str: Base64-encoded image
    """
    plt.figure(figsize=figure_size)
    
    # Create a colormap that's centered on 0
    vmin, vmax = -1, 1
    
    # Plot NDVI with colormap
    plt.imshow(ndvi, cmap=colormap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(label='NDVI Value')
    
    # Add title and labels
    plt.title(title)
    plt.axis('off')
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    # Convert to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('ascii')
    
    logger.debug("NDVI visualization created")
    return img_str

def visualize_classification(classification_map, num_classes=None, colormap='viridis', title='Classification', figure_size=(10, 8)):
    """
    Create a visualization of classification results.
    
    Args:
        classification_map (numpy.ndarray): Classification map with class labels
        num_classes (int): Number of classes (if None, determined from data)
        colormap (str): Matplotlib colormap name
        title (str): Plot title
        figure_size (tuple): Figure size in inches
        
    Returns:
        str: Base64-encoded image
    """
    plt.figure(figsize=figure_size)
    
    # Determine number of classes if not provided
    if num_classes is None:
        num_classes = len(np.unique(classification_map))
    
    # Create a colormap with distinct colors for each class
    cmap = plt.get_cmap(colormap, num_classes)
    
    # Plot classification map
    plt.imshow(classification_map, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(ticks=np.arange(num_classes), label='Class')
    
    # Add title and turn off axis
    plt.title(title)
    plt.axis('off')
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    # Convert to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('ascii')
    
    logger.debug(f"Classification visualization created with {num_classes} classes")
    return img_str

def visualize_change_detection(change_mask, original_image=None, title='Change Detection', figure_size=(10, 8)):
    """
    Create a visualization of change detection results.
    
    Args:
        change_mask (numpy.ndarray): Binary mask showing changes
        original_image (numpy.ndarray, optional): Original image for overlay
        title (str): Plot title
        figure_size (tuple): Figure size in inches
        
    Returns:
        str: Base64-encoded image
    """
    plt.figure(figsize=figure_size)
    
    if original_image is not None:
        # Create RGB composite if not already
        if len(original_image.shape) == 3 and original_image.shape[2] >= 3:
            rgb = normalize_image(original_image[:, :, :3])
        elif len(original_image.shape) == 3:
            # Use first band for all channels
            rgb = np.zeros((original_image.shape[0], original_image.shape[1], 3))
            for i in range(3):
                rgb[:, :, i] = normalize_image(original_image[:, :, 0])
        else:
            # Single band image
            rgb = np.zeros((original_image.shape[0], original_image.shape[1], 3))
            for i in range(3):
                rgb[:, :, i] = normalize_image(original_image)
                
        # Display original image
        plt.imshow(rgb)
        
        # Create a mask for highlighting changes
        # Red for changes
        mask_color = np.zeros((*change_mask.shape, 4), dtype=np.float32)  # RGBA
        mask_color[change_mask > 0] = [1, 0, 0, 0.5]  # Red with alpha=0.5
        
        # Overlay change mask
        plt.imshow(mask_color)
    else:
        # Just show the change mask
        plt.imshow(change_mask, cmap='hot')
    
    # Add title and turn off axis
    plt.title(title)
    plt.axis('off')
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    # Convert to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('ascii')
    
    logger.debug("Change detection visualization created")
    return img_str

def visualize_features(image, features, feature_type='vegetation', figure_size=(12, 10)):
    """
    Create visualizations for various types of features.
    
    Args:
        image (numpy.ndarray): Original image
        features: Features to visualize (format depends on feature_type)
        feature_type (str): Type of features ('vegetation', 'urban', 'water', 'objects')
        figure_size (tuple): Figure size in inches
        
    Returns:
        str: Base64-encoded image
    """
    plt.figure(figsize=figure_size)
    
    # Create base image for visualization
    if len(image.shape) == 3 and image.shape[2] >= 3:
        base_img = normalize_image(image[:, :, :3])
    elif len(image.shape) == 3:
        # Use first band
        base_img = normalize_image(image[:, :, 0])
        # Convert to RGB for overlay
        base_img = np.stack([base_img] * 3, axis=-1)
    else:
        # Single band
        base_img = normalize_image(image)
        # Convert to RGB for overlay
        base_img = np.stack([base_img] * 3, axis=-1)
    
    if feature_type == 'vegetation':
        # Assuming features is a dict of vegetation indices
        if isinstance(features, dict):
            # Create subplot for each vegetation index
            n_indices = len(features)
            fig, axes = plt.subplots(1, n_indices, figsize=figure_size)
            
            # Handle case with just one subplot
            if n_indices == 1:
                axes = [axes]
            
            for i, (index_name, index_data) in enumerate(features.items()):
                if index_name == 'ndvi':
                    cmap = 'RdYlGn'
                    vmin, vmax = -1, 1
                else:
                    cmap = 'viridis'
                    vmin, vmax = None, None
                
                im = axes[i].imshow(index_data, cmap=cmap, vmin=vmin, vmax=vmax)
                axes[i].set_title(index_name.upper())
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
        else:
            # Single vegetation index (e.g., NDVI)
            plt.imshow(features, cmap='RdYlGn', vmin=-1, vmax=1)
            plt.colorbar(label='Index Value')
            plt.title('Vegetation Index')
    
    elif feature_type == 'urban':
        # Assuming features is a binary mask of urban areas
        # Display base image
        plt.imshow(base_img)
        
        # Create a mask for urban areas (semi-transparent overlay)
        mask = np.zeros((*features.shape, 4), dtype=np.float32)  # RGBA
        mask[features > 0] = [1, 1, 0, 0.5]  # Yellow with alpha=0.5
        
        # Overlay urban mask
        plt.imshow(mask)
        plt.title('Urban Areas')
    
    elif feature_type == 'water':
        # Assuming features is a binary mask of water bodies
        # Display base image
        plt.imshow(base_img)
        
        # Create a mask for water areas (semi-transparent overlay)
        mask = np.zeros((*features.shape, 4), dtype=np.float32)  # RGBA
        mask[features > 0] = [0, 0, 1, 0.5]  # Blue with alpha=0.5
        
        # Overlay water mask
        plt.imshow(mask)
        plt.title('Water Bodies')
    
    elif feature_type == 'objects':
        # Assuming features is a list of detected objects
        # Display base image
        plt.imshow(base_img)
        
        # Draw bounding boxes or outlines for detected objects
        ax = plt.gca()
        for obj in features:
            if 'bounding_box' in obj:
                x, y, w, h = obj['bounding_box']
                rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                
            elif 'contour' in obj:
                contour = np.array(obj['contour'])
                poly = Polygon(contour[:, 0, :], edgecolor='r', facecolor='none')
                ax.add_patch(poly)
                
            if 'centroid' in obj:
                cx, cy = obj['centroid']
                circle = Circle((cx, cy), 3, color='r')
                ax.add_patch(circle)
        
        plt.title(f'Object Detection ({len(features)} objects)')
    
    else:
        plt.imshow(base_img)
        plt.title('Original Image')
    
    plt.axis('off')
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    # Convert to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('ascii')
    
    logger.debug(f"Feature visualization created for {feature_type}")
    return img_str

def visualize_image_histogram(image, title='Image Histogram', figure_size=(10, 6)):
    """
    Create a visualization of the image histogram.
    
    Args:
        image (numpy.ndarray): Input image
        title (str): Plot title
        figure_size (tuple): Figure size in inches
        
    Returns:
        str: Base64-encoded image
    """
    plt.figure(figsize=figure_size)
    
    # Handle different image types
    if len(image.shape) == 3 and image.shape[2] >= 3:
        # RGB or multi-band image
        colors = ['r', 'g', 'b']
        
        for i, color in enumerate(colors[:min(3, image.shape[2])]):
            hist, bins = np.histogram(image[:, :, i].flatten(), bins=256, range=(0, 1 if image.dtype == np.float32 or image.dtype == np.float64 else 255))
            plt.plot(bins[:-1], hist, color=color, alpha=0.7)
    else:
        # Grayscale image
        hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 1 if image.dtype == np.float32 or image.dtype == np.float64 else 255))
        plt.plot(bins[:-1], hist, color='k', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    # Convert to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('ascii')
    
    logger.debug("Image histogram visualization created")
    return img_str

def visualize_segmentation(image, segments, title='Image Segmentation', figure_size=(12, 8)):
    """
    Create a visualization of image segmentation.
    
    Args:
        image (numpy.ndarray): Original image
        segments (numpy.ndarray): Segmentation labels
        title (str): Plot title
        figure_size (tuple): Figure size in inches
        
    Returns:
        str: Base64-encoded image
    """
    plt.figure(figsize=figure_size)
    
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)
    
    # Display original image
    if len(image.shape) == 3 and image.shape[2] >= 3:
        ax1.imshow(normalize_image(image[:, :, :3]))
    else:
        ax1.imshow(normalize_image(image), cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Display segmentation
    ax2.imshow(segmentation.mark_boundaries(
        normalize_image(image[:, :, :3]) if len(image.shape) == 3 and image.shape[2] >= 3 else np.stack([normalize_image(image)] * 3, axis=-1),
        segments, color=(1, 0, 0), outline_color=(1, 1, 0)))
    ax2.set_title(f'Segmentation ({len(np.unique(segments))} segments)')
    ax2.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    # Convert to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('ascii')
    
    logger.debug(f"Segmentation visualization created with {len(np.unique(segments))} segments")
    return img_str

def create_multi_band_plot(image, bands_to_show=None, title='Multi-band View', figure_size=(15, 10)):
    """
    Create a visualization showing multiple bands of an image.
    
    Args:
        image (numpy.ndarray): Multi-band image
        bands_to_show (list): List of band indices to show (if None, shows all)
        title (str): Plot title
        figure_size (tuple): Figure size in inches
        
    Returns:
        str: Base64-encoded image
    """
    if len(image.shape) < 3:
        # Single band image
        plt.figure(figsize=figure_size)
        plt.imshow(normalize_image(image), cmap='gray')
        plt.title('Single Band Image')
        plt.axis('off')
    else:
        # Multi-band image
        if bands_to_show is None:
            bands_to_show = list(range(min(12, image.shape[2])))  # Show up to 12 bands
        
        n_bands = len(bands_to_show)
        
        # Calculate grid dimensions
        n_cols = min(4, n_bands)
        n_rows = (n_bands + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figure_size)
        
        # Make axes accessible for both 1D and 2D grid
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each band
        for i, band_idx in enumerate(bands_to_show):
            row, col = i // n_cols, i % n_cols
            
            if band_idx < image.shape[2]:
                band_data = image[:, :, band_idx]
                axes[row, col].imshow(normalize_image(band_data), cmap='gray')
                axes[row, col].set_title(f'Band {band_idx}')
            
            axes[row, col].axis('off')
        
        # Hide any unused subplots
        for i in range(len(bands_to_show), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
            axes[row, col].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    # Convert to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('ascii')
    
    logger.debug(f"Multi-band visualization created with {len(bands_to_show) if bands_to_show else 1} bands")
    return img_str

def create_comparison_visualization(images, titles=None, figure_size=(15, 10)):
    """
    Create a side-by-side comparison of multiple images.
    
    Args:
        images (list): List of images to compare
        titles (list): List of titles for each image
        figure_size (tuple): Figure size in inches
        
    Returns:
        str: Base64-encoded image
    """
    n_images = len(images)
    
    # Set default titles if not provided
    if titles is None:
        titles = [f'Image {i+1}' for i in range(n_images)]
    
    # Create subplot grid
    fig, axes = plt.subplots(1, n_images, figsize=figure_size)
    
    # Handle case with single image
    if n_images == 1:
        axes = [axes]
    
    # Display each image
    for i, image in enumerate(images):
        if len(image.shape) == 3 and image.shape[2] >= 3:
            axes[i].imshow(normalize_image(image[:, :, :3]))
        else:
            axes[i].imshow(normalize_image(image), cmap='gray')
        
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    # Convert to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('ascii')
    
    logger.debug(f"Comparison visualization created with {n_images} images")
    return img_str
