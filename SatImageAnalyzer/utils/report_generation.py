# utils/report_generation.py
# Module for generating analysis reports

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from jinja2 import Template
import io
import base64

# Configure logger
logger = logging.getLogger(__name__)

def generate_basic_statistics(image):
    """
    Generate basic statistical information about an image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        dict: Dictionary of statistics
    """
    stats = {}
    
    # Image dimensions
    stats['dimensions'] = {
        'height': image.shape[0],
        'width': image.shape[1],
        'bands': image.shape[2] if len(image.shape) > 2 else 1
    }
    
    # Per-band statistics
    stats['band_stats'] = []
    
    if len(image.shape) > 2:
        # Multi-band image
        for i in range(image.shape[2]):
            band_data = image[:, :, i]
            band_stats = {
                'band': i,
                'min': float(np.min(band_data)),
                'max': float(np.max(band_data)),
                'mean': float(np.mean(band_data)),
                'std': float(np.std(band_data)),
                'median': float(np.median(band_data))
            }
            stats['band_stats'].append(band_stats)
    else:
        # Single-band image
        band_stats = {
            'band': 0,
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'median': float(np.median(image))
        }
        stats['band_stats'].append(band_stats)
    
    # Overall statistics
    stats['overall'] = {
        'data_type': str(image.dtype),
        'size_bytes': image.nbytes,
        'size_mb': image.nbytes / (1024 * 1024)
    }
    
    logger.debug("Generated basic image statistics")
    return stats

def generate_vegetation_statistics(ndvi_data):
    """
    Generate statistics about vegetation from NDVI data.
    
    Args:
        ndvi_data (numpy.ndarray): NDVI data (-1 to 1 range)
        
    Returns:
        dict: Dictionary of vegetation statistics
    """
    stats = {}
    
    # Basic NDVI statistics
    stats['ndvi'] = {
        'min': float(np.min(ndvi_data)),
        'max': float(np.max(ndvi_data)),
        'mean': float(np.mean(ndvi_data)),
        'std': float(np.std(ndvi_data)),
        'median': float(np.median(ndvi_data))
    }
    
    # Calculate vegetation coverage
    # NDVI > 0.3 typically indicates vegetation
    veg_mask = ndvi_data > 0.3
    sparse_veg_mask = (ndvi_data > 0.1) & (ndvi_data <= 0.3)
    non_veg_mask = ndvi_data <= 0.1
    
    total_pixels = ndvi_data.size
    veg_pixels = np.sum(veg_mask)
    sparse_veg_pixels = np.sum(sparse_veg_mask)
    non_veg_pixels = np.sum(non_veg_mask)
    
    stats['coverage'] = {
        'vegetation_percent': float(veg_pixels / total_pixels * 100),
        'sparse_vegetation_percent': float(sparse_veg_pixels / total_pixels * 100),
        'non_vegetation_percent': float(non_veg_pixels / total_pixels * 100)
    }
    
    # NDVI distribution
    hist, bins = np.histogram(ndvi_data, bins=20, range=(-1, 1))
    stats['histogram'] = {
        'counts': hist.tolist(),
        'bin_edges': bins.tolist()
    }
    
    logger.debug("Generated vegetation statistics from NDVI data")
    return stats

def generate_classification_statistics(classification_map):
    """
    Generate statistics about classification results.
    
    Args:
        classification_map (numpy.ndarray): Classification map with class labels
        
    Returns:
        dict: Dictionary of classification statistics
    """
    stats = {}
    
    # Count pixels in each class
    unique_classes, counts = np.unique(classification_map, return_counts=True)
    
    # Convert to list of dictionaries for easier handling in templates
    class_stats = []
    total_pixels = classification_map.size
    
    for cls, count in zip(unique_classes, counts):
        class_stats.append({
            'class': int(cls),
            'count': int(count),
            'percent': float(count / total_pixels * 100)
        })
    
    stats['classes'] = class_stats
    stats['num_classes'] = len(unique_classes)
    stats['total_pixels'] = int(total_pixels)
    
    logger.debug(f"Generated classification statistics with {len(unique_classes)} classes")
    return stats

def generate_change_detection_statistics(change_mask):
    """
    Generate statistics about detected changes.
    
    Args:
        change_mask (numpy.ndarray): Binary mask showing changes
        
    Returns:
        dict: Dictionary of change statistics
    """
    stats = {}
    
    # Count changed pixels
    total_pixels = change_mask.size
    changed_pixels = np.sum(change_mask)
    unchanged_pixels = total_pixels - changed_pixels
    
    stats['pixels'] = {
        'total': int(total_pixels),
        'changed': int(changed_pixels),
        'unchanged': int(unchanged_pixels)
    }
    
    stats['percentages'] = {
        'changed': float(changed_pixels / total_pixels * 100),
        'unchanged': float(unchanged_pixels / total_pixels * 100)
    }
    
    # Analyze change distribution
    # Find connected components in the change mask
    from scipy import ndimage
    labeled_mask, num_features = ndimage.label(change_mask)
    
    if num_features > 0:
        # Calculate area of each component
        component_areas = []
        for i in range(1, num_features + 1):
            area = np.sum(labeled_mask == i)
            component_areas.append(int(area))
        
        stats['components'] = {
            'count': int(num_features),
            'min_area': int(min(component_areas)),
            'max_area': int(max(component_areas)),
            'mean_area': float(np.mean(component_areas)),
            'median_area': float(np.median(component_areas))
        }
    else:
        stats['components'] = {
            'count': 0
        }
    
    logger.debug(f"Generated change detection statistics with {stats['components']['count']} components")
    return stats

def generate_feature_statistics(features, feature_type):
    """
    Generate statistics about extracted features.
    
    Args:
        features: Extracted features
        feature_type (str): Type of features
        
    Returns:
        dict: Dictionary of feature statistics
    """
    stats = {}
    
    if feature_type == 'texture':
        # Texture features are typically returned as a dictionary of measurements
        if isinstance(features, dict):
            # Copy the features to stats
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    stats[key] = value.tolist() if value.size < 100 else {
                        'min': float(np.min(value)),
                        'max': float(np.max(value)),
                        'mean': float(np.mean(value)),
                        'std': float(np.std(value))
                    }
                else:
                    stats[key] = value
    
    elif feature_type == 'objects':
        # Object detection results - count objects and summarize properties
        if isinstance(features, list):
            stats['count'] = len(features)
            
            # Collect areas and positions
            if len(features) > 0:
                areas = []
                centroids_x = []
                centroids_y = []
                
                for obj in features:
                    if 'area' in obj:
                        areas.append(obj['area'])
                    
                    if 'centroid' in obj:
                        centroids_x.append(obj['centroid'][0])
                        centroids_y.append(obj['centroid'][1])
                
                # Calculate area statistics
                if areas:
                    stats['area'] = {
                        'min': float(min(areas)),
                        'max': float(max(areas)),
                        'mean': float(np.mean(areas)),
                        'median': float(np.median(areas)),
                        'std': float(np.std(areas))
                    }
                
                # Calculate spatial distribution
                if centroids_x and centroids_y:
                    stats['distribution'] = {
                        'x_min': float(min(centroids_x)),
                        'x_max': float(max(centroids_x)),
                        'y_min': float(min(centroids_y)),
                        'y_max': float(max(centroids_y)),
                        'x_mean': float(np.mean(centroids_x)),
                        'y_mean': float(np.mean(centroids_y))
                    }
    
    elif feature_type in ['vegetation', 'urban', 'water']:
        # Binary masks - calculate coverage
        if isinstance(features, np.ndarray):
            total_pixels = features.size
            positive_pixels = np.sum(features)
            negative_pixels = total_pixels - positive_pixels
            
            stats['coverage'] = {
                'total_pixels': int(total_pixels),
                'positive_pixels': int(positive_pixels),
                'negative_pixels': int(negative_pixels),
                'positive_percent': float(positive_pixels / total_pixels * 100),
                'negative_percent': float(negative_pixels / total_pixels * 100)
            }
    
    logger.debug(f"Generated {feature_type} feature statistics")
    return stats

def create_html_report(image_path, results, metadata=None, output_path=None):
    """
    Create an HTML report of the analysis results.
    
    Args:
        image_path (str): Path to the original image
        results (dict): Dictionary of analysis results
        metadata (dict): Image metadata
        output_path (str): Path to save the HTML report
        
    Returns:
        str: HTML report content
    """
    # Define the HTML template
    template_str = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Satellite Image Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1, h2, h3 { color: #2c3e50; }
            .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .section { margin-bottom: 30px; border: 1px solid #eee; padding: 20px; border-radius: 5px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .image-container { margin: 20px 0; text-align: center; }
            .image-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 3px; }
            .footer { margin-top: 50px; text-align: center; font-size: 0.8em; color: #777; }
            .chart-container { width: 100%; height: 400px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Satellite Imagery Analysis Report</h1>
            <p>Generated on {{ current_date }}</p>
            <p>Image: {{ image_name }}</p>
        </div>
        
        {% if metadata %}
        <div class="section">
            <h2>Image Metadata</h2>
            <table>
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
                {% for key, value in metadata.items() %}
                <tr>
                    <td>{{ key }}</td>
                    <td>{{ value }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
        
        {% if results.basic_stats %}
        <div class="section">
            <h2>Basic Image Statistics</h2>
            <h3>Dimensions</h3>
            <table>
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Width</td>
                    <td>{{ results.basic_stats.dimensions.width }} pixels</td>
                </tr>
                <tr>
                    <td>Height</td>
                    <td>{{ results.basic_stats.dimensions.height }} pixels</td>
                </tr>
                <tr>
                    <td>Bands</td>
                    <td>{{ results.basic_stats.dimensions.bands }}</td>
                </tr>
                <tr>
                    <td>Data Type</td>
                    <td>{{ results.basic_stats.overall.data_type }}</td>
                </tr>
                <tr>
                    <td>Size</td>
                    <td>{{ "%.2f"|format(results.basic_stats.overall.size_mb) }} MB</td>
                </tr>
            </table>
            
            <h3>Band Statistics</h3>
            <table>
                <tr>
                    <th>Band</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                </tr>
                {% for band in results.basic_stats.band_stats %}
                <tr>
                    <td>{{ band.band }}</td>
                    <td>{{ "%.4f"|format(band.min) }}</td>
                    <td>{{ "%.4f"|format(band.max) }}</td>
                    <td>{{ "%.4f"|format(band.mean) }}</td>
                    <td>{{ "%.4f"|format(band.median) }}</td>
                    <td>{{ "%.4f"|format(band.std) }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
        
        {% if results.processed_images %}
        <div class="section">
            <h2>Processed Images</h2>
            {% for img in results.processed_images %}
            <div class="image-container">
                <h3>{{ img.title }}</h3>
                <img src="data:image/png;base64,{{ img.data }}" alt="{{ img.title }}">
                {% if img.description %}
                <p>{{ img.description }}</p>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        {% if results.vegetation_stats %}
        <div class="section">
            <h2>Vegetation Analysis</h2>
            <h3>NDVI Statistics</h3>
            <table>
                <tr>
                    <th>Statistic</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Minimum NDVI</td>
                    <td>{{ "%.4f"|format(results.vegetation_stats.ndvi.min) }}</td>
                </tr>
                <tr>
                    <td>Maximum NDVI</td>
                    <td>{{ "%.4f"|format(results.vegetation_stats.ndvi.max) }}</td>
                </tr>
                <tr>
                    <td>Mean NDVI</td>
                    <td>{{ "%.4f"|format(results.vegetation_stats.ndvi.mean) }}</td>
                </tr>
                <tr>
                    <td>Median NDVI</td>
                    <td>{{ "%.4f"|format(results.vegetation_stats.ndvi.median) }}</td>
                </tr>
                <tr>
                    <td>Standard Deviation</td>
                    <td>{{ "%.4f"|format(results.vegetation_stats.ndvi.std) }}</td>
                </tr>
            </table>
            
            <h3>Vegetation Coverage</h3>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Percentage</th>
                </tr>
                <tr>
                    <td>Dense Vegetation (NDVI > 0.3)</td>
                    <td>{{ "%.2f"|format(results.vegetation_stats.coverage.vegetation_percent) }}%</td>
                </tr>
                <tr>
                    <td>Sparse Vegetation (0.1 < NDVI ≤ 0.3)</td>
                    <td>{{ "%.2f"|format(results.vegetation_stats.coverage.sparse_vegetation_percent) }}%</td>
                </tr>
                <tr>
                    <td>Non-Vegetation (NDVI ≤ 0.1)</td>
                    <td>{{ "%.2f"|format(results.vegetation_stats.coverage.non_vegetation_percent) }}%</td>
                </tr>
            </table>
        </div>
        {% endif %}
        
        {% if results.classification_stats %}
        <div class="section">
            <h2>Classification Results</h2>
            <h3>Class Distribution</h3>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Pixel Count</th>
                    <th>Percentage</th>
                </tr>
                {% for cls in results.classification_stats.classes %}
                <tr>
                    <td>{{ cls.class }}</td>
                    <td>{{ cls.count }}</td>
                    <td>{{ "%.2f"|format(cls.percent) }}%</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
        
        {% if results.change_stats %}
        <div class="section">
            <h2>Change Detection Results</h2>
            <h3>Overall Changes</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Changed Area</td>
                    <td>{{ "%.2f"|format(results.change_stats.percentages.changed) }}%</td>
                </tr>
                <tr>
                    <td>Unchanged Area</td>
                    <td>{{ "%.2f"|format(results.change_stats.percentages.unchanged) }}%</td>
                </tr>
                <tr>
                    <td>Total Changed Pixels</td>
                    <td>{{ results.change_stats.pixels.changed }}</td>
                </tr>
            </table>
            
            {% if results.change_stats.components.count > 0 %}
            <h3>Change Components</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Number of Change Components</td>
                    <td>{{ results.change_stats.components.count }}</td>
                </tr>
                <tr>
                    <td>Minimum Component Size</td>
                    <td>{{ results.change_stats.components.min_area }} pixels</td>
                </tr>
                <tr>
                    <td>Maximum Component Size</td>
                    <td>{{ results.change_stats.components.max_area }} pixels</td>
                </tr>
                <tr>
                    <td>Mean Component Size</td>
                    <td>{{ "%.2f"|format(results.change_stats.components.mean_area) }} pixels</td>
                </tr>
                <tr>
                    <td>Median Component Size</td>
                    <td>{{ "%.2f"|format(results.change_stats.components.median_area) }} pixels</td>
                </tr>
            </table>
            {% endif %}
        </div>
        {% endif %}
        
        {% if results.feature_stats %}
        <div class="section">
            <h2>Feature Analysis</h2>
            
            {% if results.feature_stats.objects %}
            <h3>Object Detection</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Number of Objects Detected</td>
                    <td>{{ results.feature_stats.objects.count }}</td>
                </tr>
                {% if results.feature_stats.objects.area %}
                <tr>
                    <td>Minimum Object Size</td>
                    <td>{{ "%.2f"|format(results.feature_stats.objects.area.min) }} pixels</td>
                </tr>
                <tr>
                    <td>Maximum Object Size</td>
                    <td>{{ "%.2f"|format(results.feature_stats.objects.area.max) }} pixels</td>
                </tr>
                <tr>
                    <td>Mean Object Size</td>
                    <td>{{ "%.2f"|format(results.feature_stats.objects.area.mean) }} pixels</td>
                </tr>
                {% endif %}
            </table>
            {% endif %}
            
            {% if results.feature_stats.urban %}
            <h3>Urban Areas</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Urban Coverage</td>
                    <td>{{ "%.2f"|format(results.feature_stats.urban.coverage.positive_percent) }}%</td>
                </tr>
                <tr>
                    <td>Non-Urban Coverage</td>
                    <td>{{ "%.2f"|format(results.feature_stats.urban.coverage.negative_percent) }}%</td>
                </tr>
            </table>
            {% endif %}
            
            {% if results.feature_stats.water %}
            <h3>Water Bodies</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Water Coverage</td>
                    <td>{{ "%.2f"|format(results.feature_stats.water.coverage.positive_percent) }}%</td>
                </tr>
                <tr>
                    <td>Land Coverage</td>
                    <td>{{ "%.2f"|format(results.feature_stats.water.coverage.negative_percent) }}%</td>
                </tr>
            </table>
            {% endif %}
        </div>
        {% endif %}
        
        <div class="footer">
            <p>This report was generated using the Satellite Imagery Analysis Tool.</p>
            <p>© {{ current_year }}</p>
        </div>
    </body>
    </html>
    '''
    
    # Prepare template variables
    template_vars = {
        'current_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'current_year': datetime.now().year,
        'image_name': os.path.basename(image_path),
        'results': results,
        'metadata': metadata
    }
    
    # Render the template
    template = Template(template_str)
    html_content = template.render(template_vars)
    
    # Save to file if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML report saved to {output_path}")
    
    return html_content

def create_pdf_report(image_path, results, metadata=None, output_path=None):
    """
    Create a PDF report of the analysis results.
    
    Args:
        image_path (str): Path to the original image
        results (dict): Dictionary of analysis results
        metadata (dict): Image metadata
        output_path (str): Path to save the PDF report
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Generate HTML report first
        html_content = create_html_report(image_path, results, metadata)
        
        # Convert HTML to PDF using a PDF generation library
        # Note: This requires a PDF library, which might need to be installed
        # Here we'd use something like weasyprint or pdfkit
        
        # This is a placeholder for the actual PDF generation
        logger.warning("PDF report generation requires additional libraries like WeasyPrint or pdfkit")
        logger.info("Generated HTML report instead")
        
        # For now, we'll save the HTML if an output path is provided
        if output_path:
            # Change extension to .html
            html_path = os.path.splitext(output_path)[0] + '.html'
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Saved HTML report to {html_path} instead of PDF")
        
        return True
    except Exception as e:
        logger.error(f"Error creating PDF report: {str(e)}")
        return False

def export_statistics_to_csv(stats, output_path):
    """
    Export statistics to a CSV file.
    
    Args:
        stats (dict): Dictionary of statistics
        output_path (str): Path to save the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Flatten the nested dictionary structure
        flat_stats = {}
        
        def flatten_dict(d, parent_key=''):
            for key, value in d.items():
                new_key = f"{parent_key}_{key}" if parent_key else key
                
                if isinstance(value, dict):
                    flatten_dict(value, new_key)
                else:
                    # Handle lists and numpy arrays
                    if isinstance(value, (list, np.ndarray)) and len(value) > 10:
                        flat_stats[new_key + '_summary'] = f"List/Array with {len(value)} elements"
                    else:
                        flat_stats[new_key] = value
        
        # Flatten each top-level key separately
        for key, value in stats.items():
            if isinstance(value, dict):
                flatten_dict(value, key)
            else:
                flat_stats[key] = value
        
        # Convert to DataFrame
        df = pd.DataFrame(list(flat_stats.items()), columns=['Metric', 'Value'])
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Statistics exported to CSV: {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error exporting statistics to CSV: {str(e)}")
        return False

def export_to_json(data, output_path):
    """
    Export data to a JSON file.
    
    Args:
        data (dict): Data to export
        output_path (str): Path to save the JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert numpy types to Python native types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj
        
        # Convert data
        converted_data = convert_numpy(data)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2)
            
        logger.info(f"Data exported to JSON: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error exporting to JSON: {str(e)}")
        return False

def create_analysis_summary(results):
    """
    Create a text summary of the analysis results.
    
    Args:
        results (dict): Dictionary of analysis results
        
    Returns:
        str: Summary text
    """
    summary = []
    
    # Add header
    summary.append("=== Satellite Imagery Analysis Summary ===")
    summary.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    
    # Add basic image information
    if 'basic_stats' in results:
        basic_stats = results['basic_stats']
        summary.append("--- Image Information ---")
        summary.append(f"Dimensions: {basic_stats['dimensions']['width']} x {basic_stats['dimensions']['height']} pixels")
        summary.append(f"Bands: {basic_stats['dimensions']['bands']}")
        summary.append(f"Data Type: {basic_stats['overall']['data_type']}")
        summary.append(f"Size: {basic_stats['overall']['size_mb']:.2f} MB")
        summary.append("")
    
    # Add vegetation analysis
    if 'vegetation_stats' in results:
        veg_stats = results['vegetation_stats']
        summary.append("--- Vegetation Analysis ---")
        summary.append(f"Mean NDVI: {veg_stats['ndvi']['mean']:.4f}")
        summary.append(f"Vegetation Coverage: {veg_stats['coverage']['vegetation_percent']:.2f}%")
        summary.append(f"Sparse Vegetation: {veg_stats['coverage']['sparse_vegetation_percent']:.2f}%")
        summary.append(f"Non-Vegetation: {veg_stats['coverage']['non_vegetation_percent']:.2f}%")
        summary.append("")
    
    # Add classification results
    if 'classification_stats' in results:
        class_stats = results['classification_stats']
        summary.append("--- Classification Results ---")
        summary.append(f"Number of Classes: {class_stats['num_classes']}")
        
        for cls in class_stats['classes']:
            summary.append(f"Class {cls['class']}: {cls['percent']:.2f}% ({cls['count']} pixels)")
        
        summary.append("")
    
    # Add change detection results
    if 'change_stats' in results:
        change_stats = results['change_stats']
        summary.append("--- Change Detection Results ---")
        summary.append(f"Changed Area: {change_stats['percentages']['changed']:.2f}%")
        summary.append(f"Unchanged Area: {change_stats['percentages']['unchanged']:.2f}%")
        
        if 'components' in change_stats and 'count' in change_stats['components'] and change_stats['components']['count'] > 0:
            summary.append(f"Number of Change Components: {change_stats['components']['count']}")
            summary.append(f"Mean Component Size: {change_stats['components']['mean_area']:.2f} pixels")
        
        summary.append("")
    
    # Add feature analysis summary
    if 'feature_stats' in results:
        summary.append("--- Feature Analysis ---")
        
        if 'objects' in results['feature_stats']:
            obj_stats = results['feature_stats']['objects']
            summary.append(f"Objects Detected: {obj_stats['count']}")
            
            if 'area' in obj_stats:
                summary.append(f"Mean Object Size: {obj_stats['area']['mean']:.2f} pixels")
        
        if 'urban' in results['feature_stats']:
            urban_stats = results['feature_stats']['urban']
            summary.append(f"Urban Coverage: {urban_stats['coverage']['positive_percent']:.2f}%")
        
        if 'water' in results['feature_stats']:
            water_stats = results['feature_stats']['water']
            summary.append(f"Water Coverage: {water_stats['coverage']['positive_percent']:.2f}%")
            
        summary.append("")
    
    # Join all lines with newlines
    return "\n".join(summary)

def save_text_summary(summary_text, output_path):
    """
    Save a text summary to a file.
    
    Args:
        summary_text (str): Summary text
        output_path (str): Path to save the summary
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
            
        logger.info(f"Text summary saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving text summary: {str(e)}")
        return False
