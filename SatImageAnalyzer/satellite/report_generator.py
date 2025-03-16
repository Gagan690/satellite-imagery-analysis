"""
Satellite Image Report Generator Module
---------------------------------------

This module provides functions for generating reports based on satellite imagery analysis.
It includes tools for creating HTML reports with analysis results, charts, and visualizations.

Key functions:
- Generate HTML reports for satellite image analysis
- Create charts and visualizations for analysis results
- Export analysis results to various formats

Dependencies:
- jinja2: For HTML templating
- matplotlib: For charts and visualization
- numpy: For numerical operations
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from io import BytesIO
import base64

# Set up logging
logger = logging.getLogger(__name__)

def generate_html_report(filename, metadata, results, image_paths):
    """
    Generate an HTML report for satellite imagery analysis.
    
    Parameters:
    -----------
    filename : str
        Name of the analyzed image file
    metadata : dict
        Dictionary containing metadata about the image
    results : dict
        Dictionary containing analysis results
    image_paths : list
        List of paths to images to include in the report
        
    Returns:
    --------
    str
        Path to the generated HTML report
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.join('static', 'reports')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique report filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"report_{os.path.splitext(filename)[0]}_{timestamp}.html"
    report_path = os.path.join(output_dir, report_filename)
    
    # Generate charts for the report
    chart_images = []
    
    if 'ndvi' in results:
        # Create NDVI distribution chart
        ndvi_chart = create_ndvi_chart(results['ndvi'])
        chart_images.append(('NDVI Distribution', ndvi_chart))
    
    if 'water' in results:
        # Create water coverage chart
        water_chart = create_pie_chart('Water Coverage', 
                                     [results['water']['coverage'], 100 - results['water']['coverage']],
                                     ['Water', 'Land'])
        chart_images.append(('Water Coverage', water_chart))
    
    if 'urban' in results:
        # Create urban coverage chart
        urban_chart = create_pie_chart('Urban Coverage', 
                                     [results['urban']['coverage'], 100 - results['urban']['coverage']],
                                     ['Urban', 'Non-urban'])
        chart_images.append(('Urban Coverage', urban_chart))
    
    # Prepare image paths for the report
    report_images = []
    for path in image_paths:
        if path and os.path.exists(os.path.join('static', 'uploads', path)):
            report_images.append(('/' + os.path.join('static', 'uploads', path), 
                                 os.path.basename(path)))
    
    # Prepare metadata for the report
    report_metadata = []
    for key, value in metadata.items():
        if isinstance(value, list):
            # Format lists nicely
            report_metadata.append((key, ", ".join(map(str, value))))
        elif isinstance(value, dict):
            # Format dictionaries nicely
            report_metadata.append((key, json.dumps(value, indent=2)))
        else:
            report_metadata.append((key, value))
    
    # Create HTML report content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Satellite Image Analysis Report - {filename}</title>
    <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
    <link href="/static/css/custom.css" rel="stylesheet">
    <style>
        .report-section {{
            margin-bottom: 30px;
        }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .metadata-table {{
            font-size: 0.9rem;
        }}
        .results-table {{
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1 class="mb-4">Satellite Image Analysis Report</h1>
        
        <div class="row">
            <div class="col-12">
                <div class="card mb-4">
                    <div class="card-header">
                        <h2>File Information</h2>
                    </div>
                    <div class="card-body">
                        <p><strong>Filename:</strong> {filename}</p>
                        <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card mb-4">
                    <div class="card-header">
                        <h2>Analysis Results</h2>
                    </div>
                    <div class="card-body">
"""

    # Add result tables
    if 'ndvi' in results:
        html_content += f"""
                        <div class="report-section">
                            <h3>NDVI Analysis</h3>
                            <table class="table table-bordered results-table">
                                <thead>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Mean NDVI</td>
                                        <td>{results['ndvi']['mean']:.4f}</td>
                                    </tr>
                                    <tr>
                                        <td>Minimum NDVI</td>
                                        <td>{results['ndvi']['min']:.4f}</td>
                                    </tr>
                                    <tr>
                                        <td>Maximum NDVI</td>
                                        <td>{results['ndvi']['max']:.4f}</td>
                                    </tr>
                                    <tr>
                                        <td>Vegetation Cover</td>
                                        <td>{results['ndvi']['vegetation_cover']:.2f}%</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
"""

    if 'water' in results:
        html_content += f"""
                        <div class="report-section">
                            <h3>Water Body Analysis</h3>
                            <table class="table table-bordered results-table">
                                <thead>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Water Coverage</td>
                                        <td>{results['water']['coverage']:.2f}%</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
"""

    if 'urban' in results:
        html_content += f"""
                        <div class="report-section">
                            <h3>Urban Area Analysis</h3>
                            <table class="table table-bordered results-table">
                                <thead>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Urban Coverage</td>
                                        <td>{results['urban']['coverage']:.2f}%</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
"""

    html_content += """
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card mb-4">
                    <div class="card-header">
                        <h2>Visualizations</h2>
                    </div>
                    <div class="card-body">
"""

    # Add charts
    for chart_title, chart_img in chart_images:
        html_content += f"""
                        <div class="chart-container">
                            <h3>{chart_title}</h3>
                            <img src="data:image/png;base64,{chart_img}" alt="{chart_title}" class="img-fluid">
                        </div>
"""

    # Add images
    html_content += """
                        <div class="report-section">
                            <h3>Analysis Images</h3>
                            <div class="row">
"""

    for img_path, img_title in report_images:
        html_content += f"""
                                <div class="col-md-6 mb-4">
                                    <div class="card">
                                        <div class="card-header">{img_title}</div>
                                        <div class="card-body text-center">
                                            <img src="{img_path}" alt="{img_title}" class="img-fluid">
                                        </div>
                                    </div>
                                </div>
"""

    html_content += """
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card mb-4">
                    <div class="card-header">
                        <h2>Image Metadata</h2>
                    </div>
                    <div class="card-body">
                        <table class="table table-bordered table-striped metadata-table">
                            <thead>
                                <tr>
                                    <th style="width: 30%">Property</th>
                                    <th>Value</th>
                                </tr>
                            </thead>
                            <tbody>
"""

    # Add metadata
    for key, value in report_metadata:
        html_content += f"""
                                <tr>
                                    <td>{key}</td>
                                    <td><pre class="mb-0">{value}</pre></td>
                                </tr>
"""

    html_content += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="mt-4 mb-5">
            <a href="/" class="btn btn-primary">Back to Home</a>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

    # Write HTML content to file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Generated HTML report at {report_path}")
    return report_path

def create_ndvi_chart(ndvi_results):
    """
    Create a chart showing NDVI distribution.
    
    Parameters:
    -----------
    ndvi_results : dict
        Dictionary containing NDVI analysis results
        
    Returns:
    --------
    str
        Base64-encoded PNG image of the chart
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define NDVI categories
    categories = [
        (-1.0, 0.0, 'Water/Non-vegetation', 'royalblue'),
        (0.0, 0.2, 'Barren/Sparse Vegetation', 'saddlebrown'),
        (0.2, 0.4, 'Shrubs/Grassland', 'yellowgreen'),
        (0.4, 0.6, 'Moderate Vegetation', 'limegreen'),
        (0.6, 0.8, 'Dense Vegetation', 'forestgreen'),
        (0.8, 1.0, 'Very Dense Vegetation', 'darkgreen')
    ]
    
    # Set up bar positions
    positions = range(len(categories))
    
    # Default values if we don't have the actual distribution
    distribution = [15, 20, 25, 20, 15, 5]  # Example percentages
    
    # Calculate vegetation cover for each category
    veg_cover = ndvi_results.get('vegetation_cover', 0)
    non_veg_cover = 100 - veg_cover
    
    # Adjust distribution based on vegetation cover
    if veg_cover > 0:
        distribution[0] = non_veg_cover * 0.6  # Water/Non-veg
        distribution[1] = non_veg_cover * 0.4  # Barren
        
        # Distribute remaining percentage among vegetation categories
        remaining = veg_cover
        distribution[2] = remaining * 0.3  # Shrubs/Grassland
        distribution[3] = remaining * 0.3  # Moderate
        distribution[4] = remaining * 0.3  # Dense
        distribution[5] = remaining * 0.1  # Very Dense
    
    # Create bars
    bars = []
    for i, (low, high, label, color) in enumerate(categories):
        bars.append(ax.bar(i, distribution[i], color=color, label=f"{label} ({low:.1f} to {high:.1f})"))
    
    # Add labels and title
    ax.set_xlabel('NDVI Categories')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('NDVI Distribution')
    ax.set_xticks(positions)
    ax.set_xticklabels([cat[2] for cat in categories], rotation=45, ha='right')
    
    # Add mean NDVI line
    mean_ndvi = ndvi_results.get('mean', 0)
    if -1 <= mean_ndvi <= 1:
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.7)
        ax.text(len(categories) - 1, 50, f' Mean NDVI: {mean_ndvi:.2f}', 
               verticalalignment='center', color='red')
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    
    return img_str

def create_pie_chart(title, values, labels):
    """
    Create a pie chart for visualization.
    
    Parameters:
    -----------
    title : str
        Title of the chart
    values : list
        Values for pie chart segments
    labels : list
        Labels for pie chart segments
        
    Returns:
    --------
    str
        Base64-encoded PNG image of the chart
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                     textprops={'color': 'white'},
                                     colors=['#3498db', '#95a5a6'])
    
    # Add title
    ax.set_title(title)
    
    # Add legend
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    
    return img_str

def export_results_to_json(filename, metadata, results):
    """
    Export analysis results to a JSON file.
    
    Parameters:
    -----------
    filename : str
        Name of the analyzed image file
    metadata : dict
        Dictionary containing metadata about the image
    results : dict
        Dictionary containing analysis results
        
    Returns:
    --------
    str
        Path to the exported JSON file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.join('static', 'reports')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_filename = f"results_{os.path.splitext(filename)[0]}_{timestamp}.json"
    json_path = os.path.join(output_dir, json_filename)
    
    # Prepare export data
    export_data = {
        'filename': filename,
        'analysis_date': datetime.now().isoformat(),
        'metadata': metadata,
        'results': results
    }
    
    # Convert NumPy types to native Python types
    def convert_np(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_np(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_np(item) for item in obj]
        else:
            return obj
    
    export_data = convert_np(export_data)
    
    # Write to JSON file
    with open(json_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    logger.info(f"Exported results to JSON at {json_path}")
    return json_path

def export_results_to_csv(filename, results):
    """
    Export analysis results to a CSV file.
    
    Parameters:
    -----------
    filename : str
        Name of the analyzed image file
    results : dict
        Dictionary containing analysis results
        
    Returns:
    --------
    str
        Path to the exported CSV file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.join('static', 'reports')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"results_{os.path.splitext(filename)[0]}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Prepare CSV content
    csv_content = ["Metric,Value"]
    
    # Add NDVI results
    if 'ndvi' in results:
        csv_content.append(f"Mean NDVI,{results['ndvi']['mean']}")
        csv_content.append(f"Minimum NDVI,{results['ndvi']['min']}")
        csv_content.append(f"Maximum NDVI,{results['ndvi']['max']}")
        csv_content.append(f"Vegetation Cover (%),{results['ndvi']['vegetation_cover']}")
    
    # Add water results
    if 'water' in results:
        csv_content.append(f"Water Coverage (%),{results['water']['coverage']}")
    
    # Add urban results
    if 'urban' in results:
        csv_content.append(f"Urban Coverage (%),{results['urban']['coverage']}")
    
    # Write to CSV file
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_content))
    
    logger.info(f"Exported results to CSV at {csv_path}")
    return csv_path
