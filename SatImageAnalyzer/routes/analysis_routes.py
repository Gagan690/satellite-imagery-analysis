# routes/analysis_routes.py
# Routes for handling satellite imagery analysis functions

import os
import io
import uuid
import logging
import numpy as np
import base64
from datetime import datetime
from flask import (
    Blueprint, render_template, request, redirect, url_for, 
    flash, current_app, send_file, abort, jsonify, session
)
from werkzeug.utils import secure_filename

from models import db, Analysis, SatelliteImage
from config import get_config
from utils import image_processing as ip
from utils import feature_extraction as fe
from utils import visualization as vis
from utils import report_generation as report
from utils import file_handler as fh
from sample_data.metadata import get_sample_images

# Configure logger
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Define the blueprint
analysis_bp = Blueprint('analysis_bp', __name__)

@analysis_bp.route('/analysis')
def analysis():
    """Render the analysis page with form for uploading and configuring analysis."""
    try:
        # Get sample images for the sample data dropdown
        sample_images = get_sample_images()
        
        return render_template('analysis.html', 
                              sample_images=sample_images,
                              now=datetime.now())
    except Exception as e:
        logger.error(f"Error rendering analysis page: {str(e)}")
        flash(f"Error loading analysis page: {str(e)}", "danger")
        return render_template('error.html', error=str(e))

@analysis_bp.route('/process-image', methods=['POST'])
def process_image():
    """Process the uploaded satellite image according to selected parameters."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(url_for('analysis_bp.analysis'))
        
        # Get the uploaded file
        file = request.files['file']
        
        # If no file was selected
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(url_for('analysis_bp.analysis'))
        
        # Save the uploaded file
        success, file_path, error_message = fh.save_uploaded_file(file)
        if not success:
            flash(error_message, 'danger')
            return redirect(url_for('analysis_bp.analysis'))
        
        # Get form data
        title = request.form.get('title', 'Untitled Analysis')
        description = request.form.get('description', '')
        processing_technique = request.form.get('processing_technique')
        feature_extraction = request.form.get('feature_extraction', '')
        
        # Load the image
        try:
            image_data, metadata = ip.load_image(file_path)
            logger.info(f"Successfully loaded image: {file.filename}")
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            flash(f"Error loading image: {str(e)}", "danger")
            fh.delete_file(file_path)
            return redirect(url_for('analysis_bp.analysis'))
        
        # Create a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Create a directory for results
        result_dir = fh.create_result_directory(analysis_id)
        
        # Save original image for comparison
        original_image_vis = None
        try:
            if len(image_data.shape) == 3 and image_data.shape[2] >= 3:
                original_rgb = vis.create_composite_visualization(image_data, bands=[0, 1, 2])
                original_image_vis = vis.create_comparison_visualization([original_rgb], ['Original Image'])
            else:
                # Single band visualization
                original_norm = vis.normalize_image(image_data)
                original_image_vis = vis.create_comparison_visualization([original_norm], ['Original Image'])
        except Exception as e:
            logger.error(f"Error creating original image visualization: {str(e)}")
            # Continue without the original visualization
        
        # Process the image based on selected technique
        processed_image = None
        processed_image_vis = None
        processing_stats = {"method": processing_technique}
        
        try:
            # Apply the selected processing technique
            if processing_technique == 'histogram_equalization':
                processed_image = ip.histogram_equalization(image_data)
                processing_stats["parameters"] = {}
                
            elif processing_technique == 'gaussian_blur':
                sigma = float(request.form.get('sigma', 1.0))
                processed_image = ip.gaussian_blur(image_data, sigma)
                processing_stats["parameters"] = {"sigma": sigma}
                
            elif processing_technique == 'edge_detection':
                method = request.form.get('edge_method', 'canny')
                processed_image = ip.edge_detection(image_data, method)
                processing_stats["parameters"] = {"method": method}
                
            elif processing_technique == 'ndvi_calculation':
                red_band = int(request.form.get('red_band', 2))
                nir_band = int(request.form.get('nir_band', 3))
                processed_image = ip.calculate_ndvi(image_data, red_band, nir_band)
                processing_stats["parameters"] = {"red_band": red_band, "nir_band": nir_band}
                
            elif processing_technique == 'rgb_composite':
                r_band = int(request.form.get('r_band', 2))
                g_band = int(request.form.get('g_band', 1))
                b_band = int(request.form.get('b_band', 0))
                processed_image = ip.create_composite(image_data, [r_band, g_band, b_band])
                processing_stats["parameters"] = {"r_band": r_band, "g_band": g_band, "b_band": b_band}
                
            elif processing_technique == 'unsupervised_classification':
                n_clusters = int(request.form.get('n_clusters', 5))
                processed_image = ip.unsupervised_classification(image_data, n_clusters)
                processing_stats["parameters"] = {"n_clusters": n_clusters}
                
            else:
                flash(f"Unsupported processing technique: {processing_technique}", "danger")
                fh.delete_file(file_path)
                return redirect(url_for('analysis_bp.analysis'))
            
            logger.info(f"Successfully applied {processing_technique} to image")
            
            # Create visualization of processed image
            if processing_technique == 'ndvi_calculation':
                # Use specialized NDVI visualization
                processed_image_vis = vis.visualize_ndvi(processed_image)
            elif processing_technique == 'unsupervised_classification':
                # Use specialized classification visualization
                n_clusters = int(request.form.get('n_clusters', 5))
                processed_image_vis = vis.visualize_classification(processed_image, n_clusters)
            else:
                # General visualization
                if len(processed_image.shape) == 3 and processed_image.shape[2] >= 3:
                    processed_rgb = vis.normalize_image(processed_image)
                    processed_image_vis = vis.create_comparison_visualization([processed_rgb], ['Processed Image'])
                else:
                    # Single band visualization
                    processed_norm = vis.normalize_image(processed_image)
                    processed_image_vis = vis.create_comparison_visualization([processed_norm], ['Processed Image'])
        
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            flash(f"Error processing image: {str(e)}", "danger")
            fh.delete_file(file_path)
            return redirect(url_for('analysis_bp.analysis'))
        
        # Extract features if requested
        extracted_features = None
        features_extracted_vis = None
        feature_stats = {}
        
        if feature_extraction:
            try:
                if feature_extraction == 'vegetation_indices':
                    # Extract vegetation indices
                    index_type = request.form.get('index_type', 'ndvi')
                    extracted_features = fe.extract_vegetation_indices(image_data, index_type)
                    feature_stats['vegetation'] = report.generate_vegetation_statistics(
                        extracted_features if index_type != 'all' else extracted_features['ndvi']
                    )
                    # Create visualization
                    features_extracted_vis = vis.visualize_features(
                        image_data, extracted_features, 'vegetation'
                    )
                
                elif feature_extraction == 'urban_detection':
                    # Detect urban areas
                    urban_method = request.form.get('urban_method', 'texture')
                    extracted_features = fe.detect_urban_areas(image_data, urban_method)
                    feature_stats['urban'] = report.generate_feature_statistics(
                        extracted_features, 'urban'
                    )
                    # Create visualization
                    features_extracted_vis = vis.visualize_features(
                        image_data, extracted_features, 'urban'
                    )
                
                elif feature_extraction == 'water_detection':
                    # Detect water bodies
                    water_method = request.form.get('water_method', 'ndwi')
                    extracted_features = fe.detect_water(image_data, water_method)
                    feature_stats['water'] = report.generate_feature_statistics(
                        extracted_features, 'water'
                    )
                    # Create visualization
                    features_extracted_vis = vis.visualize_features(
                        image_data, extracted_features, 'water'
                    )
                
                elif feature_extraction == 'object_detection':
                    # Detect objects
                    object_method = request.form.get('object_method', 'contour')
                    min_size = int(request.form.get('min_size', 100))
                    extracted_features = fe.object_detection(image_data, object_method, min_size)
                    feature_stats['objects'] = report.generate_feature_statistics(
                        extracted_features, 'objects'
                    )
                    # Create visualization
                    features_extracted_vis = vis.visualize_features(
                        image_data, extracted_features, 'objects'
                    )
                
                elif feature_extraction == 'texture_analysis':
                    # Extract texture features
                    texture_method = request.form.get('texture_method', 'glcm')
                    extracted_features = fe.extract_texture_features(image_data, texture_method)
                    feature_stats['texture'] = report.generate_feature_statistics(
                        extracted_features, 'texture'
                    )
                    # No standard visualization for texture features
                
                logger.info(f"Successfully extracted features using {feature_extraction}")
            
            except Exception as e:
                logger.error(f"Error extracting features: {str(e)}")
                flash(f"Error extracting features: {str(e)}", "warning")
                # Continue without feature extraction
        
        # Generate statistics
        basic_stats = report.generate_basic_statistics(image_data)
        
        # Create statistics collection
        statistics = {
            'basic_stats': basic_stats,
            'processing_stats': processing_stats
        }
        
        # Add feature statistics if available
        if feature_stats:
            statistics['feature_stats'] = feature_stats
        
        # Add NDVI-specific statistics if NDVI was calculated
        if processing_technique == 'ndvi_calculation':
            statistics['vegetation_stats'] = report.generate_vegetation_statistics(processed_image)
        
        # Add classification statistics if classification was performed
        if processing_technique == 'unsupervised_classification':
            statistics['classification_stats'] = report.generate_classification_statistics(processed_image)
        
        # Generate summary report
        summary_text = report.create_analysis_summary(statistics)
        
        # Save summary to file
        summary_path = os.path.join(result_dir, 'summary.txt')
        report.save_text_summary(summary_text, summary_path)
        
        # Export statistics to CSV if requested
        if request.form.get('export_stats'):
            csv_path = os.path.join(result_dir, 'statistics.csv')
            report.export_statistics_to_csv(statistics, csv_path)
        
        # Save processed images if requested
        if request.form.get('save_processed'):
            processed_path = os.path.join(result_dir, 'processed_image.tif')
            ip.save_image(processed_image, processed_path)
        
        # Generate HTML report if requested
        report_path = None
        if request.form.get('generate_report'):
            report_path = os.path.join(result_dir, 'report.html')
            
            # Create processed images collection for the report
            processed_images = []
            
            if processed_image_vis:
                processed_images.append({
                    'title': f'Processed Image ({processing_technique})',
                    'data': processed_image_vis,
                    'description': f'Result of applying {processing_technique} to the original image'
                })
            
            if features_extracted_vis:
                processed_images.append({
                    'title': f'Feature Extraction ({feature_extraction})',
                    'data': features_extracted_vis,
                    'description': f'Result of extracting {feature_extraction} features'
                })
            
            # Add additional visualizations based on processing technique
            if processing_technique == 'ndvi_calculation':
                ndvi_hist = vis.visualize_image_histogram(processed_image, 'NDVI Distribution')
                processed_images.append({
                    'title': 'NDVI Histogram',
                    'data': ndvi_hist,
                    'description': 'Distribution of NDVI values across the image'
                })
            
            # Create report data
            report_data = {
                'basic_stats': basic_stats,
                'processing_stats': processing_stats,
                'processed_images': processed_images
            }
            
            # Add feature statistics if available
            if feature_stats:
                report_data['feature_stats'] = feature_stats
            
            # Add NDVI-specific statistics if NDVI was calculated
            if processing_technique == 'ndvi_calculation':
                report_data['vegetation_stats'] = statistics['vegetation_stats']
            
            # Add classification statistics if classification was performed
            if processing_technique == 'unsupervised_classification':
                report_data['classification_stats'] = statistics['classification_stats']
            
            # Generate HTML report
            html_report = report.create_html_report(file.filename, report_data, metadata, report_path)
            logger.info(f"Generated HTML report at {report_path}")
        
        # Save metadata about this analysis to the database
        try:
            file_ext = os.path.splitext(file.filename)[1].lower()[1:]
            
            # Create Analysis record
            analysis_record = Analysis(
                title=title,
                description=description,
                image_path=file_path,
                original_filename=file.filename,
                image_format=file_ext,
                processing_method=processing_technique,
                features_extracted=feature_extraction,
                results_path=result_dir,
                report_path=report_path
            )
            
            # Add geospatial information if available
            if metadata and 'transform' in metadata and metadata['transform']:
                try:
                    # Extract approximate center coordinates from geotransform
                    gt = metadata['transform']
                    width = metadata.get('width', 0)
                    height = metadata.get('height', 0)
                    
                    # Calculate center coordinates
                    if width and height and gt:
                        center_x = gt[0] + width/2 * gt[1] + height/2 * gt[2]
                        center_y = gt[3] + width/2 * gt[4] + height/2 * gt[5]
                        
                        analysis_record.longitude = float(center_x)
                        analysis_record.latitude = float(center_y)
                except Exception as e:
                    logger.warning(f"Could not extract coordinates from metadata: {str(e)}")
            
            # Add resolution if available
            if metadata and 'transform' in metadata and metadata['transform']:
                try:
                    gt = metadata['transform']
                    # Approximate resolution as average of absolute values of gt[1] and gt[5]
                    resolution = (abs(gt[1]) + abs(gt[5])) / 2
                    analysis_record.resolution = float(resolution)
                except Exception as e:
                    logger.warning(f"Could not extract resolution from metadata: {str(e)}")
            
            # Save to database
            db.session.add(analysis_record)
            db.session.commit()
            
            logger.info(f"Saved analysis record to database with ID: {analysis_record.id}")
            
            # Save the image visualization data in session for the results page
            session['original_image'] = original_image_vis if original_image_vis else None
            session['processed_image'] = processed_image_vis if processed_image_vis else None
            
            # Redirect to results page
            return redirect(url_for('analysis_bp.show_results', analysis_id=analysis_record.id))
            
        except Exception as e:
            logger.error(f"Error saving analysis to database: {str(e)}")
            flash(f"Error saving analysis: {str(e)}", "danger")
            return redirect(url_for('analysis_bp.analysis'))
        
    except Exception as e:
        logger.error(f"Unexpected error in process_image: {str(e)}")
        flash(f"An unexpected error occurred: {str(e)}", "danger")
        return redirect(url_for('analysis_bp.analysis'))

@analysis_bp.route('/results/<int:analysis_id>')
def show_results(analysis_id):
    """Display the results of an analysis."""
    try:
        # Get the analysis record from the database
        analysis = Analysis.query.get_or_404(analysis_id)
        
        # Get visualizations from session
        original_image = session.get('original_image')
        processed_image = session.get('processed_image')
        
        # Get metadata if available
        metadata = {}
        if analysis.latitude and analysis.longitude:
            metadata['Latitude'] = f"{analysis.latitude:.6f}"
            metadata['Longitude'] = f"{analysis.longitude:.6f}"
        
        if analysis.resolution:
            metadata['Resolution'] = f"{analysis.resolution:.2f} m"
        
        if analysis.image_format:
            metadata['Format'] = analysis.image_format.upper()
        
        # Load statistics if available
        statistics = None
        if analysis.results_path:
            stats_path = os.path.join(analysis.results_path, 'statistics.csv')
            if os.path.exists(stats_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(stats_path)
                    statistics_dict = df.set_index('Metric')['Value'].to_dict()
                    
                    # Parse and structure statistics
                    statistics = {}
                    for key, value in statistics_dict.items():
                        parts = key.split('_')
                        
                        # Handle basic stats
                        if parts[0] == 'basic':
                            if 'basic_stats' not in statistics:
                                statistics['basic_stats'] = {'dimensions': {}, 'band_stats': [], 'overall': {}}
                            
                            if parts[1] == 'dimensions':
                                statistics['basic_stats']['dimensions'][parts[2]] = value
                            elif parts[1] == 'overall':
                                statistics['basic_stats']['overall'][parts[2]] = value
                        
                        # Handle processing stats
                        elif parts[0] == 'processing':
                            if 'processing_stats' not in statistics:
                                statistics['processing_stats'] = {'method': analysis.processing_method, 'parameters': {}, 'metrics': {}}
                            
                            if parts[1] == 'parameters':
                                statistics['processing_stats']['parameters'][parts[2]] = value
                            elif parts[1] == 'metrics':
                                statistics['processing_stats']['metrics'][parts[2]] = value
                        
                        # Handle vegetation stats
                        elif parts[0] == 'vegetation':
                            if 'vegetation_stats' not in statistics:
                                statistics['vegetation_stats'] = {'ndvi': {}, 'coverage': {}}
                            
                            if parts[1] == 'ndvi':
                                statistics['vegetation_stats']['ndvi'][parts[2]] = value
                            elif parts[1] == 'coverage':
                                statistics['vegetation_stats']['coverage'][parts[2]] = value
                        
                        # Handle classification stats
                        elif parts[0] == 'classification':
                            if 'classification_stats' not in statistics:
                                statistics['classification_stats'] = {'classes': []}
                            
                            # This is simplified, you'd need more logic for complex nested structures
                
                except Exception as e:
                    logger.warning(f"Error loading statistics: {str(e)}")
        
        # Load processed images if available
        processed_images = []
        if analysis.results_path:
            try:
                # Check if report exists
                report_path = os.path.join(analysis.results_path, 'report.html')
                if os.path.exists(report_path):
                    with open(report_path, 'r') as f:
                        report_content = f.read()
                        
                        # Extract image data from report
                        import re
                        img_tags = re.findall(r'<img src="data:image/png;base64,(.*?)".*?>', report_content)
                        titles = re.findall(r'<h3>(.*?)</h3>', report_content)
                        
                        # Create processed images list
                        for i, (img_data, title) in enumerate(zip(img_tags, titles)):
                            processed_images.append({
                                'id': i,
                                'title': title,
                                'data': img_data,
                                'description': ''  # Description not easily extractable
                            })
            except Exception as e:
                logger.warning(f"Error loading processed images: {str(e)}")
        
        # Load feature extraction results if available
        features_extracted = []
        if analysis.features_extracted:
            feature_type = analysis.features_extracted
            
            # Try to find feature extraction visualization in processed images
            for img in processed_images:
                if feature_type.lower() in img['title'].lower():
                    features_extracted.append({
                        'title': img['title'],
                        'visualization': img['data'],
                        'statistics': {}  # Would need more complex parsing to extract specific statistics
                    })
                    break
        
        return render_template('results.html', 
                              analysis=analysis,
                              metadata=metadata,
                              statistics=statistics,
                              processed_images=processed_images,
                              features_extracted=features_extracted,
                              original_image=original_image,
                              processed_image=processed_image,
                              now=datetime.now())
    except Exception as e:
        logger.error(f"Error showing results: {str(e)}")
        flash(f"Error loading results: {str(e)}", "danger")
        return redirect(url_for('analysis_bp.analysis'))

@analysis_bp.route('/download/report/<int:analysis_id>')
def download_report(analysis_id):
    """Download the analysis report."""
    try:
        # Get the analysis record
        analysis = Analysis.query.get_or_404(analysis_id)
        
        # Check if report exists
        if not analysis.report_path or not os.path.exists(analysis.report_path):
            flash("Report not found", "danger")
            return redirect(url_for('analysis_bp.show_results', analysis_id=analysis_id))
        
        # Return the file
        return send_file(analysis.report_path,
                         as_attachment=True,
                         download_name=f"satellite_analysis_report_{analysis_id}.html",
                         mimetype='text/html')
    except Exception as e:
        logger.error(f"Error downloading report: {str(e)}")
        flash(f"Error downloading report: {str(e)}", "danger")
        return redirect(url_for('analysis_bp.show_results', analysis_id=analysis_id))

@analysis_bp.route('/download/image/<int:image_id>')
def download_image(image_id):
    """Download a processed image."""
    try:
        # For this endpoint, image_id is actually the index in the processed_images list
        # We would need to store this information more robustly in a real application
        
        # Get the analysis_id from the referrer
        analysis_id = request.args.get('analysis_id')
        if not analysis_id:
            flash("Analysis ID not provided", "danger")
            return redirect(url_for('analysis_bp.analysis'))
        
        analysis = Analysis.query.get_or_404(int(analysis_id))
        
        # Check if results directory exists
        if not analysis.results_path or not os.path.exists(analysis.results_path):
            flash("Results not found", "danger")
            return redirect(url_for('analysis_bp.show_results', analysis_id=analysis_id))
        
        # Look for the image file
        if image_id == 0:  # Original image
            if not analysis.image_path or not os.path.exists(analysis.image_path):
                flash("Original image not found", "danger")
                return redirect(url_for('analysis_bp.show_results', analysis_id=analysis_id))
            
            return send_file(analysis.image_path,
                            as_attachment=True,
                            download_name=f"original_{analysis.original_filename}",
                            mimetype='image/tiff')
        else:
            # For processed images, we'll use the processed_image.tif in the results directory
            processed_path = os.path.join(analysis.results_path, 'processed_image.tif')
            if not os.path.exists(processed_path):
                flash("Processed image not found", "danger")
                return redirect(url_for('analysis_bp.show_results', analysis_id=analysis_id))
            
            return send_file(processed_path,
                            as_attachment=True,
                            download_name=f"processed_{analysis.original_filename}",
                            mimetype='image/tiff')
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        flash(f"Error downloading image: {str(e)}", "danger")
        return redirect(url_for('analysis_bp.analysis'))

@analysis_bp.route('/download/statistics/<int:analysis_id>')
def download_statistics(analysis_id):
    """Download the analysis statistics as CSV."""
    try:
        # Get the analysis record
        analysis = Analysis.query.get_or_404(analysis_id)
        
        # Check if statistics file exists
        csv_path = os.path.join(analysis.results_path, 'statistics.csv')
        if not os.path.exists(csv_path):
            flash("Statistics file not found", "danger")
            return redirect(url_for('analysis_bp.show_results', analysis_id=analysis_id))
        
        # Return the file
        return send_file(csv_path,
                         as_attachment=True,
                         download_name=f"satellite_analysis_stats_{analysis_id}.csv",
                         mimetype='text/csv')
    except Exception as e:
        logger.error(f"Error downloading statistics: {str(e)}")
        flash(f"Error downloading statistics: {str(e)}", "danger")
        return redirect(url_for('analysis_bp.show_results', analysis_id=analysis_id))

@analysis_bp.route('/use-sample-image', methods=['POST'])
def use_sample_image():
    """Use a sample satellite image for analysis."""
    try:
        # Get the sample ID from the form
        sample_id = request.form.get('sample_id')
        if not sample_id:
            flash("No sample image selected", "danger")
            return redirect(url_for('analysis_bp.analysis'))
        
        # Get sample images
        sample_images = get_sample_images()
        
        # Find the selected image
        selected_image = None
        for image in sample_images:
            if str(image['id']) == sample_id:
                selected_image = image
                break
        
        if not selected_image:
            flash("Sample image not found", "danger")
            return redirect(url_for('analysis_bp.analysis'))
        
        # Get the file path
        file_path = os.path.join('sample_data', selected_image['filename'])
        
        # Check if the file exists
        if not os.path.exists(file_path):
            flash(f"Sample file not found at {file_path}", "danger")
            return redirect(url_for('analysis_bp.analysis'))
        
        # Create a copy of the file in the uploads directory
        unique_name = f"{uuid.uuid4().hex}_{os.path.basename(file_path)}"
        dest_path = os.path.join(config.UPLOAD_FOLDER, unique_name)
        
        if not fh.copy_file(file_path, dest_path):
            flash("Error copying sample file", "danger")
            return redirect(url_for('analysis_bp.analysis'))
        
        # Store the file info in session for the analysis form
        session['sample_file'] = {
            'path': dest_path,
            'name': selected_image['name'],
            'format': selected_image['format'],
            'bands': selected_image['bands']
        }
        
        flash(f"Sample image '{selected_image['name']}' loaded successfully", "success")
        return redirect(url_for('analysis_bp.analysis'))
    except Exception as e:
        logger.error(f"Error using sample image: {str(e)}")
        flash(f"Error using sample image: {str(e)}", "danger")
        return redirect(url_for('analysis_bp.analysis'))
