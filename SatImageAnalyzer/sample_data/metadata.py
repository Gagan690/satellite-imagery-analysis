# sample_data/metadata.py
# Metadata information for the sample satellite imagery

import os
import logging

# Configure logger
logger = logging.getLogger(__name__)

def get_sample_images():
    """
    Get metadata for all available sample satellite images.
    
    Returns:
        list: List of dictionaries with sample image metadata
    """
    # Check if sample data directory exists
    if not os.path.exists('sample_data'):
        logger.warning("Sample data directory does not exist")
        return []
    
    # Define metadata for sample images
    # In a production system, this could be loaded from a database or JSON file
    sample_images = [
        {
            'id': 1,
            'name': 'Landsat 8 - Agricultural Area',
            'filename': 'landsat8_agriculture.tif',  # This file would need to be provided
            'description': 'Multispectral Landsat 8 image of an agricultural region showing crop patterns',
            'source': 'USGS Earth Explorer',
            'format': 'GeoTIFF',
            'bands': 7,
            'acquisition_date': '2020-06-15'
        },
        {
            'id': 2,
            'name': 'Sentinel-2 - Coastal Region',
            'filename': 'sentinel2_coastal.tif',  # This file would need to be provided
            'description': 'Sentinel-2 image of a coastal area showing land-water interface',
            'source': 'Copernicus Open Access Hub',
            'format': 'GeoTIFF',
            'bands': 13,
            'acquisition_date': '2021-03-22'
        },
        {
            'id': 3,
            'name': 'MODIS - Forest Cover',
            'filename': 'modis_forest.tif',  # This file would need to be provided
            'description': 'MODIS image showing forest cover and vegetation density',
            'source': 'NASA MODIS Web Service',
            'format': 'GeoTIFF',
            'bands': 7,
            'acquisition_date': '2019-08-10'
        },
        {
            'id': 4,
            'name': 'Urban Development',
            'filename': 'urban_development.tif',  # This file would need to be provided
            'description': 'High-resolution image of an urban area showing development patterns',
            'source': 'Sample Data Repository',
            'format': 'GeoTIFF',
            'bands': 4,
            'acquisition_date': '2021-01-05'
        },
        {
            'id': 5,
            'name': 'Disaster Impact - Flooding',
            'filename': 'flood_impact.tif',  # This file would need to be provided
            'description': 'Before and after images of a region affected by flooding',
            'source': 'Disaster Monitoring Constellation',
            'format': 'GeoTIFF',
            'bands': 3,
            'acquisition_date': '2020-09-18'
        }
    ]
    
    # Filter out samples whose files don't exist
    available_samples = []
    for sample in sample_images:
        file_path = os.path.join('sample_data', sample['filename'])
        if os.path.exists(file_path):
            available_samples.append(sample)
        else:
            logger.warning(f"Sample file not found: {file_path}")
    
    # If no samples are available, add a note to the first sample
    if not available_samples and sample_images:
        sample = sample_images[0].copy()
        sample['name'] += " (FILE NOT FOUND)"
        sample['description'] += " (Note: This sample file is not currently available)"
        available_samples.append(sample)
    
    return available_samples

def get_sample_metadata(sample_id):
    """
    Get detailed metadata for a specific sample image.
    
    Args:
        sample_id (int): ID of the sample image
        
    Returns:
        dict: Detailed metadata for the sample image, or None if not found
    """
    samples = get_sample_images()
    
    for sample in samples:
        if sample['id'] == sample_id:
            # Add more detailed metadata that would be extracted from the file
            if os.path.exists(os.path.join('sample_data', sample['filename'])):
                try:
                    # In a real implementation, we would use rasterio or GDAL to extract
                    # actual metadata from the file. This is a placeholder.
                    if sample['id'] == 1:  # Landsat 8
                        return {
                            **sample,
                            'projection': 'WGS 84 / UTM zone 10N',
                            'pixel_size': 30.0,
                            'nodata_value': 0,
                            'cloud_cover': 2.4,
                            'sun_azimuth': 149.45,
                            'sun_elevation': 63.86,
                            'geometric_accuracy': 12.5,
                            'spectral_bands': {
                                'band_1': 'Coastal/Aerosol (0.433-0.453 µm)',
                                'band_2': 'Blue (0.450-0.515 µm)',
                                'band_3': 'Green (0.525-0.600 µm)',
                                'band_4': 'Red (0.630-0.680 µm)',
                                'band_5': 'Near Infrared (0.845-0.885 µm)',
                                'band_6': 'Short-wave Infrared 1 (1.560-1.660 µm)',
                                'band_7': 'Short-wave Infrared 2 (2.100-2.300 µm)'
                            }
                        }
                    elif sample['id'] == 2:  # Sentinel-2
                        return {
                            **sample,
                            'projection': 'WGS 84 / UTM zone 32N',
                            'pixel_size': 10.0,
                            'nodata_value': 0,
                            'cloud_cover': 1.8,
                            'spectral_bands': {
                                'band_1': 'Coastal aerosol (443 nm)',
                                'band_2': 'Blue (490 nm)',
                                'band_3': 'Green (560 nm)',
                                'band_4': 'Red (665 nm)',
                                'band_5': 'Vegetation red edge (705 nm)',
                                'band_6': 'Vegetation red edge (740 nm)',
                                'band_7': 'Vegetation red edge (783 nm)',
                                'band_8': 'NIR (842 nm)',
                                'band_8a': 'Narrow NIR (865 nm)',
                                'band_9': 'Water vapour (945 nm)',
                                'band_10': 'SWIR - Cirrus (1375 nm)',
                                'band_11': 'SWIR (1610 nm)',
                                'band_12': 'SWIR (2190 nm)'
                            }
                        }
                    else:
                        return sample
                except Exception as e:
                    logger.error(f"Error getting sample metadata: {str(e)}")
                    return sample
            else:
                return sample
    
    return None

def get_sample_by_name(name):
    """
    Get a sample image by its name.
    
    Args:
        name (str): Name of the sample image
        
    Returns:
        dict: Metadata for the sample image, or None if not found
    """
    samples = get_sample_images()
    
    for sample in samples:
        if sample['name'].lower() == name.lower():
            return sample
    
    return None
