"""
Satellite Image Loader Module
-----------------------------

This module provides functions for loading satellite imagery from various file formats
and extracting metadata from the files.

Supported formats:
- GeoTIFF (.tif, .tiff)
- JPEG 2000 (.jp2)
- ERDAS IMAGINE (.img)
- HDF (.hdf)
- NetCDF (.nc)

Dependencies:
- rasterio: For reading raster data
- gdal: For geospatial data handling
- numpy: For array operations
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from osgeo import gdal, osr
from pyproj import CRS
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

def load_image(file_path):
    """
    Load satellite imagery from supported file formats.
    
    Parameters:
    -----------
    file_path : str
        Path to the satellite image file
        
    Returns:
    --------
    numpy.ndarray
        The image data as a NumPy array
        
    Raises:
    -------
    ValueError
        If the file format is not supported or the file cannot be read
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # Try to open with rasterio first (supports most geo formats)
        with rasterio.open(file_path) as src:
            # Read all bands
            img_data = src.read()
            
            # Rasterio reads as (bands, height, width), but we want (height, width, bands)
            # for consistency with OpenCV and other image processing libraries
            if img_data.shape[0] > 1:  # If multi-band
                img_data = np.transpose(img_data, (1, 2, 0))
            else:  # If single band
                img_data = img_data[0]
                
            logger.info(f"Loaded image with shape {img_data.shape} from {file_path}")
            return img_data
            
    except RasterioIOError:
        # If rasterio fails, try with GDAL
        try:
            gdal_dataset = gdal.Open(file_path)
            if gdal_dataset is None:
                raise ValueError(f"Unable to open {file_path} with GDAL")
            
            # Get dimensions
            width = gdal_dataset.RasterXSize
            height = gdal_dataset.RasterYSize
            bands = gdal_dataset.RasterCount
            
            # Read all bands
            if bands > 1:
                img_data = np.empty((height, width, bands), dtype=np.float32)
                for i in range(bands):
                    band = gdal_dataset.GetRasterBand(i + 1)
                    img_data[:, :, i] = band.ReadAsArray()
            else:
                band = gdal_dataset.GetRasterBand(1)
                img_data = band.ReadAsArray()
            
            logger.info(f"Loaded image with shape {img_data.shape} from {file_path} using GDAL")
            return img_data
            
        except Exception as e:
            # If both methods fail, try a simpler approach for common image formats
            if file_ext in ['.png', '.jpg', '.jpeg']:
                import cv2
                try:
                    img_data = cv2.imread(file_path)
                    if img_data is not None:
                        # Convert from BGR to RGB
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                        logger.info(f"Loaded image with shape {img_data.shape} from {file_path} using OpenCV")
                        return img_data
                except Exception as cv_error:
                    logger.error(f"OpenCV failed to load {file_path}: {str(cv_error)}")
            
            # If all methods fail, raise error
            logger.error(f"Failed to load {file_path}: {str(e)}")
            raise ValueError(f"Unable to load image file {file_path}. Error: {str(e)}")

def extract_metadata(file_path):
    """
    Extract metadata from satellite imagery file.
    
    Parameters:
    -----------
    file_path : str
        Path to the satellite image file
        
    Returns:
    --------
    dict
        Dictionary containing metadata information
        
    Notes:
    ------
    The exact metadata available will depend on the file format and
    what information is stored in the file. Common metadata includes:
    - Dimensions (width, height)
    - Coordinate reference system
    - Geotransform information
    - Acquisition date (if available)
    - Sensor information (if available)
    """
    metadata = {
        'filename': os.path.basename(file_path),
        'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
        'file_format': os.path.splitext(file_path)[1],
        'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        # Try with rasterio first
        with rasterio.open(file_path) as src:
            metadata.update({
                'width': src.width,
                'height': src.height,
                'bands': src.count,
                'crs': str(src.crs) if src.crs else 'Not specified',
                'transform': str(src.transform) if src.transform else 'Not specified',
                'bounds': str(src.bounds) if src.bounds else 'Not specified',
                'nodata_value': src.nodata
            })
            
            # Extract band information
            metadata['band_dtypes'] = []
            metadata['band_min_values'] = []
            metadata['band_max_values'] = []
            metadata['band_mean_values'] = []
            
            for i in range(1, src.count + 1):
                band = src.read(i)
                metadata['band_dtypes'].append(str(band.dtype))
                metadata['band_min_values'].append(float(band.min()))
                metadata['band_max_values'].append(float(band.max()))
                metadata['band_mean_values'].append(float(band.mean()))
            
            # Check for common tags
            if src.tags():
                metadata['tags'] = src.tags()
                
                # Look for acquisition date
                if 'TIFFTAG_DATETIME' in src.tags():
                    metadata['acquisition_date'] = src.tags()['TIFFTAG_DATETIME']
                
    except Exception as e:
        logger.warning(f"Rasterio failed to extract metadata: {str(e)}")
        # If rasterio fails, try with GDAL
        try:
            gdal_dataset = gdal.Open(file_path)
            if gdal_dataset is not None:
                metadata.update({
                    'width': gdal_dataset.RasterXSize,
                    'height': gdal_dataset.RasterYSize,
                    'bands': gdal_dataset.RasterCount
                })
                
                # Get geotransform
                geotransform = gdal_dataset.GetGeoTransform()
                if geotransform:
                    metadata['geotransform'] = geotransform
                
                # Get projection
                projection = gdal_dataset.GetProjection()
                if projection:
                    srs = osr.SpatialReference()
                    srs.ImportFromWkt(projection)
                    metadata['projection'] = srs.ExportToProj4()
                
                # Get metadata from GDAL
                gdal_metadata = gdal_dataset.GetMetadata()
                if gdal_metadata:
                    metadata['gdal_metadata'] = gdal_metadata
                    
                    # Look for acquisition date in common metadata fields
                    date_keys = ['ACQUISITIONDATE', 'ACQUISITION_DATE', 'DATE_ACQUIRED']
                    for key in date_keys:
                        if key in gdal_metadata:
                            metadata['acquisition_date'] = gdal_metadata[key]
                            break
                
                # If no acquisition date found, look in subdatasets
                if 'acquisition_date' not in metadata:
                    subdatasets = gdal_dataset.GetSubDatasets()
                    if subdatasets:
                        for subds in subdatasets:
                            subds_name = subds[0]
                            try:
                                subds_dataset = gdal.Open(subds_name)
                                if subds_dataset:
                                    subds_metadata = subds_dataset.GetMetadata()
                                    for key in date_keys:
                                        if key in subds_metadata:
                                            metadata['acquisition_date'] = subds_metadata[key]
                                            break
                            except Exception:
                                pass
                            
        except Exception as gdal_error:
            logger.warning(f"GDAL failed to extract metadata: {str(gdal_error)}")
    
    return metadata

def load_sample_data(sample_name='landsat'):
    """
    Load a sample satellite image dataset for testing and demonstration.
    
    Parameters:
    -----------
    sample_name : str
        Name of the sample dataset to load
        
    Returns:
    --------
    tuple
        (numpy.ndarray, dict) - The image data and its metadata
        
    Notes:
    ------
    Available sample datasets:
    - 'landsat': A sample Landsat 8 image
    - 'sentinel': A sample Sentinel-2 image
    - 'modis': A sample MODIS image
    """
    # This is a placeholder function - in a real application, this would
    # load actual sample data from files or a remote repository
    
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static/sample_data')
    
    if sample_name == 'landsat':
        file_path = os.path.join(sample_dir, 'landsat_sample.tif')
    elif sample_name == 'sentinel':
        file_path = os.path.join(sample_dir, 'sentinel_sample.jp2')
    elif sample_name == 'modis':
        file_path = os.path.join(sample_dir, 'modis_sample.hdf')
    else:
        raise ValueError(f"Unknown sample dataset: {sample_name}")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        # If the file doesn't exist, create a simple synthetic image for demo purposes
        logger.warning(f"Sample file {file_path} not found. Creating synthetic data.")
        
        # Create a 500x500 image with 4 bands (R, G, B, NIR)
        height, width = 500, 500
        synthetic_data = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Add some features to make it look satellite-like
        # Create a gradient background
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Red band - varies with X
        synthetic_data[:, :, 0] = (X * 200).astype(np.uint8)
        
        # Green band - varies with Y
        synthetic_data[:, :, 1] = (Y * 200).astype(np.uint8)
        
        # Blue band - constant with circular features
        synthetic_data[:, :, 2] = 100
        center_x, center_y = width // 2, height // 2
        for i in range(height):
            for j in range(width):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                if dist < 100:
                    synthetic_data[i, j, 2] = 200
        
        # NIR band - vegetation-like features
        synthetic_data[:, :, 3] = (synthetic_data[:, :, 1] * 1.5).clip(0, 255).astype(np.uint8)
        
        # Add some noise
        noise = np.random.randint(0, 30, (height, width, 4), dtype=np.uint8)
        synthetic_data = np.clip(synthetic_data.astype(np.int16) + noise - 15, 0, 255).astype(np.uint8)
        
        # Create metadata
        synthetic_metadata = {
            'filename': f'{sample_name}_synthetic.tif',
            'file_size_mb': 'N/A (synthetic)',
            'width': width,
            'height': height,
            'bands': 4,
            'band_names': ['Red', 'Green', 'Blue', 'NIR'],
            'crs': 'EPSG:4326 (synthetic)',
            'acquisition_date': datetime.now().strftime('%Y-%m-%d'),
            'note': 'This is synthetic data created for demonstration purposes'
        }
        
        return synthetic_data, synthetic_metadata
    
    # If the file exists, load it normally
    img_data = load_image(file_path)
    metadata = extract_metadata(file_path)
    
    return img_data, metadata
