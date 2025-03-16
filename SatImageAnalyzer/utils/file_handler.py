# utils/file_handler.py
# Module for handling file operations in the satellite imagery analysis application

import os
import uuid
import logging
import shutil
from werkzeug.utils import secure_filename
from config import get_config

# Configure logger
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

def create_directories():
    """
    Create necessary directories for file storage.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(config.RESULTS_FOLDER, exist_ok=True)
        logger.info(f"Created storage directories: {config.UPLOAD_FOLDER}, {config.RESULTS_FOLDER}")
        return True
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        return False

def is_allowed_file(filename):
    """
    Check if a file has an allowed extension.
    
    Args:
        filename (str): Name of the file to check
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.SUPPORTED_FORMATS

def save_uploaded_file(file):
    """
    Save an uploaded file to the upload folder.
    
    Args:
        file: File object from Flask request.files
        
    Returns:
        tuple: (success (bool), file_path (str), error_message (str))
    """
    if file.filename == '':
        logger.warning("No file selected for uploading")
        return False, None, "No file selected"
    
    if not is_allowed_file(file.filename):
        logger.warning(f"File {file.filename} has an unsupported format")
        return False, None, f"Unsupported file format. Allowed formats: {', '.join(config.SUPPORTED_FORMATS)}"
    
    try:
        # Secure the filename and add a unique identifier
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(config.UPLOAD_FOLDER, unique_name)
        
        # Save the file
        file.save(file_path)
        logger.info(f"Saved uploaded file to {file_path}")
        
        return True, file_path, None
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        return False, None, f"Error saving file: {str(e)}"

def delete_file(file_path):
    """
    Delete a file from the file system.
    
    Args:
        file_path (str): Path to the file to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
            return True
        else:
            logger.warning(f"File not found for deletion: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {str(e)}")
        return False

def create_result_directory(analysis_id):
    """
    Create a directory to store analysis results.
    
    Args:
        analysis_id: Identifier for the analysis
        
    Returns:
        str: Path to the results directory
    """
    result_dir = os.path.join(config.RESULTS_FOLDER, str(analysis_id))
    os.makedirs(result_dir, exist_ok=True)
    logger.debug(f"Created results directory: {result_dir}")
    return result_dir

def save_result_file(data, filename, analysis_id):
    """
    Save a result file to the appropriate directory.
    
    Args:
        data: Data to save (can be binary or text)
        filename (str): Name for the saved file
        analysis_id: Identifier for the analysis
        
    Returns:
        tuple: (success (bool), file_path (str), error_message (str))
    """
    try:
        # Create results directory if it doesn't exist
        result_dir = create_result_directory(analysis_id)
        
        # Create full file path
        file_path = os.path.join(result_dir, filename)
        
        # Determine whether to use binary or text mode
        is_binary = isinstance(data, bytes)
        
        # Write the file
        with open(file_path, 'wb' if is_binary else 'w') as f:
            f.write(data)
        
        logger.info(f"Saved result file to {file_path}")
        return True, file_path, None
    except Exception as e:
        logger.error(f"Error saving result file: {str(e)}")
        return False, None, f"Error saving result: {str(e)}"

def get_sample_data_path(sample_name):
    """
    Get the file path for a sample data file.
    
    Args:
        sample_name (str): Name of the sample file
        
    Returns:
        str: Full path to the sample file
    """
    # Sample data directory is at the root of the project
    sample_dir = 'sample_data'
    
    # Construct the full path
    sample_path = os.path.join(sample_dir, sample_name)
    
    # Check if the file exists
    if not os.path.exists(sample_path):
        logger.warning(f"Sample file not found: {sample_path}")
    
    return sample_path

def copy_file(src_path, dest_path):
    """
    Copy a file from source to destination.
    
    Args:
        src_path (str): Source file path
        dest_path (str): Destination file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy the file
        shutil.copy2(src_path, dest_path)
        logger.debug(f"Copied file from {src_path} to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Error copying file from {src_path} to {dest_path}: {str(e)}")
        return False

def list_directory_files(directory, filter_extensions=None):
    """
    List all files in a directory with optional extension filtering.
    
    Args:
        directory (str): Directory to list files from
        filter_extensions (list, optional): List of file extensions to include
        
    Returns:
        list: List of file paths
    """
    try:
        if not os.path.exists(directory):
            logger.warning(f"Directory does not exist: {directory}")
            return []
        
        files = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
            
            # Apply extension filter if provided
            if filter_extensions:
                ext = os.path.splitext(filename)[1].lower()[1:]  # Remove the dot
                if ext not in filter_extensions:
                    continue
            
            files.append(file_path)
        
        return files
    except Exception as e:
        logger.error(f"Error listing files in directory {directory}: {str(e)}")
        return []

def get_file_size(file_path):
    """
    Get the size of a file in bytes.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        int: File size in bytes, or 0 if file not found
    """
    try:
        if os.path.exists(file_path):
            return os.path.getsize(file_path)
        else:
            logger.warning(f"File not found: {file_path}")
            return 0
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {str(e)}")
        return 0

# Initialize directories when the module is imported
create_directories()
