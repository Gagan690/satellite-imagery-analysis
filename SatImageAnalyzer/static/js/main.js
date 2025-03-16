/**
 * main.js
 * Main JavaScript file for the Satellite Imagery Analysis Tool
 */

// Execute when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // File upload preview
    const fileInput = document.getElementById('file-input');
    const filePreview = document.getElementById('file-preview');
    const fileInfo = document.getElementById('file-info');
    
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            // Clear previous preview
            if (filePreview) {
                filePreview.innerHTML = '';
            }
            
            if (fileInfo) {
                fileInfo.innerHTML = '';
            }
            
            // Check if file is selected
            if (this.files && this.files[0]) {
                const file = this.files[0];
                
                // Check file type
                const fileExt = file.name.split('.').pop().toLowerCase();
                const supportedFormats = ['tif', 'tiff', 'jp2', 'png', 'jpg', 'jpeg'];
                
                if (!supportedFormats.includes(fileExt)) {
                    showAlert('Unsupported file format. Please upload an image file.', 'danger');
                    this.value = '';
                    return;
                }
                
                // Show file info
                if (fileInfo) {
                    const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
                    fileInfo.innerHTML = `
                        <div class="mt-3">
                            <h5>File Information:</h5>
                            <p><strong>Name:</strong> ${file.name}</p>
                            <p><strong>Size:</strong> ${fileSizeMB} MB</p>
                            <p><strong>Type:</strong> ${file.type || 'Unknown'}</p>
                        </div>
                    `;
                }
                
                // Only try to preview if it's a standard web image format
                if (['png', 'jpg', 'jpeg'].includes(fileExt) && filePreview) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        filePreview.innerHTML = `
                            <div class="mt-3">
                                <h5>Preview:</h5>
                                <img src="${e.target.result}" class="img-fluid img-thumbnail" alt="Preview">
                                <p class="text-muted small mt-2">Note: This is a simplified preview and may not represent all bands of the satellite image.</p>
                            </div>
                        `;
                    };
                    reader.readAsDataURL(file);
                } else if (filePreview) {
                    filePreview.innerHTML = `
                        <div class="mt-3">
                            <h5>Preview:</h5>
                            <div class="alert alert-info">
                                <i class="feather feather-info"></i> 
                                Preview not available for this file format (${fileExt.toUpperCase()}).
                                This is a specialized satellite image format that will be processed after upload.
                            </div>
                        </div>
                    `;
                }
            }
        });
    }
    
    // Processing technique selection
    const processingSelect = document.getElementById('processing-technique');
    const processingOptions = document.getElementById('processing-options');
    
    if (processingSelect && processingOptions) {
        processingSelect.addEventListener('change', function() {
            const selectedTechnique = this.value;
            
            // Clear previous options
            processingOptions.innerHTML = '';
            
            // Show options based on selected technique
            if (selectedTechnique === 'histogram_equalization') {
                processingOptions.innerHTML = `
                    <div class="form-text mb-3">
                        Enhances contrast by equalizing the image histogram, making features more visible.
                    </div>
                `;
            } else if (selectedTechnique === 'gaussian_blur') {
                processingOptions.innerHTML = `
                    <div class="form-group mb-3">
                        <label for="sigma">Blur Intensity (Sigma):</label>
                        <input type="range" class="form-range" id="sigma" name="sigma" 
                               min="0.1" max="5.0" step="0.1" value="1.0">
                        <div class="d-flex justify-content-between">
                            <span>Low (0.1)</span>
                            <span id="sigma-value">1.0</span>
                            <span>High (5.0)</span>
                        </div>
                    </div>
                    <div class="form-text mb-3">
                        Applies Gaussian smoothing to reduce noise in the image.
                    </div>
                `;
                
                // Update displayed value when slider is moved
                document.getElementById('sigma').addEventListener('input', function() {
                    document.getElementById('sigma-value').textContent = this.value;
                });
            } else if (selectedTechnique === 'edge_detection') {
                processingOptions.innerHTML = `
                    <div class="form-group mb-3">
                        <label for="edge-method">Edge Detection Method:</label>
                        <select class="form-select" id="edge-method" name="edge_method">
                            <option value="canny">Canny</option>
                            <option value="sobel">Sobel</option>
                            <option value="roberts">Roberts</option>
                            <option value="prewitt">Prewitt</option>
                        </select>
                    </div>
                    <div class="form-text mb-3">
                        Detects edges in the image to identify boundaries between different features.
                    </div>
                `;
            } else if (selectedTechnique === 'ndvi_calculation') {
                processingOptions.innerHTML = `
                    <div class="form-group mb-3">
                        <label for="red-band">Red Band Index:</label>
                        <input type="number" class="form-control" id="red-band" name="red_band" 
                               min="0" value="2" required>
                    </div>
                    <div class="form-group mb-3">
                        <label for="nir-band">NIR Band Index:</label>
                        <input type="number" class="form-control" id="nir-band" name="nir_band" 
                               min="0" value="3" required>
                    </div>
                    <div class="form-text mb-3">
                        Calculates Normalized Difference Vegetation Index to measure vegetation health.
                        <br>Typically uses Red (band 2) and Near-Infrared (band 3) in Landsat imagery.
                    </div>
                `;
            } else if (selectedTechnique === 'rgb_composite') {
                processingOptions.innerHTML = `
                    <div class="form-group mb-3">
                        <label for="r-band">Red Channel Band:</label>
                        <input type="number" class="form-control" id="r-band" name="r_band" 
                               min="0" value="2" required>
                    </div>
                    <div class="form-group mb-3">
                        <label for="g-band">Green Channel Band:</label>
                        <input type="number" class="form-control" id="g-band" name="g_band" 
                               min="0" value="1" required>
                    </div>
                    <div class="form-group mb-3">
                        <label for="b-band">Blue Channel Band:</label>
                        <input type="number" class="form-control" id="b-band" name="b_band" 
                               min="0" value="0" required>
                    </div>
                    <div class="form-text mb-3">
                        Creates an RGB composite image using three selected bands. This allows for 
                        visualization of non-visible bands as visible colors.
                    </div>
                `;
            } else if (selectedTechnique === 'unsupervised_classification') {
                processingOptions.innerHTML = `
                    <div class="form-group mb-3">
                        <label for="n-clusters">Number of Classes:</label>
                        <input type="range" class="form-range" id="n-clusters" name="n_clusters" 
                               min="2" max="10" step="1" value="5">
                        <div class="d-flex justify-content-between">
                            <span>2</span>
                            <span id="n-clusters-value">5</span>
                            <span>10</span>
                        </div>
                    </div>
                    <div class="form-text mb-3">
                        Performs unsupervised classification to automatically identify different land cover types.
                    </div>
                `;
                
                // Update displayed value when slider is moved
                document.getElementById('n-clusters').addEventListener('input', function() {
                    document.getElementById('n-clusters-value').textContent = this.value;
                });
            }
        });
        
        // Trigger change event to initialize the first option
        const event = new Event('change');
        processingSelect.dispatchEvent(event);
    }
    
    // Feature extraction selection
    const featureSelect = document.getElementById('feature-extraction');
    const featureOptions = document.getElementById('feature-options');
    
    if (featureSelect && featureOptions) {
        featureSelect.addEventListener('change', function() {
            const selectedFeature = this.value;
            
            // Clear previous options
            featureOptions.innerHTML = '';
            
            // Show options based on selected feature
            if (selectedFeature === 'vegetation_indices') {
                featureOptions.innerHTML = `
                    <div class="form-group mb-3">
                        <label for="index-type">Index Type:</label>
                        <select class="form-select" id="index-type" name="index_type">
                            <option value="ndvi">NDVI - Normalized Difference Vegetation Index</option>
                            <option value="evi">EVI - Enhanced Vegetation Index</option>
                            <option value="savi">SAVI - Soil Adjusted Vegetation Index</option>
                            <option value="ndwi">NDWI - Normalized Difference Water Index</option>
                            <option value="all">All Indices</option>
                        </select>
                    </div>
                `;
            } else if (selectedFeature === 'urban_detection') {
                featureOptions.innerHTML = `
                    <div class="form-group mb-3">
                        <label for="urban-method">Detection Method:</label>
                        <select class="form-select" id="urban-method" name="urban_method">
                            <option value="ndbi">NDBI (Normalized Difference Built-up Index)</option>
                            <option value="texture">Texture Analysis</option>
                            <option value="clustering">Clustering</option>
                        </select>
                    </div>
                `;
            } else if (selectedFeature === 'water_detection') {
                featureOptions.innerHTML = `
                    <div class="form-group mb-3">
                        <label for="water-method">Detection Method:</label>
                        <select class="form-select" id="water-method" name="water_method">
                            <option value="ndwi">NDWI</option>
                            <option value="modified_ndwi">Modified NDWI</option>
                            <option value="threshold">Simple Thresholding</option>
                        </select>
                    </div>
                `;
            } else if (selectedFeature === 'object_detection') {
                featureOptions.innerHTML = `
                    <div class="form-group mb-3">
                        <label for="object-method">Detection Method:</label>
                        <select class="form-select" id="object-method" name="object_method">
                            <option value="contour">Contour Detection</option>
                            <option value="blob">Blob Detection</option>
                            <option value="watershed">Watershed Segmentation</option>
                        </select>
                    </div>
                    <div class="form-group mb-3">
                        <label for="min-size">Minimum Object Size (pixels):</label>
                        <input type="number" class="form-control" id="min-size" name="min_size" 
                               min="1" value="100" required>
                    </div>
                `;
            } else if (selectedFeature === 'texture_analysis') {
                featureOptions.innerHTML = `
                    <div class="form-group mb-3">
                        <label for="texture-method">Analysis Method:</label>
                        <select class="form-select" id="texture-method" name="texture_method">
                            <option value="glcm">GLCM (Gray Level Co-occurrence Matrix)</option>
                            <option value="lbp">LBP (Local Binary Pattern)</option>
                            <option value="gabor">Gabor Filter</option>
                        </select>
                    </div>
                `;
            }
        });
        
        // Trigger change event to initialize the first option
        if (featureSelect.value) {
            const event = new Event('change');
            featureSelect.dispatchEvent(event);
        }
    }
    
    // Form validation
    const analysisForm = document.getElementById('analysis-form');
    if (analysisForm) {
        analysisForm.addEventListener('submit', function(event) {
            if (!validateAnalysisForm()) {
                event.preventDefault();
            } else {
                // Show loading spinner
                document.getElementById('loading-overlay').style.display = 'flex';
            }
        });
    }
    
    // Image comparison slider
    initializeComparisonSliders();
    
    // Initialize any tooltips
    initTooltips();
});

/**
 * Display an alert message
 * @param {string} message - The message to display
 * @param {string} type - Alert type: success, info, warning, danger
 */
function showAlert(message, type = 'info') {
    const alertsContainer = document.getElementById('alerts-container');
    if (!alertsContainer) return;
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.setAttribute('role', 'alert');
    
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertsContainer.appendChild(alert);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alert.classList.remove('show');
        setTimeout(() => {
            alert.remove();
        }, 150);
    }, 5000);
}

/**
 * Validate the analysis form before submission
 * @returns {boolean} - Whether the form is valid
 */
function validateAnalysisForm() {
    const fileInput = document.getElementById('file-input');
    
    if (!fileInput || fileInput.files.length === 0) {
        showAlert('Please select an image file to analyze.', 'danger');
        return false;
    }
    
    return true;
}

/**
 * Initialize before-after image comparison sliders
 */
function initializeComparisonSliders() {
    const comparisons = document.querySelectorAll('.image-comparison');
    
    comparisons.forEach(comparison => {
        const slider = comparison.querySelector('.comparison-slider');
        const beforeImage = comparison.querySelector('.before-image');
        const afterImage = comparison.querySelector('.after-image');
        
        if (!slider || !beforeImage || !afterImage) return;
        
        // Set initial position
        beforeImage.style.width = '50%';
        slider.style.left = '50%';
        
        // Add event listeners for mouse/touch interaction
        slider.addEventListener('mousedown', startDragging);
        slider.addEventListener('touchstart', startDragging);
        
        function startDragging(e) {
            e.preventDefault();
            
            // Add event listeners for drag and end
            document.addEventListener('mousemove', drag);
            document.addEventListener('touchmove', drag);
            document.addEventListener('mouseup', endDrag);
            document.addEventListener('touchend', endDrag);
        }
        
        function drag(e) {
            e.preventDefault();
            
            // Get cursor position
            let x;
            if (e.type === 'touchmove') {
                x = e.touches[0].clientX;
            } else {
                x = e.clientX;
            }
            
            // Calculate position relative to the comparison container
            const rect = comparison.getBoundingClientRect();
            const containerWidth = rect.width;
            x = x - rect.left;
            
            // Constrain to container bounds
            let percent = (x / containerWidth) * 100;
            percent = Math.max(0, Math.min(100, percent));
            
            // Update slider and before image position
            beforeImage.style.width = `${percent}%`;
            slider.style.left = `${percent}%`;
        }
        
        function endDrag() {
            // Remove event listeners
            document.removeEventListener('mousemove', drag);
            document.removeEventListener('touchmove', drag);
            document.removeEventListener('mouseup', endDrag);
            document.removeEventListener('touchend', endDrag);
        }
    });
}

/**
 * Initialize Bootstrap tooltips
 */
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    if (tooltipTriggerList.length > 0) {
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}
