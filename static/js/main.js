document.addEventListener('DOMContentLoaded', function() {
    // Global variables
    let currentImageId = null;
    let metricsChart = null;
    
    // Elements
    const uploadForm = document.getElementById('uploadForm');
    const processingForm = document.getElementById('processingForm');
    const imageUpload = document.getElementById('imageUpload');
    const uploadPreview = document.getElementById('uploadPreview');
    const processingSection = document.getElementById('processingSection');
    const resultsSection = document.getElementById('resultsSection');
    const noiseLevel = document.getElementById('noiseLevel');
    const noiseLevelValue = document.getElementById('noiseLevelValue');
    const usePatches = document.getElementById('usePatches');
    const patchOptions = document.getElementById('patchOptions');
    
    const originalImage = document.getElementById('originalImage');
    const noisyImage = document.getElementById('noisyImage');
    const denoisedImage = document.getElementById('denoisedImage');
    const downloadBtn = document.getElementById('downloadBtn');
    
    const psnrValue = document.getElementById('psnrValue');
    const ssimValue = document.getElementById('ssimValue');
    const processingTime = document.getElementById('processingTime');
    
    const statusToast = document.getElementById('statusToast');
    const statusMessage = document.getElementById('statusMessage');
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    
    // Event listeners
    imageUpload.addEventListener('change', previewImage);
    uploadForm.addEventListener('submit', uploadImage);
    processingForm.addEventListener('submit', processImage);
    noiseLevel.addEventListener('input', updateNoiseLevel);
    usePatches.addEventListener('change', togglePatchOptions);
    
    // Connect Browse Files button to file input
    const browseBtn = document.getElementById('browseBtn');
    if (browseBtn && imageUpload) {
        browseBtn.addEventListener('click', function() {
            imageUpload.click(); // This triggers the file dialog
        });
    }
    
    // Add drag and drop functionality for the upload zone
    const dropZone = document.getElementById('dropZone');
    if (dropZone && imageUpload) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        // Highlight drop zone when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropZone.classList.add('highlight');
        }
        
        function unhighlight() {
            dropZone.classList.remove('highlight');
        }
        
        // Handle dropped files
        dropZone.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length) {
                imageUpload.files = files; // Transfer files to the input
                handleFiles(files);
            }
        }
        
        function handleFiles(files) {
            if (files.length) {
                previewImage(files[0]);
            }
        }
        
        // Update the existing previewImage function to work with a file parameter
        function previewImage(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const uploadPreview = document.getElementById('uploadPreview');
                if (uploadPreview) {
                    uploadPreview.src = e.target.result;
                    uploadPreview.classList.add('preview-active');
                    
                    // Update preview text
                    const previewText = document.querySelector('.preview-text');
                    if (previewText) {
                        previewText.textContent = file.name;
                    }
                }
            }
            reader.readAsDataURL(file);
        }
        
        // Connect the file input change event to the preview function
        imageUpload.addEventListener('change', function() {
            if (this.files.length) {
                previewImage(this.files[0]);
            }
        });
    }
    
    // Preview the selected image
    function previewImage() {
        const file = imageUpload.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadPreview.src = e.target.result;
                uploadPreview.classList.add('highlight');
            }
            reader.readAsDataURL(file);
        }
    }
    
    // Update noise level display
    function updateNoiseLevel() {
        noiseLevelValue.textContent = noiseLevel.value;
    }
    
    // Toggle patch options visibility
    function togglePatchOptions() {
        patchOptions.style.display = usePatches.checked ? 'block' : 'none';
    }
    
    // Upload image to server
    function uploadImage(e) {
        e.preventDefault();
        
        const file = imageUpload.files[0];
        if (!file) {
            showToast('Please select an image file first', 'error');
            return;
        }
        
        // Show loading state on button
        const uploadBtn = document.getElementById('uploadBtn');
        const uploadSpinner = document.getElementById('uploadSpinner');
        const uploadBtnText = document.getElementById('uploadBtnText');
        
        uploadBtn.disabled = true;
        uploadSpinner.classList.remove('d-none');
        uploadBtnText.textContent = 'Uploading...';
        
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                currentImageId = data.image_id;
                originalImage.src = data.original_url;
                processingSection.style.display = 'block';
                showToast('Image uploaded successfully', 'success');
            } else {
                showToast(`Error: ${data.error}`, 'error');
            }
        })
        .catch(error => {
            showToast(`Upload error: ${error.message}`, 'error');
        })
        .finally(() => {
            // Reset button state
            uploadBtn.disabled = false;
            uploadSpinner.classList.add('d-none');
            uploadBtnText.textContent = 'Upload Image';
        });
    }
    
    // Process the image (add noise and denoise)
    function processImage(e) {
        e.preventDefault();
        
        if (!currentImageId) {
            showToast('Please upload an image first', 'error');
            return;
        }
        
        const requestData = {
            image_id: currentImageId,
            noise_type: document.getElementById('noiseType').value,
            noise_level: noiseLevel.value,
            use_patches: usePatches.checked,
            patch_size: document.getElementById('patchSize').value,
            patch_overlap: document.getElementById('overlap').value
        };
        
        showLoadingModal('Processing image...', 'This may take several seconds for large images');
        
        // Add timeout for processing as well
        const processingTimeout = setTimeout(() => {
            loadingModal.hide();
            showToast('Processing is taking longer than expected', 'warning');
        }, 30000); // 30 seconds timeout for processing
        
        fetch('/denoise', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            clearTimeout(processingTimeout);
            if (data.success) {
                displayResults(data);
                resultsSection.style.display = 'block';
                showToast('Image processed successfully', 'success');
            } else {
                showToast(`Error: ${data.error}`, 'error');
            }
        })
        .catch(error => {
            clearTimeout(processingTimeout);
            showToast(`Processing error: ${error.message}`, 'error');
        })
        .finally(() => {
            loadingModal.hide(); // Ensure loading modal is hidden regardless of outcome
        });
    }
    
    // Display results
    function displayResults(data) {
        noisyImage.src = data.noisy_url;
        denoisedImage.src = data.denoised_url;
        downloadBtn.href = data.denoised_url;
        
        const metrics = data.metrics;
        psnrValue.textContent = metrics.psnr.toFixed(2) + ' dB';
        ssimValue.textContent = metrics.ssim.toFixed(4);
        processingTime.textContent = metrics.processing_time.toFixed(2) + ' seconds';
        
        // Update or create the chart
        updateMetricsChart(metrics);
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Update metrics chart
    function updateMetricsChart(metrics) {
        if (metricsChart) {
            metricsChart.destroy();
        }
        
        const ctx = document.getElementById('metricsChart').getContext('2d');
        metricsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['PSNR (dB)', 'SSIM'],
                datasets: [{
                    label: 'Image Quality Metrics',
                    data: [metrics.psnr, metrics.ssim],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.5)',
                        'rgba(75, 192, 192, 0.5)'
                    ],
                    borderColor: [
                        'rgba(54, 162, 235, 1)',
                        'rgba(75, 192, 192, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Value'
                        }
                    }
                }
            }
        });
    }
    
    // Show toast notification
    function showToast(message, type = 'success') {
        statusMessage.textContent = message;
        
        // Set toast class based on message type
        const toast = bootstrap.Toast.getOrCreateInstance(statusToast);
        
        // Remove previous classes
        statusToast.classList.remove('bg-success', 'bg-danger', 'bg-warning', 'text-white');
        
        // Add appropriate class
        if (type === 'error') {
            statusToast.classList.add('bg-danger', 'text-white');
        } else if (type === 'warning') {
            statusToast.classList.add('bg-warning');
        } else {
            statusToast.classList.add('bg-success', 'text-white');
        }
        
        toast.show();
    }
    
    // Show loading modal
    function showLoadingModal(text, details = '') {
        document.getElementById('loadingText').textContent = text;
        document.getElementById('loadingDetails').textContent = details;
        loadingModal.show();
    }
    
    // Initial setup
    updateNoiseLevel();
    togglePatchOptions();
});