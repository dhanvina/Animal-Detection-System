document.addEventListener('DOMContentLoaded', function() {
    const imageForm = document.getElementById('imageUploadForm');
    const videoForm = document.getElementById('videoUploadForm');
    const imageInput = document.getElementById('imageInput');
    const videoInput = document.getElementById('videoInput');
    const resultSection = document.getElementById('resultSection');
    const previewSection = document.getElementById('previewSection');
    const resultImage = document.getElementById('resultImage');
    const resultVideo = document.getElementById('resultVideo');
    const detectionLog = document.getElementById('detectionLog');
    const loadingSpinner = document.getElementById('loadingSpinner');

    // Handle image upload
    if (imageForm) {
        imageForm.addEventListener('submit', function(e) {
            e.preventDefault();
            if (!imageInput.files.length) return;
            
            const file = imageInput.files[0];
            if (!file.type.startsWith('image/')) {
                alert('Please upload an image file');
                return;
            }
            
            processFile(file, 'image');
        });
    }

    // Handle video upload
    if (videoForm) {
        videoForm.addEventListener('submit', function(e) {
            e.preventDefault();
            if (!videoInput.files.length) return;
            
            const file = videoInput.files[0];
            if (!file.type.startsWith('video/')) {
                alert('Please upload a video file');
                return;
            }
            
            processFile(file, 'video');
        });
    }

    function processFile(file, type) {
        showLoading(true);
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', type);
        
        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            if (type === 'image') {
                previewSection.innerHTML = `<img src="${e.target.result}" class="img-fluid" id="previewImage">`;
            } else {
                previewSection.innerHTML = `
                    <video controls class="img-fluid" id="previewVideo">
                        <source src="${e.target.result}" type="${file.type}">
                        Your browser does not support the video tag.
                    </video>
                `;
            }
            previewSection.style.display = 'block';
        };
        reader.readAsDataURL(file);
        
        // Send to server for processing
        fetch('/detect', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            showLoading(false);
            if (data.error) {
                throw new Error(data.error);
            }
            showResults(data, type);
        })
        .catch(error => {
            console.error('Error:', error);
            showLoading(false);
            alert('Error processing file: ' + error.message);
        });
    }
    
    function showResults(data, type) {
        resultSection.style.display = 'block';
        detectionLog.innerHTML = '';
        
        // Display detection results
        if (data.detections && data.detections.length > 0) {
            data.detections.forEach(det => {
                const logItem = document.createElement('div');
                logItem.className = 'log-item';
                // Use display_name if available, else try det.class.name, else fallback to det.class
                let label = det.display_name || (det.class && det.class.name) || det.class || 'Animal';
                // Add emoji if available (from alert or category)
                let emoji = '';
                if (det.alert && det.alert.match(/\p{Emoji}/u)) {
                    emoji = det.alert.match(/\p{Emoji}/u)[0] + ' ';
                }
                logItem.innerHTML = `
                    <span class="log-label">${emoji}${label}</span>
                    <span class="log-confidence">${(det.confidence * 100).toFixed(2)}%</span>
                `;
                detectionLog.appendChild(logItem);
            });
        } else {
            detectionLog.innerHTML = '<div class="no-detections">No animals detected</div>';
        }
        
        // Show the processed media with bounding boxes
        if (type === 'image' && data.image_url) {
            resultImage.src = data.image_url;
            resultImage.style.display = 'block';
            resultVideo.style.display = 'none';
        } else if (type === 'video' && data.video_url) {
            resultVideo.innerHTML = `
                <source src="${data.video_url}" type="video/mp4">
                Your browser does not support the video tag.
            `;
            resultVideo.style.display = 'block';
            resultImage.style.display = 'none';
            resultVideo.load();
        }
    }
    
    function showLoading(show) {
        if (show) {
            loadingSpinner.style.display = 'block';
        } else {
            loadingSpinner.style.display = 'none';
        }
    }
});
