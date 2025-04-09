document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const slideshowContainer = document.getElementById('memory-slideshow');
    const memoryPromptContainer = document.getElementById('memory-prompt');
    const memoryJournalContainer = document.getElementById('memory-journal');
    const addPhotoForm = document.getElementById('add-memory-photo-form');
    const addJournalForm = document.getElementById('add-memory-journal-form');
    
    // Initialize slideshow if container exists
    if (slideshowContainer && slideshowContainer.querySelectorAll('.memory-slide').length > 0) {
        initSlideshow();
    }
    
    // Initialize memory prompts if container exists
    if (memoryPromptContainer) {
        generateMemoryPrompt();
    }
    
    // Handle photo upload form
    if (addPhotoForm) {
        const photoInput = addPhotoForm.querySelector('input[type="file"]');
        const photoPreview = document.getElementById('memory-photo-preview');
        
        // Show preview on file selection
        if (photoInput && photoPreview) {
            photoInput.addEventListener('change', function(event) {
                if (event.target.files.length > 0) {
                    const file = event.target.files[0];
                    
                    // Check if file is too large (over 5MB)
                    if (file.size > 5 * 1024 * 1024) {
                        compressImage(file, function(compressedFile) {
                            displayPreview(compressedFile);
                            // Replace the file in the input with compressed version
                            const dataTransfer = new DataTransfer();
                            dataTransfer.items.add(compressedFile);
                            photoInput.files = dataTransfer.files;
                        });
                    } else {
                        displayPreview(file);
                    }
                }
            });
        }
        
        function displayPreview(file) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                photoPreview.innerHTML = `<img src="${e.target.result}" class="img-fluid rounded" alt="Memory photo preview">`;
                photoPreview.classList.remove('d-none');
            };
            
            reader.readAsDataURL(file);
        }
        
        // Image compression function
        function compressImage(file, callback) {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            
            reader.onload = function(e) {
                const img = new Image();
                img.src = e.target.result;
                
                img.onload = function() {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    
                    // Calculate new dimensions while maintaining aspect ratio
                    let width = img.width;
                    let height = img.height;
                    const maxWidth = 1600;
                    const maxHeight = 1200;
                    
                    if (width > height) {
                        if (width > maxWidth) {
                            height *= maxWidth / width;
                            width = maxWidth;
                        }
                    } else {
                        if (height > maxHeight) {
                            width *= maxHeight / height;
                            height = maxHeight;
                        }
                    }
                    
                    canvas.width = width;
                    canvas.height = height;
                    
                    // Draw image on canvas with new dimensions
                    ctx.drawImage(img, 0, 0, width, height);
                    
                    // Get compressed image as Blob
                    canvas.toBlob(function(blob) {
                        // Create a new File object
                        const compressedFile = new File([blob], file.name, {
                            type: 'image/jpeg',
                            lastModified: Date.now()
                        });
                        
                        callback(compressedFile);
                    }, 'image/jpeg', 0.7); // 0.7 quality (70%)
                };
            };
        }
        
        // Handle form submission
        addPhotoForm.addEventListener('submit', function(event) {
            event.preventDefault();
            
            const formData = new FormData(addPhotoForm);
            const progressContainer = document.getElementById('memory-upload-progress');
            const progressBar = document.getElementById('memory-progress-bar');
            
            // Show progress bar
            progressContainer.classList.remove('d-none');
            progressBar.style.width = '0%';
            progressBar.setAttribute('aria-valuenow', 0);
            
            // Create XMLHttpRequest for progress tracking
            const xhr = new XMLHttpRequest();
            xhr.open('POST', addPhotoForm.getAttribute('action'), true);
            
            // Track upload progress
            xhr.upload.onprogress = function(e) {
                if (e.lengthComputable) {
                    const percentComplete = Math.round((e.loaded / e.total) * 100);
                    progressBar.style.width = percentComplete + '%';
                    progressBar.setAttribute('aria-valuenow', percentComplete);
                }
            };
            
            xhr.onload = function() {
                if (xhr.status === 200) {
                    // Show processing status
                    progressBar.style.width = '100%';
                    progressBar.setAttribute('aria-valuenow', 100);
                    
                    $('#add-memory-modal').modal('hide');
                    showToast('Success', 'Memory photo added successfully');
                    
                    // Reset form and preview
                    addPhotoForm.reset();
                    if (photoPreview) {
                        photoPreview.innerHTML = '';
                        photoPreview.classList.add('d-none');
                    }
                    progressContainer.classList.add('d-none');
                    
                    // Reload the page to show the new photo
                    setTimeout(() => {
                        window.location.reload();
                    }, 1500);
                } else {
                    showToast('Error', 'Failed to add memory photo');
                    progressContainer.classList.add('d-none');
                }
            };
            
            xhr.onerror = function() {
                showToast('Error', 'An unexpected error occurred');
                progressContainer.classList.add('d-none');
            };
            
            xhr.send(formData);
        });
    }
    
    // Handle journal entry form
    if (addJournalForm) {
        addJournalForm.addEventListener('submit', function(event) {
            event.preventDefault();
            
            const formData = new FormData(addJournalForm);
            
            fetch('/add_memory_journal', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    $('#add-journal-modal').modal('hide');
                    showToast('Success', 'Memory journal entry added successfully');
                    
                    // Reset form
                    addJournalForm.reset();
                    
                    // Reload the page to show the new entry
                    setTimeout(() => {
                        window.location.reload();
                    }, 1500);
                } else {
                    showToast('Error', data.error || 'Failed to add journal entry');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showToast('Error', 'An unexpected error occurred');
            });
        });
    }
    
    // Slideshow functionality
    function initSlideshow() {
        let slideIndex = 0;
        const slides = slideshowContainer.querySelectorAll('.memory-slide');
        
        // Hide all slides initially
        showSlides();
        
        // Add navigation buttons if there are multiple slides
        if (slides.length > 1) {
            // Previous button
            const prevButton = document.createElement('button');
            prevButton.className = 'slideshow-btn prev-btn';
            prevButton.innerHTML = '<i class="fas fa-chevron-left"></i>';
            prevButton.addEventListener('click', () => {
                slideIndex = (slideIndex - 1 + slides.length) % slides.length;
                showSlides();
            });
            
            // Next button
            const nextButton = document.createElement('button');
            nextButton.className = 'slideshow-btn next-btn';
            nextButton.innerHTML = '<i class="fas fa-chevron-right"></i>';
            nextButton.addEventListener('click', () => {
                slideIndex = (slideIndex + 1) % slides.length;
                showSlides();
            });
            
            slideshowContainer.appendChild(prevButton);
            slideshowContainer.appendChild(nextButton);
            
            // Auto advance slideshow every 8 seconds
            setInterval(() => {
                slideIndex = (slideIndex + 1) % slides.length;
                showSlides();
            }, 8000);
        }
        
        function showSlides() {
            for (let i = 0; i < slides.length; i++) {
                slides[i].style.display = 'none';
            }
            slides[slideIndex].style.display = 'block';
            
            // Update the caption with photo details
            const currentSlide = slides[slideIndex];
            const year = currentSlide.getAttribute('data-year');
            const description = currentSlide.getAttribute('data-description');
            
            let captionText = '';
            if (year) {
                captionText += `<span class="memory-year">${year}</span>`;
            }
            if (description) {
                captionText += `<p class="memory-description">${description}</p>`;
            }
            
            const captionElement = currentSlide.querySelector('.memory-caption');
            if (captionElement) {
                captionElement.innerHTML = captionText;
            }
        }
    }
    
    // Memory prompt functionality
    function generateMemoryPrompt() {
        const prompts = [
            "What was your favorite childhood game?",
            "Tell me about your wedding day.",
            "What was your first job?",
            "Share a memory of your favorite family vacation.",
            "What was your favorite subject in school?",
            "Tell me about your first car.",
            "What was your childhood home like?",
            "Share a memory of a grandparent.",
            "What was a typical family dinner like when you were a child?",
            "Tell me about your favorite teacher.",
            "What games did you play with your friends as a child?",
            "Share a memory of a special celebration or holiday.",
            "What was your favorite music when you were young?",
            "Tell me about your first date.",
            "What was the most significant historical event you lived through?",
            "Share a memory of a special pet.",
            "What was your neighborhood like growing up?"
        ];
        
        // Get a random prompt
        const randomPrompt = prompts[Math.floor(Math.random() * prompts.length)];
        memoryPromptContainer.querySelector('.prompt-text').textContent = randomPrompt;
        
        // Add functionality to refresh button
        const refreshButton = memoryPromptContainer.querySelector('.refresh-prompt');
        if (refreshButton) {
            refreshButton.addEventListener('click', generateMemoryPrompt);
        }
    }
    
    // Toast notification function
    function showToast(title, message) {
        const toastContainer = document.getElementById('toast-container');
        if (!toastContainer) return;
        
        const toastId = 'toast-' + Date.now();
        const toastHtml = `
            <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="toast-header">
                    <strong class="me-auto">${title}</strong>
                    <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        `;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, { delay: 5000 });
        toast.show();
        
        // Remove toast from DOM after it's hidden
        toastElement.addEventListener('hidden.bs.toast', function() {
            toastElement.remove();
        });
    }
}); 