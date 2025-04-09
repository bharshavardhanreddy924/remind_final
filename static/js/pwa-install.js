/**
 * Progressive Web App Installation & Enhancement Script
 * Handles PWA installation, detects standalone mode, and improves user experience
 */

(function() {
    'use strict';
    
    // Variables
    let deferredPrompt;
    const installContainer = document.getElementById('install-container');
    const installButton = document.getElementById('install-button');
    
    // Check if the app is running in standalone mode
    function isRunningStandalone() {
        return (window.matchMedia('(display-mode: standalone)').matches) ||
               (window.navigator.standalone) ||
               document.referrer.includes('android-app://');
    }
    
    // Apply PWA styling if running as installed app
    function applyPWAStyling() {
        if (isRunningStandalone()) {
            document.documentElement.classList.add('pwa-standalone');
            document.body.classList.add('pwa-standalone');
            
            // Hide any browser-only elements
            const browserOnlyElements = document.querySelectorAll('.browser-only');
            browserOnlyElements.forEach(el => {
                el.style.display = 'none';
            });
            
            console.log('Running in standalone mode - PWA styling applied');
        }
    }
    
    // Show installation prompt
    function showInstallPrompt() {
        if (installContainer && !isRunningStandalone()) {
            installContainer.classList.add('show');
            installContainer.classList.remove('d-none');
        }
    }
    
    // Initialize listeners
    function initListeners() {
        // Listen for beforeinstallprompt event
        window.addEventListener('beforeinstallprompt', (e) => {
            // Prevent the default browser install prompt
            e.preventDefault();
            
            // Store the event for later use
            deferredPrompt = e;
            
            // Show install button after a delay
            setTimeout(() => {
                showInstallPrompt();
            }, 3000);
            
            console.log('App is installable - showing install button');
        });
        
        // Listen for app installed event
        window.addEventListener('appinstalled', () => {
            // Hide install prompt
            if (installContainer) {
                installContainer.classList.remove('show');
            }
            
            // Clear the deferredPrompt
            deferredPrompt = null;
            
            // Show success message
            showInstallSuccess();
            
            console.log('App was installed successfully');
            
            // Send analytics
            if (typeof gtag === 'function') {
                gtag('event', 'pwa_install', {
                    'event_category': 'engagement',
                    'event_label': 'PWA Installed'
                });
            }
        });
        
        // Add click handler to install button
        if (installButton) {
            installButton.addEventListener('click', async () => {
                if (!deferredPrompt) {
                    return;
                }
                
                // Show the install prompt
                deferredPrompt.prompt();
                
                // Wait for the user to respond to the prompt
                const { outcome } = await deferredPrompt.userChoice;
                
                // Log the outcome
                console.log(`User ${outcome} the installation`);
                
                // Send analytics
                if (typeof gtag === 'function') {
                    gtag('event', 'install_prompt_response', {
                        'event_category': 'engagement',
                        'event_label': outcome
                    });
                }
                
                // Clear the deferredPrompt variable
                deferredPrompt = null;
                
                // Hide the install button
                installContainer.classList.remove('show');
            });
        }
        
        // Handle visibility change for better UX
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') {
                // Refresh dynamic content if needed when app comes to foreground
                refreshDynamicContent();
            }
        });
    }
    
    // Show installation success toast
    function showInstallSuccess() {
        // Create toast container if not exists
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            document.body.appendChild(toastContainer);
        }
        
        // Create toast element
        const toastEl = document.createElement('div');
        toastEl.className = 'toast';
        toastEl.setAttribute('role', 'alert');
        toastEl.setAttribute('aria-live', 'assertive');
        toastEl.setAttribute('aria-atomic', 'true');
        
        // Toast content
        toastEl.innerHTML = `
            <div class="toast-header">
                <i class="fas fa-check-circle me-2 text-success"></i>
                <strong class="me-auto">ReMind Installed</strong>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ReMind has been successfully installed on your device!
            </div>
        `;
        
        // Add toast to container
        toastContainer.appendChild(toastEl);
        
        // Initialize and show the toast
        const toast = new bootstrap.Toast(toastEl, { autohide: true, delay: 5000 });
        toast.show();
        
        // Remove toast from DOM after it's hidden
        toastEl.addEventListener('hidden.bs.toast', () => {
            toastEl.remove();
        });
    }
    
    // Refresh dynamic content (implemented separately for each page if needed)
    function refreshDynamicContent() {
        // Refresh time displays
        const timeElements = document.querySelectorAll('[data-dynamic="time"]');
        if (timeElements.length > 0) {
            const now = new Date();
            timeElements.forEach(el => {
                el.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            });
        }
        
        // Call page-specific refresh functions if they exist
        if (typeof refreshPageData === 'function') {
            refreshPageData();
        }
    }
    
    // Check online status and update UI accordingly
    function setupOfflineDetection() {
        function updateOnlineStatus() {
            const condition = navigator.onLine ? 'online' : 'offline';
            
            // Add/remove offline class from body
            document.body.classList.toggle('offline', !navigator.onLine);
            
            // Show appropriate alert
            if (!navigator.onLine) {
                showOfflineAlert();
            } else {
                hideOfflineAlert();
            }
            
            console.log(`App is now ${condition}`);
        }
        
        window.addEventListener('online', updateOnlineStatus);
        window.addEventListener('offline', updateOnlineStatus);
        
        // Initial check
        updateOnlineStatus();
    }
    
    // Show offline alert
    function showOfflineAlert() {
        // Don't show again if already exists
        if (document.querySelector('.offline-alert')) {
            return;
        }
        
        const alertEl = document.createElement('div');
        alertEl.className = 'offline-alert alert alert-warning alert-dismissible fade show';
        alertEl.setAttribute('role', 'alert');
        alertEl.innerHTML = `
            <i class="fas fa-wifi-slash me-2"></i>
            <strong>You're offline.</strong> Some features may be limited.
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Prepend to container
        const container = document.querySelector('.container');
        if (container) {
            container.prepend(alertEl);
        }
    }
    
    // Hide offline alert
    function hideOfflineAlert() {
        const alertEl = document.querySelector('.offline-alert');
        if (alertEl) {
            const bsAlert = new bootstrap.Alert(alertEl);
            bsAlert.close();
        }
    }
    
    // Initialize all functionality
    function init() {
        applyPWAStyling();
        initListeners();
        setupOfflineDetection();
        
        // Set periodic refresh for dynamic content
        setInterval(refreshDynamicContent, 60000); // Every minute
        
        console.log('PWA enhancement script initialized');
    }
    
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})(); 