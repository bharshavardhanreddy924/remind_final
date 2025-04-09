/**
 * Progressive Web App Installation & Enhancement Script
 * Handles PWA installation, detects standalone mode, and improves user experience
 */

(function() {
    'use strict';
    
    // Variables
    let deferredPrompt;
    const installContainer = document.getElementById('pwa-install-container');
    const installButton = document.getElementById('pwa-install-button');
    const androidInstallBanner = document.getElementById('android-install-banner');
    const androidInstallBtn = document.getElementById('android-install-btn');
    const androidInstallClose = document.getElementById('android-install-close');
    
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
        // Check if banners are available
        if (androidInstallBanner && isAndroid() && !isRunningStandalone()) {
            androidInstallBanner.classList.remove('d-none');
            androidInstallBanner.classList.add('show');
        } else if (installContainer && !isRunningStandalone()) {
            installContainer.style.display = 'flex';
            installContainer.classList.add('show');
        }
    }

    // Check if running on Android
    function isAndroid() {
        return /Android/i.test(navigator.userAgent);
    }

    // Check if running on iOS
    function isiOS() {
        return /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
    }

    // Create install banner popup
    function createInstallPopup() {
        // Only create if it doesn't exist
        if (document.getElementById('pwa-install-popup')) {
            return;
        }

        const isAndroidDevice = isAndroid();
        const isiOSDevice = isiOS();
        let installText = 'Install this app on your device for a better experience!';
        
        if (isAndroidDevice) {
            installText = 'Install ReMind app on your Android device';
        } else if (isiOSDevice) {
            installText = 'Add ReMind to your home screen';
        } else {
            installText = 'Install ReMind app for easier access';
        }
        
        const popup = document.createElement('div');
        popup.id = 'pwa-install-popup';
        popup.className = 'pwa-install-popup';
        popup.innerHTML = `
            <div class="pwa-install-content">
                <img src="/static/images/icons/icon-192x192.png" alt="ReMind App" class="pwa-install-logo">
                <div class="pwa-install-text">
                    <h4>ReMind App</h4>
                    <p>${installText}</p>
                </div>
                <div class="pwa-install-actions">
                    <button id="pwa-install-btn" class="pwa-install-button">Install</button>
                    <button id="pwa-install-later" class="pwa-install-later">Later</button>
                </div>
            </div>
        `;
        
        // Add styles for the popup
        const style = document.createElement('style');
        style.textContent = `
            .pwa-install-popup {
                position: fixed;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 1000;
                width: 90%;
                max-width: 400px;
                animation: slideUp 0.3s forwards;
                border-top: 4px solid #4361ee;
            }
            .pwa-install-content {
                display: flex;
                padding: 16px;
                align-items: center;
                flex-wrap: wrap;
            }
            .pwa-install-logo {
                width: 48px;
                height: 48px;
                margin-right: 16px;
            }
            .pwa-install-text {
                flex: 1;
                min-width: 150px;
            }
            .pwa-install-text h4 {
                margin: 0 0 4px 0;
                font-size: 16px;
            }
            .pwa-install-text p {
                margin: 0;
                font-size: 14px;
                color: #666;
            }
            .pwa-install-actions {
                display: flex;
                width: 100%;
                justify-content: flex-end;
                margin-top: 12px;
            }
            .pwa-install-button {
                background: #4361ee;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: 500;
                cursor: pointer;
            }
            .pwa-install-later {
                background: transparent;
                border: none;
                padding: 8px 16px;
                margin-right: 8px;
                cursor: pointer;
                color: #666;
            }
            @keyframes slideUp {
                from {
                    transform: translate(-50%, 100%);
                    opacity: 0;
                }
                to {
                    transform: translate(-50%, 0);
                    opacity: 1;
                }
            }
        `;
        
        document.head.appendChild(style);
        document.body.appendChild(popup);
        
        // Add event listeners
        document.getElementById('pwa-install-btn').addEventListener('click', () => {
            if (deferredPrompt) {
                triggerInstall();
            } else {
                window.location.href = '/pwa-install';
            }
        });
        
        document.getElementById('pwa-install-later').addEventListener('click', () => {
            popup.remove();
            // Store in localStorage to not show again for a while
            localStorage.setItem('pwaInstallDismissed', Date.now());
        });
    }
    
    // Trigger the install prompt
    function triggerInstall() {
        if (!deferredPrompt) {
            console.log('No installation prompt available');
            return;
        }
        
        // Show the install prompt
        deferredPrompt.prompt();
        
        // Wait for the user to respond to the prompt
        deferredPrompt.userChoice.then((choiceResult) => {
            if (choiceResult.outcome === 'accepted') {
                console.log('User accepted the install prompt');
                // Hide any install UI now that it's installed
                if (installContainer) installContainer.classList.remove('show');
                if (androidInstallBanner) androidInstallBanner.classList.add('d-none');
                const popup = document.getElementById('pwa-install-popup');
                if (popup) popup.remove();
            } else {
                console.log('User dismissed the install prompt');
            }
            
            // Clear the saved prompt since it can't be used twice
            deferredPrompt = null;
        });
    }
    
    // Initialize listeners
    function initListeners() {
        // Listen for beforeinstallprompt event
        window.addEventListener('beforeinstallprompt', (e) => {
            // Prevent the default browser install prompt
            e.preventDefault();
            
            // Store the event for later use
            deferredPrompt = e;
            
            console.log('App is installable - captured install prompt');
            
            // Show the install button since app is installable
            if (installContainer) {
                installContainer.style.display = 'flex';
            }
            
            // Show the appropriate install UI after a short delay
            // Don't show immediately to avoid interrupting user experience
            setTimeout(() => {
                // Check if user has recently dismissed
                const dismissedTime = localStorage.getItem('pwaInstallDismissed');
                const showAgain = !dismissedTime || (Date.now() - dismissedTime > 86400000); // 24 hours
                
                if (showAgain) {
                    showInstallPrompt();
                    createInstallPopup();
                }
            }, 3000);
        });
        
        // Listen for app installed event
        window.addEventListener('appinstalled', () => {
            // Hide install prompts
            if (installContainer) {
                installContainer.classList.remove('show');
            }
            
            if (androidInstallBanner) {
                androidInstallBanner.classList.add('d-none');
            }
            
            const popup = document.getElementById('pwa-install-popup');
            if (popup) {
                popup.remove();
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
        
        // Add click handler to install buttons
        if (installButton) {
            installButton.addEventListener('click', triggerInstall);
        }
        
        if (androidInstallBtn) {
            androidInstallBtn.addEventListener('click', triggerInstall);
        }
        
        if (androidInstallClose) {
            androidInstallClose.addEventListener('click', () => {
                if (androidInstallBanner) {
                    androidInstallBanner.classList.add('d-none');
                    androidInstallBanner.classList.remove('show');
                    // Store in localStorage to not show again for a while
                    localStorage.setItem('pwaInstallDismissed', Date.now());
                }
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
            <div class="toast-header bg-success text-white">
                <i class="fas fa-check-circle me-2"></i>
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
        
        // Check if already installed
        if (isRunningStandalone()) {
            console.log('App is already installed');
        } else {
            // If there's no install prompt yet but not installed,
            // we might be on iOS where we need custom instructions
            if (isiOS() && !localStorage.getItem('iosInstallShown')) {
                // Show iOS-specific instructions after a delay
                setTimeout(() => {
                    createInstallPopup();
                    localStorage.setItem('iosInstallShown', Date.now());
                }, 5000);
            }
        }
    }
    
    // Add event listeners when DOM is loaded
    document.addEventListener('DOMContentLoaded', () => {
        // Apply PWA styling if we're running as a standalone app
        applyPWAStyling();
        
        // Initialize installation button click handlers
        if (installButton) {
            installButton.addEventListener('click', triggerInstall);
        }
        
        if (androidInstallBtn) {
            androidInstallBtn.addEventListener('click', triggerInstall);
        }
        
        if (androidInstallClose) {
            androidInstallClose.addEventListener('click', () => {
                if (androidInstallBanner) {
                    androidInstallBanner.classList.add('d-none');
                    androidInstallBanner.classList.remove('show');
                    // Store in localStorage to not show again for a while
                    localStorage.setItem('pwaInstallDismissed', Date.now());
                }
            });
        }
        
        // Initialize all listeners
        initListeners();
    });
})(); 