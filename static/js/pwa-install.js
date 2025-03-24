// PWA Installation Script
let deferredPrompt;
const installContainer = document.getElementById('install-container');
const installButton = document.getElementById('install-button');
let installBannerShown = false;

// Check if the app is running in standalone mode (installed PWA)
function isRunningStandalone() {
    return (window.matchMedia('(display-mode: standalone)').matches) || 
           (window.matchMedia('(display-mode: fullscreen)').matches) || 
           (window.navigator.standalone === true) || // For iOS Safari
           (window.location.search.includes('standalone=true'));
}

// Apply standalone class to HTML if in standalone mode
function checkStandalone() {
    if (isRunningStandalone()) {
        document.documentElement.classList.add('pwa-standalone');
        document.body.classList.add('pwa-standalone');
        
        // For internal links, add the standalone parameter
        document.querySelectorAll('a[href^="/"]').forEach(link => {
            const url = new URL(link.href, window.location.origin);
            if (!url.searchParams.has('standalone')) {
                url.searchParams.set('standalone', 'true');
                link.href = url.toString();
            }
        });
    }
}

// Handle the beforeinstallprompt event
window.addEventListener('beforeinstallprompt', (e) => {
    // Prevent the mini-infobar from appearing on mobile
    e.preventDefault();
    // Store the event so it can be triggered later
    deferredPrompt = e;
    // Show the install button
    if (installContainer && !isRunningStandalone() && !installBannerShown) {
        installContainer.classList.remove('d-none');
        installContainer.classList.add('d-block');
    }
});

// Handle the install button click
if (installButton) {
    installButton.addEventListener('click', async () => {
        if (!deferredPrompt) {
            // The deferred prompt isn't available
            // This could happen if the app is already installed or cannot be installed
            showInstallInstructions();
            return;
        }
        
        // Hide the install button
        installContainer.classList.add('d-none');
        installBannerShown = true;
        
        // Show the install prompt
        deferredPrompt.prompt();
        
        // Wait for the user to respond to the prompt
        const { outcome } = await deferredPrompt.userChoice;
        console.log(`User ${outcome} the installation`);
        
        // We've used the prompt, and can't use it again, throw it away
        deferredPrompt = null;
        
        // If rejected, show a custom banner after delay
        if (outcome === 'dismissed') {
            localStorage.setItem('installPromptDismissed', Date.now());
            setTimeout(showCustomInstallBanner, 300000); // Show after 5 minutes
        }
    });
}

// Show custom install banner for browsers that don't support beforeinstallprompt
function showCustomInstallBanner() {
    if (isRunningStandalone() || installBannerShown) return;
    
    // Check if we've previously shown install banner recently
    const lastDismissed = localStorage.getItem('installPromptDismissed');
    if (lastDismissed && (Date.now() - lastDismissed < 86400000)) { // 24 hours
        return;
    }
    
    // Create and show the banner
    const banner = document.createElement('div');
    banner.className = 'add-to-home';
    banner.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
            <div>Install this app for a better experience!</div>
            <button class="btn btn-sm btn-light" id="close-banner">×</button>
        </div>
    `;
    document.body.prepend(banner);
    
    // Show the banner with animation
    setTimeout(() => {
        banner.classList.add('show');
        installBannerShown = true;
    }, 500);
    
    // Handle close button
    document.getElementById('close-banner').addEventListener('click', () => {
        banner.classList.remove('show');
        localStorage.setItem('installPromptDismissed', Date.now());
    });
}

// For browsers that don't support the installation API
function showInstallInstructions() {
    // Detect iOS Safari
    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
    const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
    
    let message = '';
    
    if (isIOS && isSafari) {
        message = 'To install this app on your iOS device: tap the Share button, then "Add to Home Screen"';
    } else if (navigator.userAgent.indexOf('Firefox') !== -1) {
        message = 'To install this app in Firefox: tap the menu button (three dots), then "Install"';
    } else {
        message = 'To install this app: open in Chrome, tap the menu button, then "Install App"';
    }
    
    alert(message);
}

// Detect when the PWA has been successfully installed
window.addEventListener('appinstalled', (evt) => {
    console.log('MemoryCare app was installed');
    installContainer.classList.add('d-none');
    
    // You could add analytics tracking here
    // gtag('event', 'pwa_install_success');
    
    // Show a success message
    const toast = document.createElement('div');
    toast.className = 'position-fixed bottom-0 end-0 p-3';
    toast.style.zIndex = '5';
    toast.innerHTML = `
        <div class="toast show" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header">
                <strong class="me-auto">MemoryCare Installed</strong>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                App successfully installed! You can now access it from your home screen.
            </div>
        </div>
    `;
    document.body.appendChild(toast);
    
    // Remove toast after 5 seconds
    setTimeout(() => {
        document.body.removeChild(toast);
    }, 5000);
});

// Check for standalone mode on page load
document.addEventListener('DOMContentLoaded', () => {
    checkStandalone();
    
    // Show install banner based on user interactions
    let userInteracted = false;
    const registerUserInteraction = () => {
        userInteracted = true;
        // After user has interacted, we can show the install banner
        // but add delay to not be intrusive
        if (!isRunningStandalone() && !installBannerShown && !deferredPrompt) {
            setTimeout(showCustomInstallBanner, 10000); // 10 seconds after interaction
        }
        
        // Remove event listeners once triggered
        document.removeEventListener('click', registerUserInteraction);
        document.removeEventListener('scroll', registerUserInteraction);
    };
    
    document.addEventListener('click', registerUserInteraction);
    document.addEventListener('scroll', registerUserInteraction);
    
    // Check for parameters in URL
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('standalone') && urlParams.get('standalone') === 'true') {
        document.documentElement.classList.add('pwa-standalone');
        document.body.classList.add('pwa-standalone');
    }
}); 