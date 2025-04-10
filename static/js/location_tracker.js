// Location tracking functionality
class LocationTracker {
    constructor() {
        this.lastLocation = {
            latitude: 12.9637,
            longitude: 77.5060,
            accuracy: 0,
            source: 'fixed',
            timestamp: new Date().toISOString(),
            address: 'Dr. Ambedkar Institute Of Technology, Bengaluru, Karnataka, India'
        };
        this.locationLabel = document.getElementById('location-label');
        this.initialize();
    }

    async initialize() {
        try {
            // Update location label with fixed location
            if (this.locationLabel) {
                this.locationLabel.textContent = '📍 Dr. Ambedkar Institute Of Technology';
            }
            
            // Initial update
            await this.updateLocationToServer();
        } catch (error) {
            console.error('Error initializing location tracker:', error);
            this.showError('Failed to initialize location tracking');
        }
    }

    async updateLocationToServer() {
        try {
            const response = await fetch('/api/update_location', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.lastLocation)
            });

            if (!response.ok) {
                throw new Error('Failed to update location on server');
            }
        } catch (error) {
            console.error('Error updating location to server:', error);
        }
    }

    showError(message) {
        if (window.notifications) {
            window.notifications.show('Location Error', message, 'error');
        }
        if (this.locationLabel) {
            this.locationLabel.textContent = '📍 Location unavailable';
        }
    }
}

// Initialize location tracker when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.locationTracker = new LocationTracker();
}); 