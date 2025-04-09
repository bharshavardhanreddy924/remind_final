class PermissionsManager {
    constructor() {
        this.permissions = {
            geolocation: false,
            notifications: false,
            microphone: false
        };
        this.initialize();
    }

    async initialize() {
        // Check initial permission states
        await this.checkPermissions();
        
        // Listen for permission changes
        if ('permissions' in navigator) {
            navigator.permissions.query({ name: 'geolocation' })
                .then(permissionStatus => {
                    permissionStatus.onchange = () => this.handlePermissionChange('geolocation', permissionStatus.state);
                });
            
            navigator.permissions.query({ name: 'notifications' })
                .then(permissionStatus => {
                    permissionStatus.onchange = () => this.handlePermissionChange('notifications', permissionStatus.state);
                });
            
            navigator.permissions.query({ name: 'microphone' })
                .then(permissionStatus => {
                    permissionStatus.onchange = () => this.handlePermissionChange('microphone', permissionStatus.state);
                });
        }
    }

    async checkPermissions() {
        // Check geolocation permission
        if ('geolocation' in navigator) {
            try {
                const result = await navigator.permissions.query({ name: 'geolocation' });
                this.permissions.geolocation = result.state === 'granted';
            } catch (error) {
                console.error('Error checking geolocation permission:', error);
            }
        }

        // Check notification permission
        if ('Notification' in window) {
            this.permissions.notifications = Notification.permission === 'granted';
        }

        // Check microphone permission
        if ('mediaDevices' in navigator) {
            try {
                const result = await navigator.permissions.query({ name: 'microphone' });
                this.permissions.microphone = result.state === 'granted';
            } catch (error) {
                console.error('Error checking microphone permission:', error);
            }
        }

        this.updateUI();
    }

    handlePermissionChange(permission, state) {
        this.permissions[permission] = state === 'granted';
        this.updateUI();
    }

    async requestPermission(permission) {
        switch (permission) {
            case 'geolocation':
                return new Promise((resolve, reject) => {
                    navigator.geolocation.getCurrentPosition(
                        () => {
                            this.permissions.geolocation = true;
                            this.updateUI();
                            resolve(true);
                        },
                        (error) => {
                            console.error('Geolocation error:', error);
                            reject(error);
                        }
                    );
                });

            case 'notifications':
                try {
                    const result = await Notification.requestPermission();
                    this.permissions.notifications = result === 'granted';
                    this.updateUI();
                    return this.permissions.notifications;
                } catch (error) {
                    console.error('Notification permission error:', error);
                    return false;
                }

            case 'microphone':
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    stream.getTracks().forEach(track => track.stop());
                    this.permissions.microphone = true;
                    this.updateUI();
                    return true;
                } catch (error) {
                    console.error('Microphone permission error:', error);
                    return false;
                }
        }
    }

    updateUI() {
        // Update permission indicators in the UI
        const permissionIndicators = document.querySelectorAll('.permission-indicator');
        permissionIndicators.forEach(indicator => {
            const permission = indicator.dataset.permission;
            if (this.permissions[permission]) {
                indicator.classList.add('granted');
                indicator.classList.remove('denied');
            } else {
                indicator.classList.add('denied');
                indicator.classList.remove('granted');
            }
        });
    }

    showPermissionModal() {
        const modal = document.getElementById('permissionModal');
        if (modal) {
            const modalInstance = new bootstrap.Modal(modal);
            modalInstance.show();
        }
    }

    async requestAllPermissions() {
        const results = {
            geolocation: await this.requestPermission('geolocation'),
            notifications: await this.requestPermission('notifications'),
            microphone: await this.requestPermission('microphone')
        };

        return results;
    }
}

// Initialize permissions manager
const permissionsManager = new PermissionsManager();

// Export for use in other files
window.permissionsManager = permissionsManager; 