/**
 * Notification Manager
 * Handles push notifications, reminders, and alerts for ReMind application
 */

const NotificationManager = (function() {
    'use strict';
    
    // Private variables
    let _initialized = false;
    let _permissionGranted = false;
    const _pendingNotifications = [];
    
    // Check if notifications are supported in this browser
    const _isSupported = () => {
        return 'Notification' in window && 'serviceWorker' in navigator && 'PushManager' in window;
    };
    
    // Convert an icon name to a full path
    const _getIconPath = (icon) => {
        const defaultIcon = '/static/images/icons/icon-192x192.png';
        
        if (!icon) return defaultIcon;
        
        // If it's a full URL or path, return as is
        if (icon.startsWith('http') || icon.startsWith('/')) {
            return icon;
        }
        
        // Otherwise, prepend the path
        return `/static/images/icons/${icon}`;
    };
    
    // Request notification permission
    const _requestPermission = async () => {
        if (!_isSupported()) {
            console.warn('Notifications are not supported in this browser');
            return false;
        }
        
        try {
            const permission = await Notification.requestPermission();
            _permissionGranted = permission === 'granted';
            
            if (_permissionGranted) {
                console.log('Notification permission granted');
                _processPendingNotifications();
            } else {
                console.log('Notification permission denied');
            }
            
            return _permissionGranted;
        } catch (error) {
            console.error('Error requesting notification permission:', error);
            return false;
        }
    };
    
    // Process any notifications that were queued while waiting for permission
    const _processPendingNotifications = () => {
        if (!_permissionGranted || _pendingNotifications.length === 0) return;
        
        _pendingNotifications.forEach(notification => {
            showNotification(
                notification.title,
                notification.options
            );
        });
        
        // Clear the pending notifications
        _pendingNotifications.length = 0;
    };
    
    // Subscribe to push notifications
    const _subscribeToPushNotifications = async () => {
        if (!_permissionGranted) return null;
        
        try {
            const registration = await navigator.serviceWorker.ready;
            
            // Subscribe the user to push notifications
            let subscription = await registration.pushManager.getSubscription();
            
            if (!subscription) {
                const vapidPublicKey = document.querySelector('meta[name="vapid-public-key"]')?.content;
                
                if (!vapidPublicKey) {
                    console.warn('VAPID public key not found, cannot subscribe to push notifications');
                    return null;
                }
                
                // Convert the VAPID key to the format expected by the push manager
                const convertedKey = _urlBase64ToUint8Array(vapidPublicKey);
                
                subscription = await registration.pushManager.subscribe({
                    userVisibleOnly: true,
                    applicationServerKey: convertedKey
                });
                
                // Send the subscription to the server
                await _sendSubscriptionToServer(subscription);
                
                console.log('User subscribed to push notifications');
            }
            
            return subscription;
        } catch (error) {
            console.error('Error subscribing to push notifications:', error);
            return null;
        }
    };
    
    // Send the subscription to the server
    const _sendSubscriptionToServer = async (subscription) => {
        try {
            const response = await fetch('/api/push-subscription', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(subscription)
            });
            
            if (!response.ok) {
                throw new Error('Failed to send subscription to server');
            }
            
            console.log('Push subscription sent to server');
            return true;
        } catch (error) {
            console.error('Error sending subscription to server:', error);
            return false;
        }
    };
    
    // Helper function to convert base64 to Uint8Array for VAPID key
    const _urlBase64ToUint8Array = (base64String) => {
        const padding = '='.repeat((4 - base64String.length % 4) % 4);
        const base64 = (base64String + padding)
            .replace(/-/g, '+')
            .replace(/_/g, '/');
        
        const rawData = window.atob(base64);
        const outputArray = new Uint8Array(rawData.length);
        
        for (let i = 0; i < rawData.length; ++i) {
            outputArray[i] = rawData.charCodeAt(i);
        }
        
        return outputArray;
    };
    
    // Schedule a local notification
    const _scheduleLocalNotification = (title, options, delay) => {
        if (!_isSupported() || !_permissionGranted) return false;
        
        setTimeout(() => {
            showNotification(title, options);
        }, delay);
        
        return true;
    };
    
    // Public methods
    
    // Initialize the notification manager
    const initialize = async () => {
        if (_initialized) return _permissionGranted;
        
        if (!_isSupported()) {
            console.warn('Notifications are not supported in this browser');
            return false;
        }
        
        // Check if permission is already granted
        _permissionGranted = Notification.permission === 'granted';
        
        if (Notification.permission !== 'denied') {
            _permissionGranted = await _requestPermission();
        }
        
        // Subscribe to push notifications if permission is granted
        if (_permissionGranted) {
            await _subscribeToPushNotifications();
        }
        
        _initialized = true;
        return _permissionGranted;
    };
    
    // Show a notification immediately
    const showNotification = async (title, options = {}) => {
        if (!_initialized) {
            await initialize();
        }
        
        if (!_permissionGranted) {
            // Queue the notification for later if permission hasn't been granted yet
            _pendingNotifications.push({ title, options });
            return false;
        }
        
        // Set default options
        const defaultOptions = {
            icon: _getIconPath('icon-192x192.png'),
            badge: _getIconPath('notification-badge.png'),
            vibrate: [100, 50, 100],
            requireInteraction: false,
            silent: false
        };
        
        // Merge with provided options
        const notificationOptions = { ...defaultOptions, ...options };
        
        // Convert icon paths if needed
        if (notificationOptions.icon) {
            notificationOptions.icon = _getIconPath(notificationOptions.icon);
        }
        if (notificationOptions.badge) {
            notificationOptions.badge = _getIconPath(notificationOptions.badge);
        }
        
        try {
            // Check if we have an active service worker
            const registration = await navigator.serviceWorker.ready;
            
            // Show the notification through the service worker
            await registration.showNotification(title, notificationOptions);
            return true;
        } catch (error) {
            console.error('Error showing notification:', error);
            
            // Fallback to regular Notification API if service worker fails
            try {
                new Notification(title, notificationOptions);
                return true;
            } catch (fallbackError) {
                console.error('Fallback notification also failed:', fallbackError);
                return false;
            }
        }
    };
    
    // Schedule a notification for later
    const scheduleNotification = (title, options = {}, delay = 60000) => {
        if (!_initialized) {
            // Initialize and then try again
            initialize().then(() => {
                return _scheduleLocalNotification(title, options, delay);
            });
            return false;
        }
        
        return _scheduleLocalNotification(title, options, delay);
    };
    
    // Schedule a task reminder
    const scheduleTaskReminder = (taskText, dueTime, taskId) => {
        // Calculate delay until the due time
        const now = new Date();
        const dueDate = new Date(dueTime);
        const delay = Math.max(0, dueDate.getTime() - now.getTime());
        
        // Schedule notification 15 minutes before due time
        const reminderDelay = Math.max(0, delay - (15 * 60 * 1000));
        
        return scheduleNotification(
            'Task Reminder',
            {
                body: `Reminder: ${taskText}`,
                icon: _getIconPath('task-icon.png'),
                data: {
                    url: '/tasks',
                    taskId: taskId
                },
                tag: `task-reminder-${taskId}`
            },
            reminderDelay
        );
    };
    
    // Schedule a medication reminder
    const scheduleMedicationReminder = (medicationName, dueTime, medicationId) => {
        // Calculate delay until the due time
        const now = new Date();
        const dueDate = new Date(dueTime);
        const delay = Math.max(0, dueDate.getTime() - now.getTime());
        
        // Schedule notification at due time
        return scheduleNotification(
            'Medication Reminder',
            {
                body: `Time to take your ${medicationName}`,
                icon: _getIconPath('medication-icon.png'),
                requireInteraction: true,
                data: {
                    url: '/medications',
                    medicationId: medicationId
                },
                tag: `medication-reminder-${medicationId}`
            },
            delay
        );
    };
    
    // Show an SOS alert notification
    const showSOSAlert = (patientName, location = "Unknown location") => {
        return showNotification(
            'Emergency Alert',
            {
                body: `${patientName} has triggered an emergency alert at ${location}`,
                icon: _getIconPath('sos-icon.png'),
                requireInteraction: true,
                vibrate: [100, 50, 100, 50, 100, 50, 200, 50, 200],
                actions: [
                    {
                        action: 'view',
                        title: 'View Details'
                    },
                    {
                        action: 'contact',
                        title: 'Contact Emergency'
                    }
                ],
                data: {
                    url: '/sos-alerts',
                    patientName: patientName,
                    location: location
                },
                tag: 'sos-alert'
            }
        );
    };
    
    // Show a memory training reminder
    const showMemoryTrainingReminder = () => {
        return showNotification(
            'Memory Training Reminder',
            {
                body: 'It\'s time for your daily memory training exercise',
                icon: _getIconPath('brain-icon.png'),
                data: {
                    url: '/memory_training'
                },
                tag: 'memory-training-reminder'
            }
        );
    };
    
    // Return public API
    return {
        initialize,
        showNotification,
        scheduleNotification,
        scheduleTaskReminder,
        scheduleMedicationReminder,
        showSOSAlert,
        showMemoryTrainingReminder,
        isSupported: _isSupported
    };
})();

// Automatically initialize the notification manager when the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Initialize notification manager
    NotificationManager.initialize().then(permissionGranted => {
        if (permissionGranted) {
            console.log('Notification Manager initialized successfully');
            
            // Schedule daily memory training reminder if on the main dashboard
            if (window.location.pathname === '/user_dashboard' || window.location.pathname === '/') {
                // Schedule reminder for 10AM if it's before 10AM, or tomorrow if it's after
                const now = new Date();
                const reminderTime = new Date();
                reminderTime.setHours(10, 0, 0, 0);
                
                if (now > reminderTime) {
                    reminderTime.setDate(reminderTime.getDate() + 1);
                }
                
                const delay = reminderTime.getTime() - now.getTime();
                
                NotificationManager.scheduleNotification(
                    'Memory Training',
                    {
                        body: 'It\'s time for your daily memory training session',
                        icon: 'brain-icon.png',
                        data: {
                            url: '/memory_training'
                        }
                    },
                    delay
                );
            }
        } else {
            console.warn('Notification permission not granted');
        }
    });
    
    // Schedule notifications for existing tasks/medications
    if (window.tasks) {
        window.tasks.forEach(task => {
            if (!task.completed && task.due_date) {
                NotificationManager.scheduleTaskReminder(
                    task.text, 
                    task.due_date,
                    task.id
                );
            }
        });
    }
    
    if (window.medications) {
        window.medications.forEach(medication => {
            if (medication.time) {
                NotificationManager.scheduleMedicationReminder(
                    medication.name,
                    medication.time,
                    medication.id
                );
            }
        });
    }
});

// Make the NotificationManager available globally
window.NotificationManager = NotificationManager; 