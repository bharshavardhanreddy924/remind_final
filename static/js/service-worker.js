const CACHE_NAME = 'remind-v1';
const urlsToCache = [
    '/',
    '/static/css/style.css',
    '/static/js/pwa-install.js',
    '/static/js/location_tracker.js',
    '/static/js/voice_controller.js',
    '/static/js/notification_manager.js',
    '/static/images/icons/icon-192x192.png',
    '/static/images/icons/icon-512x512.png',
    '/offline.html',
    '/static/js/permissions_manager.js'
];

// Install Service Worker
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('Opened cache');
                return cache.addAll(urlsToCache);
            })
    );
    self.skipWaiting(); // Activate immediately
});

// Activate Service Worker
self.addEventListener('activate', event => {
    event.waitUntil(
        Promise.all([
            caches.keys().then(cacheNames => {
                return Promise.all(
                    cacheNames.map(cacheName => {
                        if (cacheName !== CACHE_NAME) {
                            return caches.delete(cacheName);
                        }
                    })
                );
            }),
            self.clients.claim() // Take control of all clients immediately
        ])
    );
});

// Handle Push Notifications
self.addEventListener('push', event => {
    let data = {};
    try {
        data = event.data.json();
    } catch (e) {
        data = { body: event.data.text() };
    }

    const options = {
        body: data.body || 'New notification',
        icon: '/static/images/icons/icon-192x192.png',
        badge: '/static/images/icons/badge-72x72.png',
        vibrate: [100, 50, 100],
        data: {
            url: data.url || '/',
            type: data.type || 'general'
        },
        actions: [
            {
                action: 'view',
                title: 'View Details'
            },
            {
                action: 'close',
                title: 'Close'
            }
        ],
        requireInteraction: true,
        renotify: true,
        tag: data.tag || 'default'
    };

    event.waitUntil(
        self.registration.showNotification(data.title || 'ReMind', options)
    );
});

// Handle Notification Click
self.addEventListener('notificationclick', event => {
    event.notification.close();

    if (event.action === 'view') {
        event.waitUntil(
            clients.openWindow(event.notification.data.url)
        );
    }
});

// Handle Fetch Events
self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                // Return cached response if found
                if (response) {
                    return response;
                }

                // Clone the request because it can only be used once
                const fetchRequest = event.request.clone();

                return fetch(fetchRequest)
                    .then(response => {
                        // Check if we received a valid response
                        if (!response || response.status !== 200 || response.type !== 'basic') {
                            return response;
                        }

                        // Clone the response because it can only be used once
                        const responseToCache = response.clone();

                        caches.open(CACHE_NAME)
                            .then(cache => {
                                cache.put(event.request, responseToCache);
                            });

                        return response;
                    })
                    .catch(() => {
                        // Return offline page for navigation requests
                        if (event.request.mode === 'navigate') {
                            return caches.match('/offline.html');
                        }
                    });
            })
    );
});

// Handle Background Sync
self.addEventListener('sync', event => {
    if (event.tag === 'sync-location') {
        event.waitUntil(syncLocation());
    }
});

// Handle Permission Requests
self.addEventListener('permissionrequest', event => {
    if (event.permission === 'geolocation' || event.permission === 'notifications' || event.permission === 'microphone') {
        event.waitUntil(
            clients.matchAll().then(clientList => {
                for (const client of clientList) {
                    client.postMessage({
                        type: 'PERMISSION_REQUEST',
                        permission: event.permission
                    });
                }
            })
        );
    }
}); 