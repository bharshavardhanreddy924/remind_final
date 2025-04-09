/**
 * Memory Care PWA Service Worker
 * Provides advanced caching, offline functionality, and background features
 * Optimized for Android and mobile devices
 */

const CACHE_NAME = 'remind-pwa-v1';
const DYNAMIC_CACHE = 'remind-dynamic-v1';

// Assets to cache on install
const STATIC_ASSETS = [
  '/',
  '/static/css/main.css',
  '/static/js/main.js',
  '/static/js/pwa-install.js',
  '/static/manifest.json',
  '/static/icons/favicon.ico',
  '/static/icons/icon-192x192.png',
  '/static/icons/icon-512x512.png',
  '/static/icons/maskable_icon.png',
  '/offline'
];

// Dynamic content to cache as user interacts with the app
const DYNAMIC_CACHE_NAME = 'remind-dynamic-v4';

// API routes that should be network-first (always try network before cache)
const API_ROUTES = [
  '/api/',
  '/user_dashboard'
];

// Special assets to prefetch that are critical for the Android experience
const ANDROID_CRITICAL_ASSETS = [
  '/login',
  '/splash',
  '/static/images/icons/icon-192x192.png',
  '/static/images/icons/icon-512x512.png',
  '/static/js/pwa-install.js',
  '/offline'
];

// Install event
self.addEventListener('install', event => {
  console.log('[Service Worker] Installing Service Worker...', event);
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('[Service Worker] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        console.log('[Service Worker] Successfully installed');
        return self.skipWaiting();
      })
  );
});

// Activate event
self.addEventListener('activate', event => {
  console.log('[Service Worker] Activating Service Worker...', event);
  
  // Clean up old caches
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME && cacheName !== DYNAMIC_CACHE) {
            console.log('[Service Worker] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('[Service Worker] Now ready to handle fetches!');
      return self.clients.claim();
    })
  );
});

// Helper: Is this a navigation request?
function isNavigationRequest(request) {
  return (
    request.mode === 'navigate' || 
    (request.method === 'GET' && 
     request.headers.get('accept') && 
     request.headers.get('accept').includes('text/html'))
  );
}

// Helper: Is this an API request?
function isApiRequest(url) {
  return API_ROUTES.some(route => url.pathname.startsWith(route));
}

// Helper: Is this a static asset?
function isStaticAsset(url) {
  return (
    url.pathname.startsWith('/static/') || 
    url.pathname.startsWith('/assets/') ||
    url.origin.includes('cdnjs.cloudflare.com') ||
    url.origin.includes('cdn.jsdelivr.net') ||
    url.origin.includes('fonts.googleapis.com') ||
    url.origin.includes('fonts.gstatic.com')
  );
}

// Helper: Should this request be cached?
function shouldCache(url) {
  const uncacheablePatterns = [
    '/api/analytics',
    '/api/log',
    '/api/push-subscription',
    'sockjs-node',
    'chrome-extension',
    'browser-sync'
  ];
  
  return !uncacheablePatterns.some(pattern => url.includes(pattern));
}

// Helper: Is this an Android device?
function isAndroidDevice() {
  return /Android/i.test(self.navigator.userAgent);
}

// Fetch event
self.addEventListener('fetch', event => {
  const request = event.request;
  const url = new URL(request.url);
  
  // Skip cross-origin requests
  if (url.origin !== location.origin) {
    return;
  }

  // API calls - Network first, then offline fallback
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirstStrategy(request));
    return;
  }
  
  // HTML pages - Network first with cache fallback
  if (request.headers.get('Accept').includes('text/html')) {
    event.respondWith(
      fetch(request)
        .then(response => {
          // Cache the latest version
          let clonedResponse = response.clone();
          caches.open(DYNAMIC_CACHE).then(cache => {
            cache.put(request, clonedResponse);
          });
          return response;
        })
        .catch(() => {
          return caches.match(request)
            .then(cachedResponse => {
              if (cachedResponse) {
                return cachedResponse;
              }
              // If no cached version, return the offline page
              return caches.match('/offline');
            });
        })
    );
    return;
  }

  // For other assets, use cache-first strategy
  event.respondWith(cacheFirstStrategy(request));
});

// Cache-first strategy (for static assets)
function cacheFirstStrategy(request) {
  return caches.match(request)
    .then(cachedResponse => {
      if (cachedResponse) {
        return cachedResponse;
      }

      return fetch(request)
        .then(networkResponse => {
          // Cache the fetched resource
          let responseToCache = networkResponse.clone();
          caches.open(DYNAMIC_CACHE).then(cache => {
            cache.put(request, responseToCache);
          });
          return networkResponse;
        })
        .catch(error => {
          console.error('[Service Worker] Fetch failed:', error);
          // Return default offline content for images or other resources
          if (request.url.match(/\.(jpe?g|png|gif|svg|webp)$/)) {
            return caches.match('/static/images/offline-image.png');
          }
          return new Response('Network request failed');
        });
    });
}

// Network-first strategy (for API calls and dynamic content)
function networkFirstStrategy(request) {
  return fetch(request)
    .then(response => {
      // Cache the latest version
      let clonedResponse = response.clone();
      caches.open(DYNAMIC_CACHE).then(cache => {
        cache.put(request, clonedResponse);
      });
      return response;
    })
    .catch(() => {
      return caches.match(request)
        .then(cachedResponse => {
          if (cachedResponse) {
            return cachedResponse;
          }
          return new Response(JSON.stringify({ 
            error: 'Network connection lost' 
          }), {
            headers: { 'Content-Type': 'application/json' }
          });
        });
    });
}

// Background sync for storing tasks offline and syncing when online
self.addEventListener('sync', event => {
  if (event.tag === 'sync-new-tasks') {
    event.waitUntil(syncNewTasks());
  } else if (event.tag === 'sync-new-medications') {
    event.waitUntil(syncNewMedications());
  }
});

// Helper: Sync tasks created while offline
function syncNewTasks() {
  return caches.open(DYNAMIC_CACHE_NAME)
    .then(cache => {
      return cache.match('offlineTasks')
        .then(response => {
          if (!response) {
            return;
          }
          
          return response.json()
            .then(offlineTasks => {
              return Promise.all(
                offlineTasks.map(task => {
                  return fetch('/api/tasks', {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(task)
                  })
                  .catch(err => console.error('Error posting task:', err));
                })
              )
              .then(() => {
                // Clear the offline tasks
                return cache.delete('offlineTasks');
              });
            });
        });
    });
}

// Helper: Sync medications created while offline
function syncNewMedications() {
  return caches.open(DYNAMIC_CACHE_NAME)
    .then(cache => {
      return cache.match('offlineMedications')
        .then(response => {
          if (!response) {
            return;
          }
          
          return response.json()
            .then(offlineMeds => {
              return Promise.all(
                offlineMeds.map(med => {
                  return fetch('/api/medications', {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(med)
                  })
                  .catch(err => console.error('Error posting medication:', err));
                })
              )
              .then(() => {
                // Clear the offline medications
                return cache.delete('offlineMedications');
              });
            });
        });
    });
}

// Handle push notifications
self.addEventListener('push', event => {
  const data = event.data.json();
  const title = data.title || 'ReMind';
  const options = {
    body: data.body || 'New notification from ReMind',
    icon: data.icon || '/static/images/icons/icon-192x192.png',
    badge: data.badge || '/static/images/icons/icon-72x72.png',
    data: data.data || {},
    vibrate: data.vibrate || [100, 50, 100],
    actions: data.actions || [
      {
        action: 'view',
        title: 'View'
      }
    ]
  };
  
  // Add Android-specific options
  if (isAndroidDevice()) {
    options.renotify = true;
    options.tag = 'remind-notification';
    options.actions = [
      {
        action: 'view',
        title: 'Open',
        icon: '/static/images/icons/icon-96x96.png'
      },
      {
        action: 'dismiss',
        title: 'Dismiss',
        icon: '/static/images/icons/icon-96x96.png'
      }
    ];
  }

  event.waitUntil(
    self.registration.showNotification(title, options)
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', event => {
  event.notification.close();
  
  let targetUrl = '/';
  
  if (event.action === 'view' || event.action === '') {
    // Handle the data payload to determine where to navigate
    if (event.notification.data && event.notification.data.url) {
      targetUrl = event.notification.data.url;
    } else if (event.notification.data && event.notification.data.type) {
      // Handle different notification types
      switch (event.notification.data.type) {
        case 'task':
          targetUrl = '/tasks';
          break;
        case 'medication':
          targetUrl = '/medications';
          break;
        case 'memory':
          targetUrl = '/memory_training';
          break;
        case 'emergency':
          targetUrl = '/patient_location/' + event.notification.data.patientId;
          break;
        default:
          targetUrl = '/user_dashboard';
      }
    }
  }
  
  event.waitUntil(
    clients.matchAll({
      type: 'window'
    })
    .then(function(clientList) {
      // If we have a window client that's already open, focus it
      for (let i = 0; i < clientList.length; i++) {
        const client = clientList[i];
        if (client.url === targetUrl && 'focus' in client) {
          return client.focus();
        }
      }
      
      // Otherwise open a new window
      if (clients.openWindow) {
        return clients.openWindow(targetUrl);
      }
    })
  );
});

// Listen for messages from clients
self.addEventListener('message', event => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

console.log('[Service Worker] Script loaded and ready');