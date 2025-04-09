/**
 * Memory Care PWA Service Worker
 * Provides advanced caching, offline functionality, and background features
 * Optimized for Android and mobile devices
 */

const CACHE_NAME = 'remind-v4';

// Assets to cache on install
const STATIC_CACHE_URLS = [
  '/',
  '/splash',
  '/login',
  '/register',
  '/offline',
  '/static/css/style.css',
  '/static/js/pwa-install.js',
  '/static/js/notification_manager.js', 
  '/static/js/voice_controller.js',
  '/static/js/permissions_manager.js',
  '/static/manifest.json',
  '/static/images/icons/icon-192x192.png',
  '/static/images/icons/icon-512x512.png',
  '/static/images/icons/apple-touch-icon.png',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js',
  'https://code.jquery.com/jquery-3.6.0.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/webfonts/fa-solid-900.woff2',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/webfonts/fa-regular-400.woff2',
  'https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap'
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

// Install event: cache static assets
self.addEventListener('install', event => {
  console.log('[Service Worker] Installing...');
  
  // Skip waiting to ensure the new service worker activates immediately
  self.skipWaiting();
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('[Service Worker] Caching static assets');
        return cache.addAll(STATIC_CACHE_URLS);
      })
      .then(() => {
        // Special handling for Android - prefetch critical assets
        if (/Android/i.test(self.navigator.userAgent)) {
          console.log('[Service Worker] Prefetching Android-specific assets');
          return caches.open(CACHE_NAME).then(cache => {
            return cache.addAll(ANDROID_CRITICAL_ASSETS);
          });
        }
        return Promise.resolve();
      })
      .catch(error => {
        console.error('[Service Worker] Cache installation failed:', error);
      })
  );
});

// Activate event: clean up old caches
self.addEventListener('activate', event => {
  console.log('[Service Worker] Activating...');
  
  // Claim clients to ensure that the service worker controls all clients immediately
  event.waitUntil(
    clients.claim()
      .then(() => {
        // Remove old caches
        return caches.keys().then(cacheNames => {
          return Promise.all(
            cacheNames.filter(cacheName => {
              return (cacheName !== CACHE_NAME && cacheName !== DYNAMIC_CACHE_NAME);
            }).map(cacheName => {
              console.log('[Service Worker] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            })
          );
        });
      })
      .then(() => {
        // Android-specific optimization: perform fetch for critical routes to warm the cache
        if (/Android/i.test(self.navigator.userAgent)) {
          console.log('[Service Worker] Warming cache for Android');
          return Promise.all(
            ANDROID_CRITICAL_ASSETS.map(url => 
              fetch(new Request(url, { cache: 'reload' }))
                .catch(err => console.log(`Cache warming failed for ${url}: ${err}`))
            )
          );
        }
        return Promise.resolve();
      })
  );
  
  return self.clients.claim();
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

// Fetch event: handle different caching strategies
self.addEventListener('fetch', event => {
  const requestUrl = new URL(event.request.url);
  
  // Skip non-GET requests and browser extensions
  if (event.request.method !== 'GET' || 
      requestUrl.protocol === 'chrome-extension:' ||
      requestUrl.protocol === 'chrome:' ||
      requestUrl.hostname === 'localhost') {
    return;
  }
  
  // 1. Handle standalone mode navigation to root - redirect to splash
  if (event.request.mode === 'navigate' && 
      requestUrl.pathname === '/' && 
      (requestUrl.searchParams.has('standalone') || 
       requestUrl.searchParams.has('source'))) {
    event.respondWith(
      fetch(new Request('/splash?standalone=true', { 
        credentials: 'include', 
        mode: event.request.mode,
        headers: event.request.headers
      }))
    );
    return;
  }
  
  // 2. For API requests - Network First strategy
  if (isApiRequest(requestUrl)) {
    event.respondWith(networkFirstStrategy(event.request));
    return;
  }
  
  // 3. For HTML navigation - Network First with offline fallback
  if (isNavigationRequest(event.request)) {
    event.respondWith(
      networkFirstStrategy(event.request)
        .catch(() => {
          return caches.match('/offline')
            .then(response => {
              return response || fetch(event.request);
            });
        })
    );
    return;
  }
  
  // 4. For static assets - Cache First strategy
  if (isStaticAsset(requestUrl)) {
    event.respondWith(cacheFirstStrategy(event.request));
    return;
  }
  
  // 5. Default - Stale While Revalidate strategy
  event.respondWith(staleWhileRevalidateStrategy(event.request));
});

// Cache First Strategy: Try cache first, fallback to network and cache the response
function cacheFirstStrategy(request) {
  return caches.match(request)
    .then(cachedResponse => {
      if (cachedResponse) {
        // Return cached response and update cache in background for Android
        if (isAndroidDevice()) {
          updateCache(request);
        }
        return cachedResponse;
      }
      
      // If not in cache, fetch from network and add to cache
      return fetchAndCache(request);
    });
}

// Network First Strategy: Try network first, fallback to cache
function networkFirstStrategy(request) {
  return fetch(request)
    .then(networkResponse => {
      // Clone the response before using it
      const responseToCache = networkResponse.clone();
      
      // Only cache valid responses
      if (networkResponse.ok && shouldCache(request.url)) {
        const cacheName = request.url.includes('/api/') ? DYNAMIC_CACHE_NAME : CACHE_NAME;
        caches.open(cacheName)
          .then(cache => {
            cache.put(request, responseToCache);
          });
      }
      
      return networkResponse;
    })
    .catch(() => {
      // If network fails, try to get from cache
      return caches.match(request)
        .then(cachedResponse => {
          if (cachedResponse) {
            return cachedResponse;
          }
          
          // If it's a navigation request and we still don't have a cached response,
          // return the offline page for Android
          if (isNavigationRequest(request) && isAndroidDevice()) {
            return caches.match('/offline');
          }
          
          throw new Error('No cached response available');
        });
    });
}

// Stale While Revalidate: Return cached version immediately, then update cache
function staleWhileRevalidateStrategy(request) {
  return caches.match(request)
    .then(cachedResponse => {
      // Start network fetch in parallel
      const fetchPromise = fetchAndCache(request);
      
      // Return the cached response immediately if we have one
      return cachedResponse || fetchPromise;
    });
}

// Helper: Fetch from network and cache response
function fetchAndCache(request) {
  return fetch(request)
    .then(networkResponse => {
      // Only cache valid & GET responses
      if (networkResponse.ok && request.method === 'GET' && shouldCache(request.url)) {
        const responseToCache = networkResponse.clone();
        const cacheName = isApiRequest(new URL(request.url)) ? DYNAMIC_CACHE_NAME : CACHE_NAME;
        
        caches.open(cacheName)
          .then(cache => {
            cache.put(request, responseToCache);
          })
          .catch(err => {
            console.error(`[Service Worker] Error caching ${request.url}:`, err);
          });
      }
      
      return networkResponse;
    });
}

// Helper: Update cache with a fresh network response
function updateCache(request) {
  return fetch(request)
    .then(networkResponse => {
      if (networkResponse.ok && shouldCache(request.url)) {
        const cacheName = isApiRequest(new URL(request.url)) ? DYNAMIC_CACHE_NAME : CACHE_NAME;
        
        return caches.open(cacheName)
          .then(cache => {
            return cache.put(request, networkResponse);
          });
      }
      return networkResponse;
    })
    .catch(err => {
      console.log('[Service Worker] Background update failed:', err);
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

console.log('[Service Worker] Script loaded and ready');