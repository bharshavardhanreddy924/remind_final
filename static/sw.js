const CACHE_NAME = 'memorycare-v2';
const urlsToCache = [
  '/',
  '/login',
  '/register',
  '/offline',
  '/splash',
  '/dashboard',
  '/static/css/style.css',
  '/static/js/pwa-install.js',
  '/static/manifest.json',
  '/static/images/icons/icon-16x16.png',
  '/static/images/icons/icon-32x32.png',
  '/static/images/icons/icon-144x144.png',
  '/static/images/icons/icon-192x192.png',
  '/static/images/icons/icon-512x512.png',
  '/static/images/icons/apple-touch-icon.png',
  '/static/images/icons/badge-72x72.png',
  '/static/images/icons/favicon.ico',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js',
  'https://code.jquery.com/jquery-3.6.0.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
];

// Install a service worker
self.addEventListener('install', event => {
  console.log('Service Worker: Installing...');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Service Worker: Caching Files');
        return cache.addAll(urlsToCache);
      })
      .then(() => self.skipWaiting())
  );
});

// Cache and return requests with improved strategy
self.addEventListener('fetch', event => {
  // Special handling for navigation requests (HTML pages)
  if (event.request.mode === 'navigate') {
    // Handle direct navigation to the app's root with standalone parameter
    if (event.request.url.includes('standalone=true') && 
        !event.request.url.includes('/splash')) {
      
      // Construct the splash screen URL with standalone parameter
      const splashUrl = new URL('/splash', self.location.origin);
      splashUrl.searchParams.set('standalone', 'true');
      splashUrl.searchParams.set('source', 'pwa');
      
      console.log('[Service Worker] Redirecting to splash screen');
      event.respondWith(fetch(splashUrl.toString()));
      return;
    }

    // For other navigation requests, try network first then fallback to offline page
    event.respondWith(
      fetch(event.request)
        .catch(() => caches.match('/offline'))
    );
    return;
  }

  // For non-navigation requests, use cache-first strategy
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Cache hit - return response
        if (response) {
          return response;
        }

        // Special handling for API requests - network only, no caching
        if (event.request.url.includes('/api/')) {
          return fetch(event.request);
        }

        // Clone the request for fetch and cache
        const fetchRequest = event.request.clone();

        // Make network request and cache the response
        return fetch(fetchRequest)
          .then(response => {
            // Check if we received a valid response
            if (!response || response.status !== 200 || response.type !== 'basic') {
              return response;
            }

            // Clone the response
            const responseToCache = response.clone();

            // Open the cache
            caches.open(CACHE_NAME)
              .then(cache => {
                // Add response to cache
                cache.put(event.request, responseToCache);
              });

            return response;
          })
          .catch(error => {
            console.log('Service Worker: Fetch Error', error);
            // For image requests that fail, return a placeholder if available
            if (event.request.url.match(/\.(jpg|jpeg|png|gif|svg)$/)) {
              return caches.match('/static/images/placeholder.png');
            }
          });
      })
  );
});

// Update service worker and clean old caches
self.addEventListener('activate', event => {
  console.log('Service Worker: Activating...');
  const cacheWhitelist = [CACHE_NAME];
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheWhitelist.indexOf(cacheName) === -1) {
            console.log('Service Worker: Clearing Old Cache', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('Service Worker: Claiming Clients');
      return self.clients.claim();
    })
  );
});

// Handle push notifications
self.addEventListener('push', event => {
  const data = event.data.json();
  const options = {
    body: data.body,
    icon: '/static/images/icons/icon-192x192.png',
    badge: '/static/images/icons/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      url: data.url || '/'
    }
  };

  event.waitUntil(
    self.registration.showNotification(data.title, options)
  );
});

// Handle notification click
self.addEventListener('notificationclick', event => {
  event.notification.close();
  
  event.waitUntil(
    clients.matchAll({type: 'window'}).then(windowClients => {
      const url = event.notification.data.url;
      
      // Check if there is already a window/tab open with the target URL
      for (let i = 0; i < windowClients.length; i++) {
        const client = windowClients[i];
        // If so, focus it
        if (client.url === url && 'focus' in client) {
          return client.focus();
        }
      }
      
      // If not, open a new window/tab
      if (clients.openWindow) {
        return clients.openWindow(url);
      }
    })
  );
});
