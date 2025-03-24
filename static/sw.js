const CACHE_NAME = 'memorycare-v1';
const urlsToCache = [
  '/',
  '/login',
  '/register',
  '/offline',
  '/splash',
  '/static/css/style.css',
  '/static/js/pwa-install.js',
  '/static/manifest.json',
  '/static/images/icons/icon-192x192.svg',
  '/static/images/icons/icon-512x512.svg',
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

// Cache and return requests
self.addEventListener('fetch', event => {
  // Handle direct navigation to the app's root
  if (event.request.mode === 'navigate' && 
      event.request.url.includes('standalone=true') && 
      !event.request.url.includes('/splash')) {
    
    // Construct the splash screen URL with standalone parameter
    const splashUrl = new URL('/splash', self.location.origin);
    splashUrl.searchParams.set('standalone', 'true');
    
    console.log('[Service Worker] Redirecting to splash screen');
    event.respondWith(fetch(splashUrl.toString()));
    return;
  }

  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Cache hit - return response
        if (response) {
          return response;
        }

        // Clone the request
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
            // Check if the request is for a page navigation
            if (event.request.mode === 'navigate') {
              return caches.match('/offline');
            }
          });
      })
  );
});

// Update service worker
self.addEventListener('activate', event => {
  console.log('Service Worker: Activating...');
  const cacheWhitelist = [CACHE_NAME];
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheWhitelist.indexOf(cacheName) === -1) {
            console.log('Service Worker: Clearing Old Cache');
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});