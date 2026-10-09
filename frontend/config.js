// Where the prediction API lives.
//   ''  -> same origin (local: python scripts/serve_app.py serves frontend + API)
//   URL -> separately hosted backend (e.g. the Render service)
// On localhost the local server is always used.
window.NEUROPHARMA_API_BASE = ['localhost', '127.0.0.1'].includes(window.location.hostname)
  ? ''
  : 'https://neuropharma-api.onrender.com';
