const BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000';

// One shared refresh call: parallel 401s must not each rotate the refresh token.
let refreshing = null;
function refreshOnce() {
  if (!refreshing) {
    refreshing = fetch(`${BASE}/api/auth/refresh`, { method: 'POST', credentials: 'include' })
      .then((r) => r.ok)
      .catch(() => false)
      .finally(() => { refreshing = null; });
  }
  return refreshing;
}

export async function api(path, opts = {}, flags = {}) {
  const url = `${BASE}${path}`;

  const isFormData = typeof FormData !== 'undefined' && opts.body instanceof FormData;
  const defaultHeaders = isFormData ? {} : { 'Content-Type': 'application/json' };

  const defaults = {
    credentials: 'include',
    headers: { ...defaultHeaders, ...opts.headers },
  };

  const merged = { ...defaults, ...opts, headers: { ...defaults.headers, ...opts.headers } };

  // Set timeout if specified (or abort signal)
  let timeoutId = null;
  if (flags.timeoutMs && !merged.signal) {
    const controller = new AbortController();
    merged.signal = controller.signal;
    timeoutId = setTimeout(() => controller.abort(), flags.timeoutMs);
  }

  let res;
  try {
    res = await fetch(url, merged);
  } catch (err) {
    if (timeoutId) clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      const timeoutErr = new Error('The analysis took too long, probably because the models are still loading. Please retry in a minute.');
      timeoutErr.status = 504;
      throw timeoutErr;
    }
    throw err;
  }
  if (timeoutId) clearTimeout(timeoutId);

  if (res.status === 401 && !flags.skipRefresh) {
    if (await refreshOnce()) {
      res = await fetch(url, merged);
    }
  }

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    let message =
      data.details?.map((d) => d.message).join('. ') ||
      data.error;

    if (!message) {
      if (res.status === 502) message = 'The analysis service is unavailable, try again';
      else if (res.status === 503) message = 'The analysis service is busy with another analysis. Please retry shortly.';
      else if (res.status === 504) message = 'The analysis took too long, probably because the models are still loading. Please retry in a minute.';
      else if (res.status === 429) message = 'Too many analyses, wait a few minutes';
      else message = `Request failed (${res.status})`;
    }

    const err = new Error(message);
    err.status = res.status;
    err.data = data;
    throw err;
  }

  return data;
}

export const analysisAPI = {
  analyze: (formData) => api('/api/analysis/analyze', { method: 'POST', body: formData }, { timeoutMs: 260000 }),
  image: (formData) => api('/api/analysis/image', { method: 'POST', body: formData }, { timeoutMs: 140000 }),
  list: (page = 1, limit = 20) => api(`/api/analysis?page=${page}&limit=${limit}`),
  get: (id) => api(`/api/analysis/${id}`),
  delete: (id) => api(`/api/analysis/${id}`, { method: 'DELETE' }),
};

export const authAPI = {
  signup: (body) => api('/api/auth/signup', { method: 'POST', body: JSON.stringify(body) }, { skipRefresh: true }),
  login:  (body) => api('/api/auth/login',  { method: 'POST', body: JSON.stringify(body) }, { skipRefresh: true }),
  logout: ()     => api('/api/auth/logout', { method: 'POST' }),
  refresh: ()    => api('/api/auth/refresh', { method: 'POST' }, { skipRefresh: true }),
  me: ()         => api('/api/auth/me'),
};
