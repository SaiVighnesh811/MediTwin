/**
 * Client service for dispatching authorized requests to the ML microservice.
 * Manages timeout, header injection, and centralized error classification.
 */
class MLServiceError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.name = 'MLServiceError';
    this.statusCode = statusCode;
  }
}

async function forwardToMLService(endpoint, options = {}) {
  const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000';
  const internalKey = process.env.INTERNAL_ML_KEY || '';
  const timeoutMs = parseInt(process.env.ML_TIMEOUT_MS || '130000', 10);

  const url = `${mlServiceUrl}${endpoint}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const headers = {
    ...(options.headers || {}),
    'X-Internal-Key': internalKey,
  };

  try {
    const response = await fetch(url, {
      ...options,
      headers,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        console.error('[MLGateway] Key misconfigured or unauthorized response from ML Service. Check INTERNAL_ML_KEY.');
        throw new MLServiceError('Analysis service unavailable', 502);
      }
      if (response.status === 503) {
        throw new MLServiceError('Analysis service is currently busy running another model. Please retry shortly.', 503);
      }
      if (response.status === 400) {
        const errorBody = await response.json().catch(() => ({}));
        const msg = typeof errorBody.error === 'string' ? errorBody.error : 'Invalid request to analysis service';
        throw new MLServiceError(msg, 400);
      }
      if (response.status === 422) {
        const errorBody = await response.json().catch(() => ({}));
        const msg = typeof errorBody.error === 'string' ? errorBody.error : 'Validation error in ML service';
        throw new MLServiceError(msg, 400);
      }
      if (response.status >= 500) {
        throw new MLServiceError('Analysis service unavailable', 502);
      }
      throw new MLServiceError('Analysis service request failed', response.status);
    }

    const data = await response.json();
    return data;
  } catch (err) {
    clearTimeout(timeoutId);

    if (err instanceof MLServiceError) {
      throw err;
    }

    if (err.name === 'AbortError') {
      throw new MLServiceError('Analysis request timed out', 504);
    }

    // Network errors (e.g. connection refused, ML service down)
    console.error(`[MLGateway] Failed to connect to ML service at ${url}: ${err.message}`);
    throw new MLServiceError('Analysis service unavailable', 502);
  }
}

module.exports = {
  forwardToMLService,
  MLServiceError,
};
