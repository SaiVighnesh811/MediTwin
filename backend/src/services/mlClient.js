/**
 * Client service for dispatching authorized requests to the ML microservice.
 * Manages timeout, header injection, and centralized error classification.
 *
 * Timeout policy:
 *   /v1/analyze  → 250 s (ML service budget is 240 s; 10 s margin)
 *   all others   → process.env.ML_TIMEOUT_MS (default 130 s)
 *
 * Error mapping:
 *   503  → lock-wait timeout  → "busy with another analysis"
 *   504  → execution timeout  → "took too long / retry in a minute"
 *   AbortError (gateway) → 504 message (same wording as ML 504)
 */
class MLServiceError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.name = 'MLServiceError';
    this.statusCode = statusCode;
  }
}

const ANALYZE_TIMEOUT_MS = 250_000; // 250 s for /v1/analyze
const DEFAULT_TIMEOUT_MS = parseInt(process.env.ML_TIMEOUT_MS || '130000', 10);

const MSG_503 = 'The analysis service is busy with another analysis. Please retry shortly.';
const MSG_504 = 'The analysis took too long, probably because the models are still loading. Please retry in a minute.';

async function forwardToMLService(endpoint, options = {}) {
  const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000';
  const internalKey = process.env.INTERNAL_ML_KEY || '';

  const isAnalyze = endpoint === '/v1/analyze';
  const timeoutMs = isAnalyze ? ANALYZE_TIMEOUT_MS : DEFAULT_TIMEOUT_MS;

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
        // Try to read the ML service's own message; fall back to canonical wording
        const body = await response.json().catch(() => ({}));
        const msg = (typeof body?.error === 'string' && body.error) ? body.error : MSG_503;
        throw new MLServiceError(msg, 503);
      }
      if (response.status === 504) {
        const body = await response.json().catch(() => ({}));
        const msg = (typeof body?.error === 'string' && body.error) ? body.error : MSG_504;
        throw new MLServiceError(msg, 504);
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
      // Gateway-level timeout — same user-facing message as ML 504
      throw new MLServiceError(MSG_504, 504);
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
