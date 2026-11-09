const express = require('express');
const rateLimit = require('express-rate-limit');

const { requireAuth } = require('../middleware/auth');
const { upload, handleUploadError } = require('../middleware/upload');
const {
  validateFeatures,
  validateNlp,
  handleAnalysisValidationErrors,
} = require('../middleware/analysisValidate');
const {
  analyzeRisk,
  explainRisk,
  analyzeNLP,
  analyzeImage,
  analyzeOCR,
  runFullAnalysis,
  listAnalyses,
  getAnalysisById,
  deleteAnalysisById,
} = require('../controllers/analysis.controller');

const router = express.Router();

// Rate limit: 20 requests per 10 minutes per user
const analysisRateLimiter = rateLimit({
  windowMs: 10 * 60 * 1000, // 10 minutes
  max: 20,
  keyGenerator: (req) => req.user?._id?.toString() || 'anonymous',
  message: { error: 'Too many analysis requests. Please try again in 10 minutes.' },
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false, default: true },
});

// Protect all analysis routes with authentication
router.use(requireAuth);

// 1. Prediction & Extraction Endpoints (rate limited)
router.post(
  '/risk',
  analysisRateLimiter,
  validateFeatures,
  handleAnalysisValidationErrors,
  analyzeRisk
);

router.post(
  '/explain',
  analysisRateLimiter,
  validateFeatures,
  handleAnalysisValidationErrors,
  explainRisk
);

router.post(
  '/nlp',
  analysisRateLimiter,
  validateNlp,
  handleAnalysisValidationErrors,
  analyzeNLP
);

router.post(
  '/image',
  analysisRateLimiter,
  upload.single('file'),
  handleUploadError,
  analyzeImage
);

router.post(
  '/ocr',
  analysisRateLimiter,
  upload.single('file'),
  handleUploadError,
  analyzeOCR
);

router.post(
  '/analyze',
  analysisRateLimiter,
  upload.fields([
    { name: 'document_image', maxCount: 1 },
    { name: 'xray_image', maxCount: 1 },
  ]),
  handleUploadError,
  runFullAnalysis
);

// 2. Saved Analysis Query Endpoints
router.get('/', listAnalyses);
router.get('/:id', getAnalysisById);
router.delete('/:id', deleteAnalysisById);

module.exports = router;
