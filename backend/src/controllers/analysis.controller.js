const mongoose = require('mongoose');
const Analysis = require('../models/Analysis');
const { forwardToMLService } = require('../services/mlClient');

/**
 * POST /api/analysis/risk
 */
async function analyzeRisk(req, res, next) {
  try {
    const features = req.body || {};
    const result = await forwardToMLService('/v1/risk', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(features),
    });

    const analysis = await Analysis.create({
      userId: req.user._id,
      type: 'risk',
      inputSummary: { featuresProvided: features },
      result,
      reliability: result.reliability || null,
    });

    return res.status(200).json({
      analysisId: analysis._id,
      ...result,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/analysis/explain
 */
async function explainRisk(req, res, next) {
  try {
    const features = req.body || {};
    const result = await forwardToMLService('/v1/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(features),
    });

    const analysis = await Analysis.create({
      userId: req.user._id,
      type: 'explain',
      inputSummary: { featuresProvided: features },
      result,
      reliability: result.reliability || null,
    });

    return res.status(200).json({
      analysisId: analysis._id,
      ...result,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/analysis/nlp
 */
async function analyzeNLP(req, res, next) {
  try {
    const { text } = req.body;
    const result = await forwardToMLService('/v1/nlp', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    const analysis = await Analysis.create({
      userId: req.user._id,
      type: 'nlp',
      inputSummary: { textLength: text.length, snippet: text.slice(0, 100) },
      result,
      reliability: null,
    });

    return res.status(200).json({
      analysisId: analysis._id,
      ...result,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/analysis/image
 */
async function analyzeImage(req, res, next) {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Image file is required' });
    }

    // age_years is required to select the correct X-ray model
    const rawAge = req.body.age_years !== undefined ? req.body.age_years : req.body.age;
    const ageYears = rawAge !== undefined && rawAge !== '' ? parseInt(rawAge, 10) : NaN;
    if (Number.isNaN(ageYears) || ageYears < 0 || ageYears > 120) {
      return res
        .status(400)
        .json({ error: 'age_years (0-120) is required for chest X-ray analysis' });
    }

    const formData = new FormData();
    const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
    formData.append('file', blob, req.file.originalname);
    formData.append('age_years', String(ageYears));

    const result = await forwardToMLService('/v1/image', {
      method: 'POST',
      body: formData,
    });

    const analysis = await Analysis.create({
      userId: req.user._id,
      type: 'image',
      inputSummary: {
        filename: req.file.originalname,
        size: req.file.size,
        mimetype: req.file.mimetype,
        age_years: ageYears,
      },
      result,
      reliability: null,
    });

    return res.status(200).json({
      analysisId: analysis._id,
      ...result,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/analysis/ocr
 */
async function analyzeOCR(req, res, next) {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Document image file is required' });
    }

    const formData = new FormData();
    const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
    formData.append('file', blob, req.file.originalname);
    if (req.body.handwriting) {
      formData.append('handwriting', String(req.body.handwriting));
    }

    const result = await forwardToMLService('/v1/ocr', {
      method: 'POST',
      body: formData,
    });

    const analysis = await Analysis.create({
      userId: req.user._id,
      type: 'ocr',
      inputSummary: {
        filename: req.file.originalname,
        size: req.file.size,
        mimetype: req.file.mimetype,
        handwriting: Boolean(req.body.handwriting),
      },
      result,
      reliability: null,
    });

    return res.status(200).json({
      analysisId: analysis._id,
      ...result,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/analysis/analyze
 */
async function runFullAnalysis(req, res, next) {
  try {
    const formData = new FormData();

    // Enforce patient_id is ALWAYS the logged-in user's id
    formData.append('patient_id', req.user._id.toString());

    if (req.body.features) {
      formData.append('features', req.body.features);
    }
    if (req.body.text) {
      formData.append('text', req.body.text);
    }

    const inputSummary = {
      hasFeatures: Boolean(req.body.features),
      hasText: Boolean(req.body.text),
    };

    if (req.files) {
      if (req.files.document_image && req.files.document_image[0]) {
        const docFile = req.files.document_image[0];
        const blob = new Blob([docFile.buffer], { type: docFile.mimetype });
        formData.append('document_image', blob, docFile.originalname);
        inputSummary.documentImage = {
          filename: docFile.originalname,
          size: docFile.size,
          mimetype: docFile.mimetype,
        };
      }
      if (req.files.xray_image && req.files.xray_image[0]) {
        const xrayFile = req.files.xray_image[0];
        const blob = new Blob([xrayFile.buffer], { type: xrayFile.mimetype });
        formData.append('xray_image', blob, xrayFile.originalname);
        inputSummary.xrayImage = {
          filename: xrayFile.originalname,
          size: xrayFile.size,
          mimetype: xrayFile.mimetype,
        };
      }
    }

    // Forward age_years so the ML service can route to the correct X-ray model
    if (req.body.age_years !== undefined && req.body.age_years !== '') {
      formData.append('age_years', String(req.body.age_years));
    }

    const result = await forwardToMLService('/v1/analyze', {
      method: 'POST',
      body: formData,
    });

    const analysis = await Analysis.create({
      userId: req.user._id,
      type: 'analyze',
      inputSummary,
      result,
      reliability: result.reliability || null,
    });

    return res.status(200).json({
      analysisId: analysis._id,
      ...result,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/analysis (own analyses only, newest first, paginated)
 */
async function listAnalyses(req, res, next) {
  try {
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const limit = Math.min(50, Math.max(1, parseInt(req.query.limit, 10) || 20));
    const skip = (page - 1) * limit;

    const query = { userId: req.user._id };
    const [analyses, total] = await Promise.all([
      Analysis.find(query)
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limit)
        .lean(),
      Analysis.countDocuments(query),
    ]);

    return res.status(200).json({
      page,
      limit,
      total,
      pages: Math.ceil(total / limit) || 1,
      analyses,
    });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/analysis/:id
 */
async function getAnalysisById(req, res, next) {
  try {
    const { id } = req.params;
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(404).json({ error: 'Analysis not found' });
    }

    // Must belong to the logged-in user; return 404 if not found or belongs to another user
    const analysis = await Analysis.findOne({ _id: id, userId: req.user._id });
    if (!analysis) {
      return res.status(404).json({ error: 'Analysis not found' });
    }

    return res.status(200).json(analysis);
  } catch (err) {
    next(err);
  }
}

/**
 * DELETE /api/analysis/:id
 */
async function deleteAnalysisById(req, res, next) {
  try {
    const { id } = req.params;
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(404).json({ error: 'Analysis not found' });
    }

    const analysis = await Analysis.findOneAndDelete({ _id: id, userId: req.user._id });
    if (!analysis) {
      return res.status(404).json({ error: 'Analysis not found' });
    }

    return res.status(200).json({ message: 'Analysis deleted successfully' });
  } catch (err) {
    next(err);
  }
}

module.exports = {
  analyzeRisk,
  explainRisk,
  analyzeNLP,
  analyzeImage,
  analyzeOCR,
  runFullAnalysis,
  listAnalyses,
  getAnalysisById,
  deleteAnalysisById,
};

