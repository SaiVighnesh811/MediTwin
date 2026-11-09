const { body, check, validationResult } = require('express-validator');

const BRFSS_FEATURES = {
  HighBP: { min: 0, max: 1 },
  HighChol: { min: 0, max: 1 },
  CholCheck: { min: 0, max: 1 },
  BMI: { min: 12, max: 98 },
  Smoker: { min: 0, max: 1 },
  Stroke: { min: 0, max: 1 },
  HeartDiseaseorAttack: { min: 0, max: 1 },
  PhysActivity: { min: 0, max: 1 },
  Fruits: { min: 0, max: 1 },
  Veggies: { min: 0, max: 1 },
  HvyAlcoholConsump: { min: 0, max: 1 },
  AnyHealthcare: { min: 0, max: 1 },
  NoDocbcCost: { min: 0, max: 1 },
  GenHlth: { min: 1, max: 5 },
  MentHlth: { min: 0, max: 30 },
  PhysHlth: { min: 0, max: 30 },
  DiffWalk: { min: 0, max: 1 },
  Sex: { min: 0, max: 1 },
  Age: { min: 1, max: 13 },
  Education: { min: 1, max: 6 },
  Income: { min: 1, max: 8 },
};

const validateFeatures = [
  // Reject unknown fields in body
  body().custom((value, { req }) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error('Request body must be a JSON object');
    }
    const keys = Object.keys(value);
    const unknown = keys.filter((k) => !(k in BRFSS_FEATURES));
    if (unknown.length > 0) {
      throw new Error(`Unknown feature(s) not permitted: ${unknown.join(', ')}`);
    }
    return true;
  }),
  // Validate individual field numeric ranges if present
  ...Object.entries(BRFSS_FEATURES).map(([feat, limits]) =>
    body(feat)
      .optional()
      .isFloat({ min: limits.min, max: limits.max })
      .withMessage(`${feat} must be a number between ${limits.min} and ${limits.max}`)
      .toFloat()
  ),
];

const validateNlp = [
  body('text')
    .isString()
    .withMessage('text must be a string')
    .trim()
    .notEmpty()
    .withMessage('text cannot be empty')
    .isLength({ max: 20000 })
    .withMessage('text exceeds maximum length of 20,000 characters'),
];

function handleAnalysisValidationErrors(req, res, next) {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      error: 'Validation failed',
      details: errors.array().map((err) => ({
        field: err.path || err.param,
        message: err.msg,
      })),
    });
  }
  next();
}

module.exports = {
  BRFSS_FEATURES,
  validateFeatures,
  validateNlp,
  handleAnalysisValidationErrors,
};

