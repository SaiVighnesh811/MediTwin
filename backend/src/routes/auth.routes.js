const express = require('express');
const rateLimit = require('express-rate-limit');
const {
  signup,
  login,
  logout,
  refresh,
  me,
} = require('../controllers/auth.controller');
const { requireAuth } = require('../middleware/auth');
const {
  validateSignup,
  validateLogin,
  handleValidationErrors,
} = require('../middleware/validate');

const router = express.Router();

// Rate limiter: 10 requests per 15 minutes per IP for signup and login
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: 'Too many attempts from this IP, please try again after 15 minutes',
  },
  skip: () => process.env.NODE_ENV === 'test', // bypass rate limit during unit tests
});

router.post('/signup', authLimiter, validateSignup, handleValidationErrors, signup);
router.post('/login', authLimiter, validateLogin, handleValidationErrors, login);
router.post('/logout', logout);
router.post('/refresh', refresh);
router.get('/me', requireAuth, me);

module.exports = router;

