const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const cookieParser = require('cookie-parser');
const morgan = require('morgan');

const authRoutes = require('./routes/auth.routes');
const analysisRoutes = require('./routes/analysis.routes');
const { errorHandler, notFoundHandler } = require('./middleware/error');

const app = express();

// Security headers
app.use(helmet());

// CORS configuration: strict CLIENT_ORIGIN with credentials
const clientOrigin = process.env.CLIENT_ORIGIN || 'http://localhost:5173';
app.use(
  cors({
    origin: clientOrigin,
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
  })
);

// Body parsers and cookies
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());

// HTTP logger (skip in test environment)
if (process.env.NODE_ENV !== 'test') {
  app.use(morgan('dev'));
}

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.status(200).json({
    status: 'ok',
    timestamp: new Date().toISOString(),
  });
});

// Auth routes
app.use('/api/auth', authRoutes);

// Analysis gateway routes
app.use('/api/analysis', analysisRoutes);

// 404 & Central error handler
app.use(notFoundHandler);
app.use(errorHandler);

module.exports = app;

