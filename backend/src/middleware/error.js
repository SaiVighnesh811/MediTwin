function errorHandler(err, req, res, next) {
  const statusCode = err.statusCode || (res.statusCode === 200 ? 500 : res.statusCode);
  
  if (process.env.NODE_ENV !== 'test') {
    console.error(`[Error] ${err.message}`, err.stack);
  }

  res.status(statusCode).json({
    error: err.message || 'Internal Server Error',
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
  });
}

function notFoundHandler(req, res, next) {
  res.status(404).json({ error: `Not Found - ${req.originalUrl}` });
}

module.exports = { errorHandler, notFoundHandler };

