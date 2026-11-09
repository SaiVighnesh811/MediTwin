const multer = require('multer');

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 8 * 1024 * 1024, // 8 MB
  },
  fileFilter: (req, file, cb) => {
    const allowedMimes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (allowedMimes.includes(file.mimetype.toLowerCase())) {
      cb(null, true);
    } else {
      const err = new Error('Invalid file type. Only JPEG and PNG images are allowed.');
      err.statusCode = 400;
      cb(err, false);
    }
  },
});

function handleUploadError(err, req, res, next) {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'File size exceeds 8 MB limit' });
    }
    return res.status(400).json({ error: `Upload error: ${err.message}` });
  }
  if (err && err.statusCode === 400) {
    return res.status(400).json({ error: err.message });
  }
  next(err);
}

module.exports = {
  upload,
  handleUploadError,
};

