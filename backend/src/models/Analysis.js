const mongoose = require('mongoose');

const analysisSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true,
  },
  type: {
    type: String,
    required: true,
    enum: ['risk', 'explain', 'image', 'ocr', 'nlp', 'analyze'],
  },
  createdAt: {
    type: Date,
    default: Date.now,
    index: true,
  },
  inputSummary: {
    type: mongoose.Schema.Types.Mixed,
    default: {},
  },
  result: {
    type: mongoose.Schema.Types.Mixed,
    required: true,
  },
  reliability: {
    type: mongoose.Schema.Types.Mixed,
    default: null,
  },
});

analysisSchema.set('toJSON', {
  transform: (doc, ret) => {
    delete ret.__v;
    return ret;
  },
});

module.exports = mongoose.model('Analysis', analysisSchema);

