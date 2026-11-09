const os = require('os');
const mongoose = require('mongoose');

async function connectDB(uri) {
  const mongoURI = uri || process.env.MONGODB_URI;
  if (!mongoURI) {
    throw new Error('MongoDB connection URI not specified');
  }

  try {
    const conn = await mongoose.connect(mongoURI, {
      runtimeAdapters: { os },
    });
    if (process.env.NODE_ENV !== 'test') {
      console.log(`MongoDB connected: ${conn.connection.host}`);
    }
    return conn;
  } catch (error) {
    if (process.env.NODE_ENV !== 'test') {
      console.error(`MongoDB connection error: ${error.message}`);
    }
    throw error;
  }
}

async function disconnectDB() {
  await mongoose.disconnect();
}

module.exports = { connectDB, disconnectDB };

