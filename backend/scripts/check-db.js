const path = require('path');
const os = require('os');
const dotenv = require('dotenv');

dotenv.config({ path: path.join(__dirname, '..', '.env') });

const mongoose = require('mongoose');

function sanitizeError(msg) {
  if (!msg) return '';
  return msg.replace(/:([^@/:]+)@/g, ':***@');
}

async function checkDatabase() {
  const uri = process.env.MONGODB_URI;
  if (!uri) {
    console.error('Error: MONGODB_URI is not set in backend/.env');
    process.exit(1);
  }

  try {
    await mongoose.connect(uri, {
      runtimeAdapters: { os },
      serverSelectionTimeoutMS: 7000,
    });

    await mongoose.connection.db.admin().ping();
    const dbName = mongoose.connection.name;
    console.log(`MongoDB connection OK, database: ${dbName}`);

    await mongoose.disconnect();
    process.exit(0);
  } catch (err) {
    const safeError = sanitizeError(err.message || String(err));
    console.error(`MongoDB connection FAILED: ${safeError}`);
    console.error('\nLikely causes:');
    console.error('  1. Atlas Network Access IP not allowed (Current IP not whitelisted in Atlas Network Access)');
    console.error('  2. Wrong database user permissions');
    console.error('  3. Wrong password or credentials');
    
    try {
      await mongoose.disconnect();
    } catch (_) {}
    process.exit(1);
  }
}

checkDatabase();

