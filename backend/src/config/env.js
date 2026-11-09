const requiredEnvVars = ['MONGODB_URI', 'JWT_ACCESS_SECRET', 'JWT_REFRESH_SECRET'];

function validateEnv() {
  if (process.env.NODE_ENV === 'test') {
    return;
  }

  const missing = requiredEnvVars.filter((varName) => !process.env[varName] || process.env[varName].trim() === '');

  if (missing.length > 0) {
    console.error('\n============================================================');
    console.error('ERROR: Missing required environment variable(s):');
    missing.forEach((v) => console.error(`  - ${v}`));
    console.error('\nPlease copy .env.example to .env and fill in the missing values.');
    console.error('============================================================\n');
    process.exit(1);
  }
}

module.exports = { validateEnv, requiredEnvVars };

