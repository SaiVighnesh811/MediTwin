require('dotenv').config();

const app = require('./app');
const { validateEnv } = require('./config/env');
const { connectDB } = require('./config/db');

// Validate environment variables before startup
validateEnv();

const PORT = process.env.PORT || 5000;

async function startServer() {
  try {
    await connectDB();
    const server = app.listen(PORT, () => {
      console.log(`Backend server running on http://localhost:${PORT}`);
      console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
    });

    const shutdown = async (signal) => {
      console.log(`\nReceived ${signal}. Gracefully shutting down...`);
      server.close(async () => {
        const { disconnectDB } = require('./config/db');
        await disconnectDB();
        console.log('MongoDB disconnected and server closed.');
        process.exit(0);
      });
    };

    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));
  } catch (err) {
    console.error(`Failed to start server: ${err.message}`);
    process.exit(1);
  }
}

startServer();

