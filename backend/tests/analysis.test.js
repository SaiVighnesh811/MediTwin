const http = require('http');
const request = require('supertest');
const { MongoMemoryServer } = require('mongodb-memory-server');
const mongoose = require('mongoose');

process.env.NODE_ENV = 'test';
process.env.JWT_ACCESS_SECRET = 'super-secret-access-token-key-32chars';
process.env.JWT_REFRESH_SECRET = 'super-secret-refresh-token-key-32chars';
process.env.INTERNAL_ML_KEY = 'test-secret-internal-ml-key';
process.env.ML_TIMEOUT_MS = '500'; // short timeout (500ms) for timeout test

jest.setTimeout(60000);

const app = require('../src/app');
const User = require('../src/models/User');
const Analysis = require('../src/models/Analysis');
const { connectDB, disconnectDB } = require('../src/config/db');
const { generateAccessToken, getCookieOptions, ACCESS_COOKIE_MAX_AGE } = require('../src/utils/tokens');

let mongoServer;
let fakeMLServer;
let fakeMLPort;
let lastMLHeaders = {};
let lastMLBody = null;
let lastMLUrl = '';
let simulatedDelayMs = 0;
let simulatedStatus = 200;
let simulatedResponse = {};

beforeAll(async () => {
  // Start in-memory Mongo
  mongoServer = await MongoMemoryServer.create({
    instance: { launchTimeout: 60000 },
  });
  const uri = mongoServer.getUri();
  await connectDB(uri);

  // Start fake ML server on random port
  await new Promise((resolve) => {
    fakeMLServer = http.createServer((req, res) => {
      lastMLUrl = req.url;
      lastMLHeaders = { ...req.headers };
      let bodyData = [];

      req.on('data', (chunk) => bodyData.push(chunk));
      req.on('end', () => {
        lastMLBody = Buffer.concat(bodyData).toString();

        setTimeout(() => {
          if (simulatedStatus === 200) {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(simulatedResponse || { status: 'ok' }));
          } else {
            res.writeHead(simulatedStatus, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(simulatedResponse || { error: 'ML Service Error' }));
          }
        }, simulatedDelayMs);
      });
    });

    fakeMLServer.listen(0, '127.0.0.1', () => {
      fakeMLPort = fakeMLServer.address().port;
      process.env.ML_SERVICE_URL = `http://127.0.0.1:${fakeMLPort}`;
      resolve();
    });
  });
}, 60000);

afterAll(async () => {
  await disconnectDB();
  if (mongoServer) await mongoServer.stop();
  if (fakeMLServer) {
    await new Promise((resolve) => fakeMLServer.close(resolve));
  }
}, 60000);

beforeEach(async () => {
  await User.deleteMany({});
  await Analysis.deleteMany({});
  simulatedDelayMs = 0;
  simulatedStatus = 200;
  simulatedResponse = {
    predicted_class: 'High',
    probabilities: { Low: 0.1, Medium: 0.2, High: 0.7 },
    completeness: '21 of 21 provided',
    reliability: { level: 'high', reasons: ['All features provided'] },
  };
  lastMLHeaders = {};
  lastMLBody = null;
  lastMLUrl = '';
});

// Helper to create user & token cookie
async function createTestUser(email = 'user@example.com', name = 'Test User') {
  const user = await User.create({
    name,
    email,
    passwordHash: 'dummyhash',
  });
  const token = generateAccessToken({ id: user._id, role: user.role });
  const cookie = `accessToken=${token}`;
  return { user, token, cookie };
}

describe('Chapter 13: Analysis Gateway & MongoDB Storage Tests', () => {
  test('1. unauthenticated request -> 401', async () => {
    const res = await request(app).post('/api/analysis/risk').send({ BMI: 25.0 });
    expect(res.status).toBe(401);
    expect(res.body.error).toMatch(/Authentication required/i);
  });

  test('2. wrong file type -> 400', async () => {
    const { cookie } = await createTestUser();
    const res = await request(app)
      .post('/api/analysis/image')
      .set('Cookie', cookie)
      .attach('file', Buffer.from('plain text not image'), 'test.txt');

    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/Invalid file type/i);
  });

  test('3. oversize file -> 400 (exceeds 8 MB limit)', async () => {
    const { cookie } = await createTestUser();
    // 9 MB buffer
    const hugeBuffer = Buffer.alloc(9 * 1024 * 1024);
    const res = await request(app)
      .post('/api/analysis/image')
      .set('Cookie', cookie)
      .attach('file', hugeBuffer, { filename: 'huge.jpg', contentType: 'image/jpeg' });

    expect([400, 413]).toContain(res.status);
    expect(res.body.error).toMatch(/exceeds 8 MB limit|File too large/i);
  });

  test('4. fake ML server receives X-Internal-Key and response does not contain it', async () => {
    const { cookie } = await createTestUser();
    const res = await request(app)
      .post('/api/analysis/risk')
      .set('Cookie', cookie)
      .send({ BMI: 28.5, Age: 9 });

    expect(res.status).toBe(200);
    expect(lastMLHeaders['x-internal-key']).toBe('test-secret-internal-ml-key');

    // The client response must not reveal the key or the internal url
    const resString = JSON.stringify(res.body);
    expect(resString).not.toContain('test-secret-internal-ml-key');
    expect(resString).not.toContain(String(fakeMLPort));
    expect(res.body.analysisId).toBeDefined();
    expect(res.body.reliability).toBeDefined();

    // Verify saved to database
    const saved = await Analysis.findById(res.body.analysisId);
    expect(saved).not.toBeNull();
    expect(saved.type).toBe('risk');
    expect(saved.inputSummary.featuresProvided.BMI).toBe(28.5);
  });

  test('5. client-supplied patient_id is ignored (enforced as logged-in user id)', async () => {
    const { user, cookie } = await createTestUser('alice@example.com', 'Alice');
    const res = await request(app)
      .post('/api/analysis/analyze')
      .set('Cookie', cookie)
      .field('patient_id', 'ATTACKER_CHOSEN_ID')
      .field('features', JSON.stringify({ BMI: 24.0 }));

    expect(res.status).toBe(200);
    expect(lastMLBody).toContain(`name="patient_id"\r\n\r\n${user._id.toString()}`);
    expect(lastMLBody).not.toContain('ATTACKER_CHOSEN_ID');
  });

  test('6. ML down -> 502 {"error": "Analysis service unavailable"}', async () => {
    const { cookie } = await createTestUser();
    // Temporarily point to a non-existent port
    const originalUrl = process.env.ML_SERVICE_URL;
    process.env.ML_SERVICE_URL = 'http://127.0.0.1:59999';

    try {
      const res = await request(app)
        .post('/api/analysis/risk')
        .set('Cookie', cookie)
        .send({ BMI: 25.0 });

      expect(res.status).toBe(502);
      expect(res.body.error).toBe('Analysis service unavailable');
    } finally {
      process.env.ML_SERVICE_URL = originalUrl;
    }
  });

  test('7. ML slow -> 504 (timeout handler)', async () => {
    const { cookie } = await createTestUser();
    simulatedDelayMs = 1200; // longer than ML_TIMEOUT_MS (500ms)

    const res = await request(app)
      .post('/api/analysis/risk')
      .set('Cookie', cookie)
      .send({ BMI: 25.0 });

    expect(res.status).toBe(504);
    expect(res.body.error).toMatch(/took too long|timed out/i);
  });

  test('8. user A cannot read or delete user B\'s analysis (404, never 403)', async () => {
    const userA = await createTestUser('usera@example.com', 'User A');
    const userB = await createTestUser('userb@example.com', 'User B');

    // User A creates an analysis
    const analysisA = await Analysis.create({
      userId: userA.user._id,
      type: 'risk',
      inputSummary: { BMI: 22 },
      result: { predicted_class: 'Low' },
    });

    // User B tries to GET user A's analysis -> 404
    const getRes = await request(app)
      .get(`/api/analysis/${analysisA._id}`)
      .set('Cookie', userB.cookie);

    expect(getRes.status).toBe(404);
    expect(getRes.body.error).toBe('Analysis not found');

    // User B tries to DELETE user A's analysis -> 404
    const delRes = await request(app)
      .delete(`/api/analysis/${analysisA._id}`)
      .set('Cookie', userB.cookie);

    expect(delRes.status).toBe(404);
    expect(delRes.body.error).toBe('Analysis not found');

    // Verify it still exists in DB
    const check = await Analysis.findById(analysisA._id);
    expect(check).not.toBeNull();
  });

  test('9. list returns only own items with pagination', async () => {
    const userA = await createTestUser('usera_list@example.com', 'User A');
    const userB = await createTestUser('userb_list@example.com', 'User B');

    await Analysis.create({
      userId: userA.user._id,
      type: 'risk',
      result: { class: 'Low' },
    });
    await Analysis.create({
      userId: userA.user._id,
      type: 'explain',
      result: { class: 'Low' },
    });
    await Analysis.create({
      userId: userB.user._id,
      type: 'risk',
      result: { class: 'High' },
    });

    const res = await request(app)
      .get('/api/analysis?page=1&limit=10')
      .set('Cookie', userA.cookie);

    expect(res.status).toBe(200);
    expect(res.body.total).toBe(2);
    expect(res.body.analyses.length).toBe(2);
    expect(res.body.analyses.every((a) => a.userId.toString() === userA.user._id.toString())).toBe(true);
  });

  test('10. unknown feature name -> 400 validation error', async () => {
    const { cookie } = await createTestUser();
    const res = await request(app)
      .post('/api/analysis/risk')
      .set('Cookie', cookie)
      .send({ BMI: 25.0, UnknownFeatureName: 123 });

    expect(res.status).toBe(400);
    expect(res.body.error).toBe('Validation failed');
    expect(JSON.stringify(res.body.details)).toContain('Unknown feature(s) not permitted');
  });

  test('11. delete works for own analysis', async () => {
    const { user, cookie } = await createTestUser('delete_user@example.com');
    const analysis = await Analysis.create({
      userId: user._id,
      type: 'nlp',
      result: { entities: [] },
    });

    const res = await request(app)
      .delete(`/api/analysis/${analysis._id}`)
      .set('Cookie', cookie);

    expect(res.status).toBe(200);
    expect(res.body.message).toMatch(/deleted successfully/i);

    const check = await Analysis.findById(analysis._id);
    expect(check).toBeNull();
  });

  test('12. ML service returns 400 with string error -> gateway forwards 400 with that error string', async () => {
    const { cookie } = await createTestUser();
    simulatedStatus = 400;
    simulatedResponse = { error: 'File is not a valid JPEG or PNG image' };

    const res = await request(app)
      .post('/api/analysis/risk')
      .set('Cookie', cookie)
      .send({ BMI: 25.0 });

    expect(res.status).toBe(400);
    expect(res.body.error).toBe('File is not a valid JPEG or PNG image');
  });

  test('13. image route requires age_years and forwards it to ML service', async () => {
    const { cookie } = await createTestUser();
    const dummyPng = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

    // 1. Missing age_years -> 400
    const resNoAge = await request(app)
      .post('/api/analysis/image')
      .set('Cookie', cookie)
      .attach('file', dummyPng, { filename: 'chest.png', contentType: 'image/png' });

    expect(resNoAge.status).toBe(400);
    expect(resNoAge.body.error).toMatch(/age_years.*required/i);

    // 2. Out-of-range age_years -> 400
    const resBadAge = await request(app)
      .post('/api/analysis/image')
      .set('Cookie', cookie)
      .field('age_years', '150')
      .attach('file', dummyPng, { filename: 'chest.png', contentType: 'image/png' });

    expect(resBadAge.status).toBe(400);
    expect(resBadAge.body.error).toMatch(/age_years.*required/i);

    // 3. Valid age_years -> 200 and forwards age_years to ML service
    simulatedResponse = {
      model_used: 'adult_torchxrayvision_densenet121_all',
      pneumonia_score: 0.1234,
      limitations: ['Screening aid only'],
    };

    const resValid = await request(app)
      .post('/api/analysis/image')
      .set('Cookie', cookie)
      .field('age_years', '45')
      .attach('file', dummyPng, { filename: 'chest.png', contentType: 'image/png' });

    expect(resValid.status).toBe(200);
    expect(resValid.body.analysisId).toBeDefined();
    expect(resValid.body.model_used).toBe('adult_torchxrayvision_densenet121_all');
    expect(lastMLUrl).toBe('/v1/image');
    expect(lastMLBody).toContain('age_years');
    expect(lastMLBody).toContain('45');

    // Verify saved document has type 'image' and age_years recorded
    const saved = await Analysis.findById(resValid.body.analysisId);
    expect(saved).not.toBeNull();
    expect(saved.type).toBe('image');
    expect(saved.inputSummary.age_years).toBe(45);
  });

  // ─────────────────────────────────────────────────────────────
  // Chapter 15a: Honest error message tests
  // ─────────────────────────────────────────────────────────────

  test('14. /api/analysis/risk returns 503 with busy message when ML service returns 503', async () => {
    const { cookie } = await createTestUser();
    simulatedStatus = 503;
    simulatedResponse = { error: 'The analysis service is busy with another analysis. Please retry shortly.' };

    const res = await request(app)
      .post('/api/analysis/risk')
      .set('Cookie', cookie)
      .send({ BMI: 25.0, Age: 9 });

    expect(res.status).toBe(503);
    expect(res.body.error).toMatch(/busy with another analysis/i);
  });

  test('15. /api/analysis/analyze returns 504 with timeout message when ML service returns 504', async () => {
    const { cookie } = await createTestUser();
    simulatedStatus = 504;
    simulatedResponse = { error: 'The analysis took too long, probably because the models are still loading. Please retry in a minute.' };

    const dummyPng = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

    const res = await request(app)
      .post('/api/analysis/analyze')
      .set('Cookie', cookie)
      .field('features', JSON.stringify({ BMI: 25.0, Age: 9, Sex: 1, GenHlth: 3 }));

    expect(res.status).toBe(504);
    expect(res.body.error).toMatch(/took too long/i);
  });

  test('16. /api/analysis/risk returns 504 with timeout message when gateway times out (AbortError)', async () => {
    const { cookie } = await createTestUser();
    // Set delay longer than 500ms timeout set via ML_TIMEOUT_MS
    simulatedDelayMs = 600;
    simulatedStatus = 200;

    const res = await request(app)
      .post('/api/analysis/risk')
      .set('Cookie', cookie)
      .send({ BMI: 25.0, Age: 9 });

    expect(res.status).toBe(504);
    expect(res.body.error).toMatch(/took too long/i);
  });
});

