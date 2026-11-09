const request = require('supertest');
const { MongoMemoryServer } = require('mongodb-memory-server');
const mongoose = require('mongoose');

process.env.NODE_ENV = 'test';
process.env.JWT_ACCESS_SECRET = 'super-secret-access-token-key-32chars';
process.env.JWT_REFRESH_SECRET = 'super-secret-refresh-token-key-32chars';

jest.setTimeout(60000);

const app = require('../src/app');
const User = require('../src/models/User');
const { connectDB, disconnectDB } = require('../src/config/db');

let mongoServer;

beforeAll(async () => {
  mongoServer = await MongoMemoryServer.create({
    instance: { launchTimeout: 60000 },
  });
  const uri = mongoServer.getUri();
  await connectDB(uri);
}, 60000);

afterAll(async () => {
  await disconnectDB();
  if (mongoServer) {
    await mongoServer.stop();
  }
}, 60000);

beforeEach(async () => {
  await User.deleteMany({});
});

describe('Chapter 9: Backend Auth Integration Tests', () => {
  const validUser = {
    name: 'Jane Doe',
    email: 'jane.doe@example.com',
    password: 'SecurePassword1',
  };

  test('1. signup success: creates user, hashes password, returns cookies and omits passwordHash', async () => {
    const res = await request(app)
      .post('/api/auth/signup')
      .send(validUser);

    expect(res.status).toBe(201);
    expect(res.body.user).toBeDefined();
    expect(res.body.user.email).toBe('jane.doe@example.com');
    expect(res.body.user.name).toBe('Jane Doe');
    expect(res.body.user.role).toBe('patient');
    expect(res.body.user.passwordHash).toBeUndefined();
    expect(res.body.user.refreshTokenHash).toBeUndefined();

    // Check cookies
    const cookies = res.headers['set-cookie'];
    expect(cookies).toBeDefined();
    expect(cookies.some((c) => c.startsWith('accessToken='))).toBe(true);
    expect(cookies.some((c) => c.startsWith('refreshToken='))).toBe(true);

    // Verify DB user
    const dbUser = await User.findOne({ email: 'jane.doe@example.com' });
    expect(dbUser).not.toBeNull();
    expect(dbUser.passwordHash).toBeDefined();
    expect(dbUser.passwordHash).not.toBe(validUser.password);
  });

  test('2. duplicate email: rejects already registered email address', async () => {
    // First signup
    await request(app).post('/api/auth/signup').send(validUser);

    // Second signup with identical email
    const res = await request(app).post('/api/auth/signup').send(validUser);

    expect(res.status).toBe(409);
    expect(res.body.error).toMatch(/already registered/i);
  });

  test('3. weak password: rejects password with <8 chars or missing letter/number', async () => {
    // Too short
    const shortRes = await request(app)
      .post('/api/auth/signup')
      .send({ name: 'Short Pwd', email: 'short@example.com', password: 'Aa1' });
    expect(shortRes.status).toBe(400);

    // No numbers
    const noNumRes = await request(app)
      .post('/api/auth/signup')
      .send({ name: 'No Num', email: 'nonum@example.com', password: 'OnlyLettersHere' });
    expect(noNumRes.status).toBe(400);

    // No letters
    const noLettersRes = await request(app)
      .post('/api/auth/signup')
      .send({ name: 'No Letters', email: 'nolet@example.com', password: '1234567890' });
    expect(noLettersRes.status).toBe(400);
  });

  test('4. login success: authenticates valid credentials and sets tokens', async () => {
    await request(app).post('/api/auth/signup').send(validUser);

    const res = await request(app)
      .post('/api/auth/login')
      .send({ email: validUser.email, password: validUser.password });

    expect(res.status).toBe(200);
    expect(res.body.user.email).toBe(validUser.email);
    expect(res.body.user.passwordHash).toBeUndefined();

    const cookies = res.headers['set-cookie'];
    expect(cookies.some((c) => c.startsWith('accessToken='))).toBe(true);
    expect(cookies.some((c) => c.startsWith('refreshToken='))).toBe(true);
  });

  test('5. wrong password: returns generic 401 error without revealing user existence', async () => {
    await request(app).post('/api/auth/signup').send(validUser);

    // Wrong password
    const wrongPassRes = await request(app)
      .post('/api/auth/login')
      .send({ email: validUser.email, password: 'WrongPassword99' });

    expect(wrongPassRes.status).toBe(401);
    expect(wrongPassRes.body.error).toBe('Invalid email or password');

    // Non-existent email
    const wrongEmailRes = await request(app)
      .post('/api/auth/login')
      .send({ email: 'nonexistent@example.com', password: 'SomePassword1' });

    expect(wrongEmailRes.status).toBe(401);
    expect(wrongEmailRes.body.error).toBe('Invalid email or password');
  });

  test('6. /me with and without cookie: protected profile access', async () => {
    // Without cookie
    const unauthRes = await request(app).get('/api/auth/me');
    expect(unauthRes.status).toBe(401);

    // With cookie
    const loginRes = await request(app).post('/api/auth/signup').send(validUser);
    const cookies = loginRes.headers['set-cookie'];
    const accessCookie = cookies.find((c) => c.startsWith('accessToken='));

    const authRes = await request(app)
      .get('/api/auth/me')
      .set('Cookie', [accessCookie]);

    expect(authRes.status).toBe(200);
    expect(authRes.body.user.email).toBe(validUser.email);
    expect(authRes.body.user.name).toBe(validUser.name);
  });

  test('7. refresh rotation: rotates refresh token and invalidates reused old token', async () => {
    const signupRes = await request(app).post('/api/auth/signup').send(validUser);
    const initialCookies = signupRes.headers['set-cookie'];
    const oldRefreshCookie = initialCookies.find((c) => c.startsWith('refreshToken='));
    const oldRefreshToken = oldRefreshCookie.split(';')[0].split('=')[1];

    // Wait slightly or call refresh
    const refreshRes = await request(app)
      .post('/api/auth/refresh')
      .set('Cookie', [oldRefreshCookie]);

    expect(refreshRes.status).toBe(200);
    const newCookies = refreshRes.headers['set-cookie'];
    const newRefreshCookie = newCookies.find((c) => c.startsWith('refreshToken='));
    const newRefreshToken = newRefreshCookie.split(';')[0].split('=')[1];

    // Ensure token was rotated
    expect(newRefreshToken).not.toBe(oldRefreshToken);

    // Old refresh token must be rejected upon reuse
    const reuseRes = await request(app)
      .post('/api/auth/refresh')
      .set('Cookie', [oldRefreshCookie]);

    expect([401, 403]).toContain(reuseRes.status);
  });

  test('8. logout clears cookies: invalidates session and clears cookie headers', async () => {
    const signupRes = await request(app).post('/api/auth/signup').send(validUser);
    const cookies = signupRes.headers['set-cookie'];
    const refreshCookie = cookies.find((c) => c.startsWith('refreshToken='));

    const logoutRes = await request(app)
      .post('/api/auth/logout')
      .set('Cookie', [refreshCookie]);

    expect(logoutRes.status).toBe(200);
    expect(logoutRes.body.message).toMatch(/logged out/i);

    const logoutCookies = logoutRes.headers['set-cookie'];
    expect(logoutCookies).toBeDefined();

    // Check that cookies are cleared (either empty value or past expiration)
    const accessCleared = logoutCookies.some(
      (c) => c.includes('accessToken=;') || c.includes('accessToken=') && (c.includes('Expires=Thu, 01 Jan 1970') || c.includes('Max-Age=0'))
    );
    const refreshCleared = logoutCookies.some(
      (c) => c.includes('refreshToken=;') || c.includes('refreshToken=') && (c.includes('Expires=Thu, 01 Jan 1970') || c.includes('Max-Age=0'))
    );
    expect(accessCleared).toBe(true);
    expect(refreshCleared).toBe(true);
  });

  test('9. health check: GET /api/health returns ok status', async () => {
    const res = await request(app).get('/api/health');
    expect(res.status).toBe(200);
    expect(res.body.status).toBe('ok');
    expect(res.body.timestamp).toBeDefined();
  });
});

