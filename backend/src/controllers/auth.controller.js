const crypto = require('crypto');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const {
  generateAccessToken,
  generateRefreshToken,
  getCookieOptions,
  ACCESS_COOKIE_MAX_AGE,
  REFRESH_COOKIE_MAX_AGE,
} = require('../utils/tokens');

const BCRYPT_SALT_ROUNDS = 12;

function hashToken(token) {
  // Pre-hash with SHA-256 to ensure input fits comfortably within bcrypt's 72-byte limit
  return crypto.createHash('sha256').update(token).digest('hex');
}

async function signup(req, res, next) {
  try {
    const { name, email, password, role } = req.body;

    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(409).json({ error: 'Email is already registered' });
    }

    const passwordHash = await bcrypt.hash(password, BCRYPT_SALT_ROUNDS);

    const user = new User({
      name,
      email,
      passwordHash,
      role: role || 'patient',
    });

    const accessToken = generateAccessToken({ id: user._id, role: user.role });
    const refreshToken = generateRefreshToken({ id: user._id });

    user.refreshTokenHash = await bcrypt.hash(hashToken(refreshToken), 10);
    await user.save();

    res.cookie('accessToken', accessToken, getCookieOptions(ACCESS_COOKIE_MAX_AGE));
    res.cookie('refreshToken', refreshToken, getCookieOptions(REFRESH_COOKIE_MAX_AGE));

    return res.status(201).json({
      message: 'Account created successfully',
      user,
    });
  } catch (err) {
    next(err);
  }
}

async function login(req, res, next) {
  try {
    const { email, password } = req.body;

    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ error: 'Invalid email or password' });
    }

    const isMatch = await bcrypt.compare(password, user.passwordHash);
    if (!isMatch) {
      return res.status(401).json({ error: 'Invalid email or password' });
    }

    const accessToken = generateAccessToken({ id: user._id, role: user.role });
    const refreshToken = generateRefreshToken({ id: user._id });

    user.refreshTokenHash = await bcrypt.hash(hashToken(refreshToken), 10);
    await user.save();

    res.cookie('accessToken', accessToken, getCookieOptions(ACCESS_COOKIE_MAX_AGE));
    res.cookie('refreshToken', refreshToken, getCookieOptions(REFRESH_COOKIE_MAX_AGE));

    return res.status(200).json({
      message: 'Login successful',
      user,
    });
  } catch (err) {
    next(err);
  }
}

async function logout(req, res, next) {
  try {
    const token = req.cookies?.refreshToken;
    if (token) {
      try {
        const decoded = jwt.verify(token, process.env.JWT_REFRESH_SECRET);
        await User.findByIdAndUpdate(decoded.id, { refreshTokenHash: null });
      } catch (_) {
        // If token is invalid or expired, proceed to clear cookies anyway
      }
    }

    const clearOpts = {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
    };

    res.clearCookie('accessToken', clearOpts);
    res.clearCookie('refreshToken', clearOpts);

    return res.status(200).json({ message: 'Logged out successfully' });
  } catch (err) {
    next(err);
  }
}

async function refresh(req, res, next) {
  try {
    const token = req.cookies?.refreshToken;
    if (!token) {
      return res.status(401).json({ error: 'Refresh token required' });
    }

    let decoded;
    try {
      decoded = jwt.verify(token, process.env.JWT_REFRESH_SECRET);
    } catch (err) {
      return res.status(401).json({ error: 'Invalid or expired refresh token' });
    }

    const user = await User.findById(decoded.id);
    if (!user || !user.refreshTokenHash) {
      return res.status(401).json({ error: 'Invalid session' });
    }

    const isMatch = await bcrypt.compare(hashToken(token), user.refreshTokenHash);
    if (!isMatch) {
      // Possible token theft or replay: invalidate stored refresh token
      user.refreshTokenHash = null;
      await user.save();
      return res.status(403).json({ error: 'Invalid refresh token - session terminated' });
    }

    // Token rotation: issue new pair
    const newAccessToken = generateAccessToken({ id: user._id, role: user.role });
    const newRefreshToken = generateRefreshToken({ id: user._id });

    user.refreshTokenHash = await bcrypt.hash(hashToken(newRefreshToken), 10);
    await user.save();

    res.cookie('accessToken', newAccessToken, getCookieOptions(ACCESS_COOKIE_MAX_AGE));
    res.cookie('refreshToken', newRefreshToken, getCookieOptions(REFRESH_COOKIE_MAX_AGE));

    return res.status(200).json({
      message: 'Token refreshed successfully',
      user,
    });
  } catch (err) {
    next(err);
  }
}

async function me(req, res) {
  return res.status(200).json({ user: req.user });
}

module.exports = {
  signup,
  login,
  logout,
  refresh,
  me,
};

