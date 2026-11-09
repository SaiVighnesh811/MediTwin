# MediTwin AI - Backend Authentication Service

Node.js + Express + Mongoose backend authentication service with JWT access & refresh token rotation, bcrypt password hashing, and cookie-based session management.

## Environment Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Fill in the required variables in `.env`:
   - `PORT`: Server port (default: `5000`)
   - `NODE_ENV`: Environment (`development` or `production`)
   - `CLIENT_ORIGIN`: Allowed client CORS origin (e.g. `http://localhost:5173`)
   - `MONGODB_URI`: MongoDB connection string (e.g. `mongodb+srv://...`)
   - `JWT_ACCESS_SECRET`: Secret key for 15-minute access tokens
   - `JWT_REFRESH_SECRET`: Secret key for 7-day refresh tokens

> **Note**: If `.env` is missing or any required variable is empty, the server automatically halts at startup with a message identifying the missing configuration.

## Available Scripts

```bash
# Install dependencies
npm install

# Start in development mode (with nodemon auto-restart)
npm run dev

# Start in production mode
npm start

# Run integration tests (with mongodb-memory-server)
npm test
```

## API Endpoints

| Method | Endpoint | Description | Auth Required | Rate Limited |
|--------|----------|-------------|:-------------:|:------------:|
| `POST` | `/api/auth/signup` | Register new account (name, email, password min 8 chars with letter & number) | No | Yes (10/15min) |
| `POST` | `/api/auth/login` | Log in with email & password, sets httpOnly cookies | No | Yes (10/15min) |
| `POST` | `/api/auth/logout` | Clear auth cookies and invalidate refresh token | No | No |
| `POST` | `/api/auth/refresh` | Rotate refresh token and issue fresh access token | Cookie | No |
| `GET` | `/api/auth/me` | Fetch authenticated user profile | Cookie/Bearer | No |
| `GET` | `/api/health` | Health check endpoint | No | No |
| `POST` | `/api/analysis/risk` | Forward BRFSS survey features to ML Service for metabolic risk & save analysis | Cookie/Bearer | Yes (20/10min) |
| `POST` | `/api/analysis/explain` | Forward BRFSS features for SHAP + LIME explainability & save analysis | Cookie/Bearer | Yes (20/10min) |
| `POST` | `/api/analysis/nlp` | Forward clinical text for biomedical NER & save analysis | Cookie/Bearer | Yes (20/10min) |
| `POST` | `/api/analysis/image` | Forward chest X-ray image for pneumonia screening & save analysis. **Requires `age_years` field (0-120)** to route to the correct model: age < 18 → pediatric Kermany ResNet18; age ≥ 18 → adult TorchXRayVision DenseNet121-res224-all. Returns 400 if `age_years` is missing or out of range. | Cookie/Bearer | Yes (20/10min) |
| `POST` | `/api/analysis/ocr` | Forward prescription/lab document image for OCR & save analysis | Cookie/Bearer | Yes (20/10min) |
| `POST` | `/api/analysis/analyze` | Multi-modal orchestrator case analysis with partial execution fallback | Cookie/Bearer | Yes (20/10min) |
| `GET` | `/api/analysis` | List authenticated user's saved analyses (newest first, paginated) | Cookie/Bearer | No |
| `GET` | `/api/analysis/:id` | Get specific saved analysis (returns 404 if not owned by user) | Cookie/Bearer | No |
| `DELETE` | `/api/analysis/:id` | Delete specific saved analysis (returns 404 if not owned by user) | Cookie/Bearer | No |

## Security Features

- **Password Hashing**: bcrypt with salt rounds cost 12.
- **Token Strategy**: 15-minute access token + 7-day refresh token stored in `httpOnly`, `sameSite=lax` cookies.
- **Refresh Token Rotation**: Each refresh issues a new token pair and revokes the previous token; token reuse detection invalidates the session.
- **Rate Limiting**: 10 requests per 15 minutes per IP on auth endpoints.
- **Sanitized Responses**: `passwordHash` and `refreshTokenHash` are never returned in responses.
- **Generic Auth Errors**: Uniform "Invalid email or password" error prevents user enumeration.
- **CORS & Headers**: Strict CORS origin verification and Helmet security headers.

