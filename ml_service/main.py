"""
MediTwin ML Service — FastAPI Entrypoint (Chapter 12)
"""
import time
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ml_service.config import ML_DEBUG, HOST, PORT
from ml_service.routes.api import router as api_router

# Setup structured clean logging without sensitive patient or input payload data
class RequestIdFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return super().format(record)

logger = logging.getLogger("meditwin.ml_service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(RequestIdFormatter("%(asctime)s [%(levelname)s] [ReqID: %(request_id)s] %(message)s"))
logger.handlers = [handler]
logger.propagate = False

# Ensure orchestrator logs ([START], [END], [FAIL]) print to console and propagate for tests
orch_logger = logging.getLogger("meditwin.orchestrator")
orch_logger.setLevel(logging.INFO)
orch_logger.handlers = [handler]
orch_logger.propagate = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background warm-up on startup; nothing needed on shutdown."""
    from ml_service.inference import start_warmup_task
    start_warmup_task()
    logger.info("[STARTUP] Background model warm-up task scheduled")
    yield


app = FastAPI(
    title="MediTwin ML Service",
    description="Internal machine learning microservice serving MediTwin multi-agent diagnostic & screening models.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if ML_DEBUG else None,
    redoc_url=None,
    openapi_url="/openapi.json" if ML_DEBUG else None,
)


@app.middleware("http")
async def request_timing_and_sanitization_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    t0 = time.time()

    # Route request
    try:
        response = await call_next(request)
        duration_ms = round((time.time() - t0) * 1000, 2)
        # Log route, status, duration only — NEVER body or patient data
        extra = {"request_id": request_id}
        logger.info(
            f"METHOD={request.method} PATH={request.url.path} STATUS={response.status_code} DURATION={duration_ms}ms",
            extra=extra
        )
        response.headers["X-Request-Id"] = request_id
        return response
    except Exception as exc:
        duration_ms = round((time.time() - t0) * 1000, 2)
        extra = {"request_id": request_id}
        logger.error(
            f"METHOD={request.method} PATH={request.url.path} ERROR={type(exc).__name__}: {str(exc)} DURATION={duration_ms}ms",
            extra=extra,
            exc_info=True
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal ML Service error occurred",
                "request_id": request_id
            },
            headers={"X-Request-Id": request_id}
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = getattr(request.state, "request_id", "-")
    # Clean error representation
    errors = []
    for err in exc.errors():
        field = ".".join(str(loc) for loc in err.get("loc", []))
        errors.append(f"{field}: {err.get('msg')}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error: " + "; ".join(errors),
            "details": exc.errors(),
            "request_id": req_id
        },
        headers={"X-Request-Id": req_id}
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    req_id = getattr(request.state, "request_id", "-")
    content = exc.detail if isinstance(exc.detail, dict) else {"error": exc.detail}
    content["request_id"] = req_id
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers={"X-Request-Id": req_id}
    )


# Register API routes
app.include_router(api_router)
