"""
Security & upload validation utilities for MediTwin ML Service.
"""
import os
import uuid
import secrets
import tempfile
from typing import Tuple
from fastapi import Header, HTTPException, UploadFile, status
from PIL import Image

from .config import INTERNAL_API_KEY

MAX_UPLOAD_SIZE = 8 * 1024 * 1024  # 8 MB
MAX_PIXELS = 25_000_000  # reject decompression bombs
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG"}


def verify_api_key(x_internal_key: str = Header(None, alias="X-Internal-Key")) -> str:
    """
    Validates X-Internal-Key header using constant-time comparison.
    Raises HTTP 401 if missing or invalid.
    """
    if not x_internal_key or not INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Unauthorized: Invalid or missing API key"}
        )
    if not secrets.compare_digest(x_internal_key, INTERNAL_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Unauthorized: Invalid API key"}
        )
    return x_internal_key


async def validate_and_save_upload(file: UploadFile) -> Tuple[str, str]:
    """
    Strictly validates uploaded image:
    1. Checks file size (max 8MB).
    2. Checks MIME type header.
    3. Opens image with PIL to verify valid JPEG/PNG integrity.
    4. Writes into a randomized temporary file.
    
    Returns (temp_file_path, original_format).
    Caller is responsible for deleting the temp file in a finally block.
    """
    # 1. Read content into memory up to max limit + 1 byte
    content = await file.read(MAX_UPLOAD_SIZE + 1)
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "File size exceeds 8 MB limit"}
        )
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Empty file provided"}
        )

    # 2. Check content-type header if present
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"Invalid content type: {content_type}. Only JPEG and PNG are supported."}
        )

    # 3. Save to a temporary file with a random uuid
    temp_dir = tempfile.gettempdir()
    random_filename = f"meditwin_up_{uuid.uuid4().hex}.tmp"
    temp_path = os.path.join(temp_dir, random_filename)

    with open(temp_path, "wb") as f:
        f.write(content)

    # 4. Open with PIL to verify integrity and real format
    try:
        with Image.open(temp_path) as img:
            img_format = img.format
            if img.width * img.height > MAX_PIXELS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "Image dimensions too large (max 25 megapixels)"}
                )
            img.verify()
        if img_format not in ALLOWED_PIL_FORMATS:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": f"Invalid image format: {img_format}. Only JPEG and PNG are allowed."}
            )
    except HTTPException:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "File is not a valid JPEG or PNG image"}
        )

    return temp_path, img_format

