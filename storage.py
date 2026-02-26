"""Hornet Radar: Supabase upload helpers for images and JSON payloads."""
import logging
import requests
from pathlib import Path
from typing import Any, Dict, Optional
from config import SUPABASE_URL, SUPABASE_KEY, BUCKET_NAME, TABLE_NAME

logger = logging.getLogger(__name__)   

def _auth_headers(content_type: str) -> Dict[str, str]:
    """Build Supabase authentication headers."""
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": content_type,
    }

def upload_image_to_supabase(image_path: str, image_name: str, *, is_thumb: bool = False) -> Optional[str]:
    """Upload an image (or thumbnail) to Supabase Storage.

    Args:
        image_path: Local image path.
        image_name: Object name in the bucket.
        is_thumb: If True, the image is stored under the "thumbnails" prefix.

    Returns:
        Public URL on success, otherwise None.

    Notes:
        Supabase Storage typically expects PUT for uploads. This function tries PUT first
        and falls back to POST for compatibility with older setups.
    """
    folder = "thumbnails" if is_thumb else ""
    # Avoid double slashes when folder is empty
    key = f"{folder}/{image_name}" if folder else image_name
    upload_url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{key}"

    path = Path(image_path)
    data = path.read_bytes()
    headers = _auth_headers("image/jpeg")

    # Prefer PUT (recommended), fall back to POST
    response = requests.put(upload_url, headers=headers, data=data)
    if response.status_code not in (200, 201):
        response = requests.post(upload_url, headers=headers, data=data)

    if response.status_code in (200, 201):
        public_url = upload_url.replace("/object/", "/object/public/")
        logger.info("Uploaded %s", image_name)
        return public_url

    logger.warning("Upload failed for %s: %s %s", image_name, response.status_code, response.text)
    return None

def upload_json_to_supabase(data: Dict[str, Any]) -> bool:
    """Upload one JSON record to the Supabase REST API table.

    Args:
        data: A JSON-serializable dict.

    Returns:
        True if Supabase accepted the record (HTTP 201), else False.
    """
    json_url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    headers = _auth_headers("application/json")

    response = requests.post(json_url, headers=headers, json=data)
    if response.status_code == 201:
        logger.info("Uploaded JSON record")
        return True
    logger.warning("JSON upload failed: %s %s", response.status_code, response.text)
    return False