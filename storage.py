"""
Handles Supabase uploads for images and JSON.
"""

import os
import requests
from config import SUPABASE_URL, SUPABASE_KEY, BUCKET_NAME, TABLE_NAME

def upload_image_to_supabase(image_path, image_name, is_thumb=False):
    """Upload an image or thumbnail to Supabase Storage."""
    folder = "thumbnails" if is_thumb else ""
    upload_url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{folder}/{image_name}"

    with open(image_path, "rb") as f:
        image_data = f.read()

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "image/jpeg"
    }

    response = requests.post(upload_url, headers=headers, data=image_data)

    if response.status_code in [200, 201]:
        public_url = upload_url.replace("/object/", "/object/public/")
        print(f"Successfully uploaded {image_name}")
        return public_url
    else:
        print(f"Upload failed for {image_name}: {response.status_code}")
        return None


def upload_json_to_supabase(data):
    """Upload a single JSON record to Supabase REST API."""
    json_url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(json_url, headers=headers, json=data)
    if response.status_code == 201:
        print("Successfully uploaded detection!")
        return response.status_code == 201
    else:
        print("Upload detection failed:", response.status_code, response.text)

def get_last_detection_id(pi_id):
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    params = {
        "select": "detection_id",
        "pi_id": f"eq.{pi_id}",
        "order": "detection_id.desc",
        "limit": 1
    }

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=3)
        if r.status_code == 200 and r.json():
            return int(r.json()[0]["detection_id"])
    except Exception:
        pass

    return 0
