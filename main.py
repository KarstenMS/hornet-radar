"""
Main pipeline: iterates through images, runs detection,
uploads results and thumbnails to Supabase.
"""
import argparse
import os
import cv2
from detection import load_model, run_detection, count_species
from storage import upload_image_to_supabase, upload_json_to_supabase
from helpers import create_thumbnail, timestamp, ensure_directories
from config import *


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--videos', default=True, action='store_true',  # Später ändern auf False und Kamera auf True
                    help="Analyses any .mp4 from detections/frames")
parser.add_argument('-i', '--images', default=False, action='store_true',
                    help="Analyses any .jpg from detections/frames")
args = parser.parse_args()


def image_recognition(frames_dir, model):
    frame_id = 0
    for image_name in os.listdir(frames_dir):
        if not image_name.endswith(".jpg"):
            continue

        frame_id += 1
        image_path = os.path.join(frames_dir, image_name)
        print(f"Processing {image_name} ...")

        frame = cv2.imread(image_path)
        if frame is None:
            continue

        # --- YOLO detection ---
        predictions, render_img = run_detection(frame, model)
        ah_count, eh_count = count_species(predictions)

        # --- Skip if no hornets detected ---
        if not (ah_count or eh_count):
            print(f"No hornets detected in {image_name}, skipping upload.")
            continue
        
        print(f'Positive hornet detections in {image_name}')

        image_name = f"{PI_ID}_Frame_{frame_id}.jpg"
        thumb_name = f"{PI_ID}_Frame_{frame_id}_thumb.jpg"
        local_image_path = os.path.join(LABELED_FRAMES_DIR, image_name)
        local_thumb_path = os.path.join(LABELED_FRAMES_THUMBS_DIR, thumb_name)
        cv2.imwrite(local_image_path, render_img)
        create_thumbnail(local_image_path, local_thumb_path)

        # Upload images
        image_url = upload_image_to_supabase(local_image_path, image_name)
        thumb_url = upload_image_to_supabase(local_thumb_path, thumb_name, is_thumb=True)

        data = {
            "pi_id": PI_ID,
            "frame_id": f"{PI_ID}_Frame_{frame_id}",
            "timestamp": timestamp(),
            "species": "asian_hornet" if ah_count > 0 else "european_hornet",
            "approach_angle": None,
            "departure_angle": None,
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "image_url": image_url,
            "thumb_url": thumb_url
        }

        if upload_json_to_supabase(data):
            print("Detection record uploaded.")
        else:
            print("JSON upload failed.")            


def video_tracking(videos_dir, model):
    for video_name in os.listdir(videos_dir):
        frame_id = 0
        image_name = ""
        if not video_name.endswith(".mp4"):
            continue

        tracker = None
        trajectory = []
        tracking_active = False
        frame_id = 0
        video_path = os.path.join(videos_dir, video_name)

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read {video_name}.") 
            continue

        predictions = run_detection(frame, model)
        ah_count, eh_count = count_species(predictions)

        if not (ah_count or eh_count):
            print(f"No hornets detected in {video_name}, skipping upload.") 
            continue

        # Create images and Thumbnails of first detection for upload
        if not image_name:
            image_name = f"{PI_ID}_Frame_{frame_id}.jpg"
            thumb_name = f"{PI_ID}_Frame_{frame_id}_thumb.jpg"
            local_image_path = os.path.join(LABELED_FRAMES_DIR, image_name)
            local_thumb_path = os.path.join(LABELED_FRAMES_THUMBS_DIR, thumb_name)
            cv2.imwrite(local_image_path, frame)
            create_thumbnail(local_image_path, local_thumb_path)
            

        # extract Bounding Box from predictions
        x1, y1, x2, y2 = predictions[0]["bbox"]
        bbox = (x1, y1, x2-x1, y2-y1)

        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        tracking_active = True

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            success, bbox = tracker.update(frame)

            if success:
                x, y, w, h = map(int, bbox)
                cx = x + w // 2
                cy = y + h // 2
                trajectory.append((frame_id, cx, cy))
            else:
                break  # Hornet lost

       
        # Upload images
        image_url = upload_image_to_supabase(local_image_path, image_name)
        thumb_url = upload_image_to_supabase(local_thumb_path, thumb_name, is_thumb=True)       

        data = {
            "pi_id": PI_ID,
            "frame_id": f"{PI_ID}_Frame_{frame_id}",
            "timestamp": timestamp(),
            "species": "asian_hornet" if ah_count > 0 else "european_hornet",
            "approach_angle": trajectory[0],
            "departure_angle": trajectory[-1],
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "image_url": image_url,
            "thumb_url": thumb_url
        }

        upload_json_to_supabase(data)


def main():
    ensure_directories(FRAMES_DIR, LABELED_FRAMES_DIR, LABELED_FRAMES_THUMBS_DIR, VIDEOS_DIR, LABELED_VIDEOS_DIR, LABELED_VIDEOS_THUMBS_DIR)
    model = load_model()
    

    if args.images:
        print(f"Reading {FRAMES_DIR} directory.") 
        image_recognition(FRAMES_DIR, model)
    elif args.videos:
        print(f"Reading {VIDEOS_DIR} directory.") 
        video_tracking(VIDEOS_DIR, model)

        

if __name__ == "__main__":
    main()
