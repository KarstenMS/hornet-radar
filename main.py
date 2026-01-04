"""
Main pipeline: iterates through images, runs detection,
uploads results and thumbnails to Supabase.
"""
import argparse
import os
import cv2
from detection import load_model, run_detection
from storage import get_last_detection_id
from helpers import ensure_directories
from pipeline_utils import save_and_upload_detection_frame
from config import *


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--videos', default=False, action='store_true',  # Später ändern auf False und Kamera auf True
                    help="Analyses any .mp4 from detections/frames")
parser.add_argument('-i', '--images', default=False, action='store_true',
                    help="Analyses any .jpg from detections/frames")
args = parser.parse_args()


def image_recognition(frames_dir, model, start_detection_id):
    detection_id = start_detection_id

    for image_name in os.listdir(frames_dir):
        if not image_name.lower().endswith(".jpg"):
            continue


        image_path = os.path.join(frames_dir, image_name)
        print(f"Processing {image_name} ...")

        img = cv2.imread(image_path)
        if img is None:
            continue

        # --- YOLO detection ---
        predictions = run_detection(img, model)
        if not predictions:
            continue
        hornets = [p for p in predictions if p["class_id"] in (0, 1)]

        # --- Skip if no hornets detected ---
        if not (hornets):
            print(f"No hornets detected in {image_name}, skipping upload.")
            continue
        
        print(f'Positive hornet detections in {image_name}')
        detection_id += 1
        save_and_upload_detection_frame(img, hornets, detection_id)




def video_tracking(videos_dir, model, start_detection_id):
    detection_id = start_detection_id

    for video_name in os.listdir(videos_dir):

        image_name = ""
        if not video_name.lower().endswith(".mp4"):
            continue

        video_path = os.path.join(videos_dir, video_name)
        print(f"Processing video {video_name} ...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open {video_name}.")
            continue

        found = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Could not read {video_name}.") 
                continue

            predictions = run_detection(frame, model)
            if not predictions:
                continue           

            hornets = [p for p in predictions if p["class_id"] in (0, 1)]
            if not hornets:
                continue      

            detection_id += 1
            found = True
            save_and_upload_detection_frame(frame, hornets, detection_id)
            
           
            break  # stop after first detection

        cap.release()

        if not found:
            print(f"No hornets detected in {video_name}.")

       
def camera_tracking(model, start_detection_id):

    from camera import Camera

    cam = Camera()

    while True:
        frame = cam.read()
        if frame is None:
            print("Cannot read camera")
            break

        # OpenCV processing
        cv2.imshow("Debug", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


       
def main():
    ensure_directories(FRAMES_DIR, LABELED_FRAMES_DIR, LABELED_FRAMES_THUMBS_DIR, VIDEOS_DIR, LABELED_VIDEOS_DIR, LABELED_VIDEOS_THUMBS_DIR)
    model = load_model()
    
    start_detection_id = get_last_detection_id(PI_ID) #Continue with latest detection_id from Supabase
    print(f"Starting detection_id at {start_detection_id}")

    if args.images:
        print(f"Reading {FRAMES_DIR} directory.") 
        image_recognition(FRAMES_DIR, model, start_detection_id)
    elif args.videos:
        print(f"Reading {VIDEOS_DIR} directory.") 
        video_tracking(VIDEOS_DIR, model, start_detection_id)
    else:
        print(f"Capturing from camera with {CAMERA_FPS} FPS") 
        video_tracking(model, start_detection_id)


        

if __name__ == "__main__":
    main()
