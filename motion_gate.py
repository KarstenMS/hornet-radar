import cv2
import time
from camera import Camera
from config import *
from detection import load_model, run_detection

# =====================
# Settings
# =====================

tracker = None
tracking_active = False
last_motion_time = 0


# =====================
# Functions
# =====================

def create_tracker():
    t = TRACKER_TYPE.upper()

    if t == "KCF" and hasattr(cv2, "TrackerKCF_create"):
        print("Tracker: KCF")
        return cv2.TrackerKCF_create()

    if t == "CSRT" and hasattr(cv2, "TrackerCSRT_create"):
        print("Tracker: CSRT")
        return cv2.TrackerCSRT_create()

    if t == "MOSSE" and hasattr(cv2, "TrackerMOSSE_create"):
        print("Tracker: MOSSE")
        return cv2.TrackerMOSSE_create()

    if t == "AUTO":
        if hasattr(cv2, "TrackerKCF_create"):
            print("Tracker: AUTO → KCF")
            return cv2.TrackerKCF_create()
        if hasattr(cv2, "TrackerCSRT_create"):
            print("Tracker: AUTO → CSRT")
            return cv2.TrackerCSRT_create()

    raise RuntimeError(
        f"Tracker '{TRACKER_TYPE}' not available. "
        "Install opencv-contrib-python."
    )

def start_tracker(frame, bbox):
    global tracker, tracking_active, last_motion_time
    global tracking_frames, detection_done

    tracker = create_tracker()
    tracker.init(frame, bbox)

    last_motion_time = time.time()

    tracking_frames = 0
    detection_done = False
    tracking_active = True

def update_tracker(frame):
    global tracking_active, last_motion_time
    
    if not tracking_active or tracker is None:
        return None

    success, bbox = tracker.update(frame)

    if success:
        last_motion_time = time.time()
        return bbox
    else:
        reset_tracking()
        return None
    
def reset_tracking():
    global tracker, tracking_active
    global tracking_frames, detection_done

    tracker = None
    tracking_active = False
    tracking_frames = 0
    detection_done = False

    print("Tracking reset")

def check_tracker_timeout():
    global tracking_active

    if tracking_active and time.time() - last_motion_time > TRACKER_TIMEOUT:
        reset_tracking()

def draw_tracking(frame, bbox):
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv2.putText(frame, "TRACKING",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)

def extract_roi(frame, bbox):
    x, y, w, h = map(int, bbox)

    h_frame, w_frame = frame.shape[:2]

    x = max(0, x)
    y = max(0, y)
    w = min(w, w_frame - x)
    h = min(h, h_frame - y)

    roi = frame[y:y+h, x:x+w]
    return roi, (x, y)

def run_yolo_on_roi(frame, bbox):
    roi, offset = extract_roi(frame, bbox)

    if roi.size == 0:
        return []

    predictions = run_detection(roi, yolo_model)

    detections = []
    for p in predictions:
        x1, y1, x2, y2 = p["bbox"]

        # ROI → Full frame Koordinaten
        detections.append({
            "class_id": p["class_id"],
            "confidence": p["confidence"],
            "bbox": [
                x1 + offset[0],
                y1 + offset[1],
                x2 + offset[0],
                y2 + offset[1],
            ]
        })

    return detections

# =====================
# Init
# =====================
cam = Camera()

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=MOTION_HISTORY,
    varThreshold=MOTION_VAR_THRESHOLD,
    detectShadows=False
)

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (MOTION_KERNEL_SIZE, MOTION_KERNEL_SIZE)
)


frame_count = 0
last_time = time.time()
fps = 0.0

yolo_model = load_model()

print("Motion-Gate gestartet – ESC zum Beenden")

# =====================
# Main Loop
# =====================

while True:
    frame = cam.picam2.capture_array()
    if frame is None:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    display = frame.copy()

    # Update FPS
    now = time.time()
    dt = now - last_time
    if dt > 0:
        fps = 1.0 / dt
    last_time = now

    frame_count += 1
    process_this_frame = (frame_count % FRAME_SKIP == 0)
    if process_this_frame:
    
        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        motion_boxes = []

        for c in contours:
            if cv2.contourArea(c) < MOTION_MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(c)
            motion_boxes.append((x, y, w, h))
            motion_detected = True
            cv2.rectangle(display, (x,y), (x+w,y+h), (0,255,255), 2)

        if motion_detected and not tracking_active:
            x, y, w, h = max(motion_boxes, key=lambda b: b[2]*b[3])
            start_tracker(frame, (x, y, w, h))

        bbox = update_tracker(frame)

        if bbox is not None:
            tracking_frames += 1

            # Visualize Tracking
            draw_tracking(display, bbox)

            # === STABIL ===
            if tracking_frames >= TRACKING_STABLE_FRAMES and not detection_done:
                print("Stable tracking - running YOLO on ROI")

                detections = run_yolo_on_roi(frame, bbox)

                if detections:
                    print(f"YOLO detections: {len(detections)}")

                    # HIER später Übergabe an main.py (Save + Upload)
                    print(detections)

                detection_done = True
        else:
            # Lost Trackking - reset
            reset_tracking()


        check_tracker_timeout()
        cv2.imshow("Hornet Debug", display)


    # =====================
    # Debug Overlay
    # =====================

    if SHOW_DEBUG_VIDEO:
    
        y = 20
        line = 25

        cv2.putText(display, f"PI: {PI_ID}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        y += line
        cv2.putText(display, f"FPS: {fps:.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        y += line
        cv2.putText(display,
                    f"Motion: {'YES' if motion_detected else 'NO'}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,255) if motion_detected else (0,255,0),
                    2)

        y += line
        cv2.putText(display,
                    f"Tracking: {'YES' if tracking_active else 'NO'}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,255) if tracking_active else (150,150,150),
                    2)

        y += line
        cv2.putText(display,
                    f"Tracking frames: {tracking_frames}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2)

        y += line
        cv2.putText(display,
                    f"YOLO done: {'YES' if detection_done else 'NO'}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0) if detection_done else (0,0,255),
                    2)

        y += line
        cv2.putText(display,
                    f"Tracker: {TRACKER_TYPE}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2)

        #temp print
        print(          
            "tracking_active:", tracking_active,
            "| tracking_frames:", tracking_frames,
            "| detection_done:", detection_done,
            "| last_motion_time:", last_motion_time,

        )


        cv2.imshow("Hornet Debug", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break


# =====================
# Cleanup
# =====================
cam.release()
cv2.destroyAllWindows()
print("Motion-Gate beendet")
