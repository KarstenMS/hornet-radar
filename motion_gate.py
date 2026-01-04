import cv2
import time
from camera import Camera
from config import *

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

    tracker = create_tracker()
    tracker.init(frame, bbox)
    
    tracking_active = True
    last_motion_time = time.time()

def update_tracker(frame):
    global tracking_active, last_motion_time

    if not tracking_active or tracker is None:
        return None

    success, bbox = tracker.update(frame)

    if success:
        last_motion_time = time.time()
        return bbox
    else:
        tracking_active = False
        return None
    
def check_tracker_timeout():
    global tracking_active

    if tracking_active and time.time() - last_motion_time > TRACKER_TIMEOUT:
        tracking_active = False

def draw_tracking(frame, bbox):
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv2.putText(frame, "TRACKING",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)

# =====================
# Init
# =====================
cam = Camera()

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

frame_count = 0
last_time = time.time()
fps = 0

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

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        cv2.imshow("Hornet Debug", display)
        continue

    fg_mask = bg_subtractor.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    motion_boxes = []

    for c in contours:
        if cv2.contourArea(c) < 500:
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
        draw_tracking(display, bbox)

    check_tracker_timeout()

    cv2.putText(display, f"Motion: {'YES' if motion_detected else 'NO'}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,0,255) if motion_detected else (0,255,0), 2)

    cv2.imshow("Hornet Debug", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break


# =====================
# Cleanup
# =====================
cam.release()
cv2.destroyAllWindows()
print("Motion-Gate beendet")
