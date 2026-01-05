import cv2
import time
from camera import Camera
from config import *
from detection import load_model, run_detection

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

def start_tracker(frame, bbox, state: TrackingState):
    tracker = create_tracker()
    tracker.init(frame, bbox)
    state.start(tracker)

def update_tracker(frame, state: TrackingState):
    if not state.active or state.tracker is None:
        return None

    ok, bbox = state.tracker.update(frame)
    if ok:
        state.last_update = time.time()
        state.tracking_frames += 1
        return bbox

    return None

    
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


yolo_model = load_model()

# === Tracking State ===
from tracking_state import TrackingState
tracking_state = TrackingState()

frame_count = 0
last_time = time.time()
fps = 0.0

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

    ## === Reset per Frame ===
    motion_detected = False
    motion_boxes = []

    # === Update FPS ===
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

        motion_boxes = []
        motion_detected = False

        for c in contours:

            # == Detecting Motions ===
            if cv2.contourArea(c) < MOTION_MIN_AREA:
                continue
          
            motion_boxes.append((x, y, w, h))
            motion_detected = True
 
        if motion_detected and not state.active:
            bbox = largest_motion_box
            tracker = create_tracker()
            tracker.init(frame, bbox)
            state.start(tracker, bbox)


        # === Tracking Update ===

        if state.active:
            ok, bbox = state.tracker.update(frame)
            if ok:
                state.update(bbox)
            else:
                state.reset()

        
        if bbox is not None:

            # === Visualize Tracking ===
            draw_tracking(display, bbox)

            # === Tracking stable and no detection yet ===
            if tracking_state.is_stable(TRACKING_STABLE_FRAMES) and not tracking_state.detection_done:
                print("Stable tracking - running YOLO on ROI")

                detections = run_yolo_on_roi(frame, bbox)

                if detections:
                    print(f"YOLO detections: {len(detections)}")

                    # HIER später Übergabe an main.py (Save + Upload)
                    print(detections)

                tracking_state.detection_done = True

        # === Tracking Timeout ===
        if tracking_state.active:
            if time.time() - tracking_state.last_update > TRACKER_TIMEOUT:
                print("Tracking timeout → reset")
                tracking_state.reset()
   

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

        cv2.putText(display,
            f"Tracking: {'YES' if tracking_state.active else 'NO'}",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0,255,255) if tracking_state.active else (150,150,150), 2)

        y += line
        cv2.putText(display,
            f"Tracking frames: {tracking_state.frames_tracked}",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255,255,255), 2)

        y += line
        cv2.putText(display,
            f"YOLO done: {'YES' if tracking_state.detection_done else 'NO'}",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0,255,0) if tracking_state.detection_done else (0,0,255), 2)


        y += line
        cv2.putText(display,
                    f"Tracker: {TRACKER_TYPE}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2)

        #temp print
        if process_this_frame:
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
