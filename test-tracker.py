import cv2
import time

# =====================
# CONFIG
# =====================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_CONTOUR_AREA = 500     # anpassen je nach Szene
TRACKER_TYPE = "CSRT"      # später einfach auf "KCF" ändern
SHOW_MASK = False          # Taste 'm' toggelt Maske

# =====================
# TRACKER FACTORY
# =====================
def create_tracker():
    if TRACKER_TYPE == "CSRT":
        return cv2.TrackerCSRT_create()
    elif TRACKER_TYPE == "KCF":
        return cv2.TrackerKCF_create()
    else:
        raise ValueError("Unbekannter Tracker")

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Kamera konnte nicht geöffnet werden")

# =====================
# BACKGROUND SUBTRACTOR
# =====================
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=25,
    detectShadows=False
)

tracker = None
tracking = False
bbox = None

prev_time = time.time()

print("DEBUG MODE gestartet")
print("Tasten:")
print("  q = Beenden")
print("  m = Motion-Maske ein/aus")
print("  r = Tracker zurücksetzen")

# =====================
# MAIN LOOP
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Motion Detection ---
    fgmask = fgbg.apply(gray)
    fgmask = cv2.medianBlur(fgmask, 5)

    contours, _ = cv2.findContours(
        fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    motion_candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            motion_candidates.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # --- Tracker Update ---
    if tracking:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "TRACKING", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            tracking = False
            tracker = None

    # --- Tracker Start ---
    if not tracking and motion_candidates:
        bbox = motion_candidates[0]
        tracker = create_tracker()
        tracker.init(frame, bbox)
        tracking = True

    # --- FPS ---
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Motion objects: {len(motion_candidates)}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- Anzeige ---
    if SHOW_MASK:
        cv2.imshow("Hornet Radar DEBUG (Mask)", fgmask)
    cv2.imshow("Hornet Radar DEBUG", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        SHOW_MASK = not SHOW_MASK
    elif key == ord('r'):
        tracking = False
        tracker = None

# =====================
# CLEANUP
# =====================
cap.release()
cv2.destroyAllWindows()
