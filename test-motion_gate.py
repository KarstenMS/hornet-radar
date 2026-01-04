import cv2
import time
from camera import Camera
from config import PI_ID

# =====================
# Settings
# =====================
FRAME_SKIP = 3              # nur jeder 3. Frame wird ausgewertet
MIN_MOTION_AREA = 800       # Pixel – Filter gegen Rauschen

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
    frame = cam.read()

    if frame is None:
        print("Kein Frame erhalten")
        break

    # Important: Picamera2 → OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    # =====================
    # Motion Detection
    # =====================
    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, 5)

    contours, _ = cv2.findContours(
        fg_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    motion_detected = False
    motion_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_MOTION_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        motion_boxes.append((x, y, w, h))
        motion_detected = True

    # =====================
    # Draw Motion Boxes
    # =====================
    for (x, y, w, h) in motion_boxes:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )

    # =====================
    # FPS
    # =====================
    now = time.time()
    fps = 1 / max(now - last_time, 1e-6)
    last_time = now

    # =====================
    # Debug Overlay
    # =====================
    cv2.putText(frame, f"PI: {PI_ID}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame,
                f"Motion: {'YES' if motion_detected else 'NO'}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255) if motion_detected else (0, 255, 0),
                2)

    cv2.putText(frame,
                f"Motion boxes: {len(motion_boxes)}",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)

    cv2.imshow("Hornet Radar – Motion Gate", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =====================
# Cleanup
# =====================
cam.release()
cv2.destroyAllWindows()
print("Motion-Gate beendet")
