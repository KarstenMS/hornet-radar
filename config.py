"""Hornet Radar: project-wide configuration constants (paths, thresholds, camera, Supabase)."""

import os

# --- Raspberry Pi setup ---
PI_ID = "PI-X"

LATITUDE = 00.0                                    # Get the values from Google maps
LONGITUDE = 00.0

SHOW_DEBUG_VIDEO = True                                         # Shows Debug Video on the PI (requires GUI), default False
DEBUG_DISPLAY_WIDTH = 960                                        # Downscale the debug window to this width before imshow. Huge bandwidth cut over Pi Connect / remote desktop (the full desktop incl. this window is re-encoded and streamed). Set 0 to show full resolution.

# --- Directories ---
ROOT = "/home/hornet/hornet-radar"

YOLO_DIR = os.path.join(ROOT, "yolov5")
MODEL_DIR = os.path.join(ROOT, "model", "yolov5s-all-data.pt")
MODEL_NAME = "yolo5"

IMAGES_DIR = os.path.join(ROOT, "detections", "images")         # For analyzing single pictures
VIDEOS_DIR = os.path.join(ROOT, "detections","videos")          # For analyzing videos
EVENTS_DIR = os.path.join(ROOT, "detections", "events")         # Directory for storing local events

# --- Camera configuration ---
CAMERA_TYPE = "picamera2"                                       # "picamera2" | "webcam"

CAMERA_WIDTH = 2048 #1024 2048
CAMERA_HEIGHT = 1536 #768 1536
CAMERA_FPS = 15

# Webcam only
WEBCAM_INDEX = 0

# Picamera2 only
PICAM_FORMAT = "RGB888"                                         # Picamera2 naming is reversed vs numpy: "RGB888" actually yields BGR arrays, matching cv2.VideoCapture.
FOCUS_DISTANCE_CM = 15                                          # Camera Module 3: focus distance in cm to the target (e.g. hive entrance). Set per-Pi. Ignored on IMX500 (fixed focus).

# --- Exposure ---
# The frame duration is pinned to CAMERA_FPS (FrameDurationLimits), so the
# capture rate can no longer collapse when auto-exposure picks a long exposure.
# This is critical for tracking: at a low frame rate fast insects jump hundreds
# of pixels between frames and cannot be associated across frames.
# In bright daylight, set a short EXPOSURE_TIME_US to freeze motion (sharp,
# blur-free insects). Leave it None to let auto-exposure adapt (safer for
# changing light, but exposure may lengthen and reintroduce motion blur).
EXPOSURE_TIME_US = None                                         # e.g. 6000 (=6 ms) in daylight to freeze fast movers; None = auto-exposure (bounded by the frame budget)
ANALOGUE_GAIN = None                                            # Sensor gain to pair with manual EXPOSURE_TIME_US (e.g. 1.0-2.0 in daylight). None = auto. Only used when EXPOSURE_TIME_US is set.

# --- White balance ---
# Auto AWB drifts under a saturated background (e.g. greenscreen bait surface) and
# differs per camera unit, so manual ColourGains is the more reliable default.
AWB_ENABLE = False                                              # True = use AWB_MODE preset; False = lock to COLOUR_GAINS below.
AWB_MODE = 5                                                    # libcamera preset (used only if AWB_ENABLE=True): 0=Auto 1=Incandescent 2=Tungsten 3=Fluorescent 4=Indoor 5=Daylight 6=Cloudy
COLOUR_GAINS = (1.6, 2.0)                                       # (red_gain, blue_gain). Increase blue to cool the image (less yellow). Tune per-Pi.

# --- Detection Settings ---
CONFIDENCE_THRESHOLD = 0.93                                     # Optional: confidence threshold for detections
MAX_YOLO_ATTEMPTS = 8                                           # Max YOLO inferences per track before giving up on confirmation.
YOLO_RETRY_INTERVAL_FRAMES = 5                                  # Frames to wait between consecutive YOLO attempts (so each retry sees a meaningfully different view).

# --- Presence check (confirmed insect feeding at the bait) ---
# Once a track is confirmed and sits still at the bait, MOG2 stops emitting a box
# (it learns the motionless insect into the background). Instead of relying on the
# coast budget alone, we periodically re-run YOLO on the frame: as long as YOLO
# still finds the insect at the track's box, the track is kept alive and its proof
# image is refreshed from the (sharp) sitting frame. When YOLO no longer finds it
# there, the insect has departed and the track is finalized into one event.
YOLO_PRESENCE_INTERVAL_FRAMES = 15                             # Re-check a sitting confirmed track with YOLO every N frames (~1 s at 15 FPS). Lower = faster departure detection but more inferences.
YOLO_PRESENCE_LOST_LIMIT = 2                                   # Consecutive failed presence checks (YOLO no longer sees the insect at the box) before the track counts as departed.

# --- Save event ---
THUMB_SIZE = 192, 108                                           # Pixel-Size for thumbnails. Default: 192, 108 
EVENT_RETENTION_DAYS = 180                                      # Number of days to keep local event data before deletion
MAX_EVENT_STORAGE_GB = 5                                        # Maximum storage for events in GB. If exceeded, oldest events will be deleted. 

# --- Vector settings ---
VECTOR_WINDOW = 8                                               # Number of boxes for approach/departure vector: approach = first 8 centers (arrival flight), departure = last 8 centers (exit flight). Matches TRACKING_STABLE_FRAMES so the approach vector is available by the time YOLO confirmation kicks in.
VECTOR_MIN_DISTANCE = 10.0                                      # Minimum pixel distance for a valid vector (to filter out noise)

# --- Supabase ---
SUPABASE_URL = "https://lebtnjdpjntaqheahjoi.supabase.co"
SUPABASE_KEY = ""                                               # SECURITY NOTE: Get API key from admin@hornet-radar.com
BUCKET_NAME = "hornet-detections"
TABLE_NAME = "sightings"

# --- Motion_Gate Tracking settings ---
TRACKER_TYPE = "AUTO"                                           # Available: "KCF", "CSRT", "MOSSE", "AUTO" (legacy single-tracker; unused by the multi-tracker)
FRAME_SKIP = 3                                                  # Video mode: analyse only every 3rd frame. Camera mode runs motion detection on every frame.
TRACKING_STABLE_FRAMES = 8                                      # Number of frames a track must exist before running YOLO confirmation

# --- Multi-object tracking (camera mode) ---
# The camera path tracks every moving insect by associating MOG2 motion boxes to
# persistent tracks each frame (greedy IoU + proximity, see matching.py). No
# OpenCV appearance tracker is used, so motion detection can run every frame.
MOTION_DOWNSCALE = 0.5                                          # Downscale factor for MOG2 only (0.5 = quarter the pixels => much faster on a Pi 5). Boxes are scaled back to full res. Use 1.0 to disable.
MATCH_IOU_THRESHOLD = 0.2                                       # Min IoU to associate a motion box with an existing track (overlap match)
MATCH_MAX_DISTANCE_RATIO = 0.12                                 # Proximity fallback: max center distance as a fraction of frame width (~245 px at 2048). Fast movers (e.g. flies) shift 120-280 px/frame at low FPS; too small => tracks fragment into new IDs. Raise toward 0.15 if fast movers still fragment, lower if nearby objects swap IDs.
MAX_COAST_FRAMES = 8                                            # Keep a track alive this many frames without a matched box (handles brief occlusion / insect flying past). Also enables re-association on reappearance.

# An insect sitting still at the bait produces no motion, so MOG2 stops emitting
# a box and the track would normally be killed after MAX_COAST_FRAMES. A track
# that stopped producing motion (coasting) while its last position is not near a
# frame edge -- i.e. it likely landed at the centrally-placed bait rather than
# flying out of frame -- gets a much larger coast budget instead, so the same
# insect keeps its ID across the whole feeding bout and resumes cleanly.
MAX_COAST_FRAMES_STATIONARY = 120                              # Coast budget for a track judged to be sitting at the bait (~8 s at 15 FPS). Acts as a safety net; the YOLO presence check is what actually keeps a feeding insect alive. Raise for longer bouts if the presence check is disabled.
STATIONARY_EDGE_MARGIN_RATIO = 0.06                            # A track whose last center is within this fraction of any frame edge is treated as "left the frame" (short coast), never as sitting.

YOLO_MATCH_IOU = 0.3                                            # Min IoU between a YOLO detection and a track's box to confirm that specific track

# --- Tracker Geometry Abort Thresholds (Abort Criterion) ---

TRACKER_INIT_MAX_AREA_RATIO = 0.15                              # 15% of frame
TRACKER_MAX_AREA_RATIO = 0.35                                   # >35% of frame = to big for a hornet
TRACKER_MIN_AREA_RATIO = 0.015                                  # <1.5% = too small to be a hornet at the fixed ~20 cm camera distance (~47k px at 2048x1536). A hornet at the bait fills ~6% of frame; flies/bees are <1%, so this rejects them before they spawn a track and get mislabeled. Lower toward 0.01 if real hornets get filtered, raise if small insects still slip through.
TRACKER_MAX_ASPECT_RATIO = 5.0                                  # extreme wide
TRACKER_MIN_ASPECT_RATIO = 0.2                                  # extreme small  
TRACKER_EDGE_MARGIN_RATIO = 0.02                                # 2% marge from edge
TRACKER_MAX_INVALID_FRAMES = 5                                  # Max consecutive implausible frames before aborting tracking

MIN_POST_CONFIRM_FRAMES = 6                                     # e.g 6–10 

# -- Motion Settings ---
MOTION_HISTORY = 300                                            # Amount of frames used for Backgroundmodel (low = faster, high = slower)
MOTION_VAR_THRESHOLD = 40                                       # Sensibility of motion detection (higher = less sensitive to slow/small movers)
MOTION_MIN_AREA = 47000                                         # Min pixel area for a relevant motion box. Kept ~= the track spawn min-area (TRACKER_MIN_AREA_RATIO ~47k px), so fly/bee-sized blobs (<1% of frame) never become motion boxes and cannot spawn tracks or steal matches from real hornets.
MOTION_KERNEL_SIZE = 5                                          # Size of morphological filtering (larger = erodes small blobs like ants before area check)


# --- Per-Pi overrides ---
# Anything defined in config_local.py wins over the defaults above.
# config_local.py is gitignored, so per-Pi values survive `git pull`.
# See config_local.example.py for the template.
try:
    from config_local import *  # noqa: F401, F403
except ImportError:
    pass
