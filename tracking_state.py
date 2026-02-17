import time


class TrackingState:
    # =====================
    # Initialization
    # =====================

    def __init__(self):
        self.reset()

    def reset(self):
            # --- Core ---
            self.active = False
            self.tracker = None

            # --- Geometry ---
            self.bbox = None
            self.centers = []
            self.frames_tracked = 0
            self.invalid_frames = 0

            # --- Timing ---
            self.start_frame_ts = None
            self.end_frame_ts = None
            self.start_time = None
            self.last_update = None
            self.dwell_time = 0.0

           # --- YOLO / Confirmation ---
            self.confirmed: bool = False
            self.confirmed_label = None
            self.confirmed_confidence = None
            self.detection_done: bool = False
            self.yolo_attempts = 0
            self.confirmed_frame = None
            self.confirmed_frame_shape = None
            self.confirmed_frame_ts = None
            self.stable_bbox = None
            self.confirmed_bbox = None
            self.confirmed_yolo_bbox = None
            self.confirmed_centers = []
            self.detections = []
            self.last_good_frame = None
            self.last_good_frame_shape = None
            self.frames_since_confirmed = 0
            

            

    # =====================
    # Lifecycle
    # =====================
    def start(self, tracker, bbox, frame_shape):    
        self.tracker = tracker
        self.bbox = bbox
        self.active = True
        self.frames_tracked = 0
        self.detection_done = False
        self.centers = []

        self.start_frame_ts = time.time()
        self.end_frame_ts = self.start_frame_ts
        self.last_update = self.start_frame_ts

        self.frame_shape = frame_shape

    def update(self, bbox):
        self.bbox = bbox
        self.frames_tracked += 1

        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        self.centers.append((cx, cy))

        self.end_frame_ts = time.time()
        self.dwell_time = self.end_frame_ts - self.start_frame_ts
        self.last_update = self.end_frame_ts