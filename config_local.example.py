"""Per-Pi overrides for config.py.

Copy this file to `config_local.py` on each Raspberry Pi and edit the
values. `config_local.py` is gitignored, so its contents survive
`git pull` while shared defaults in `config.py` keep updating normally.

Any name declared here overrides the matching name in `config.py`.
Only declare what differs per-Pi — leave everything else to config.py.
"""

# --- Identity ---
PI_ID = "PI-1"
LATITUDE = 0.0          # from Google Maps
LONGITUDE = 0.0

# --- Camera ---
FOCUS_DISTANCE_CM = 20  # Camera Module 3 only; ignored on IMX500

# --- Runtime ---
SHOW_DEBUG_VIDEO = False

# --- Secrets ---
SUPABASE_KEY = ""       # Get from admin@hornet-radar.com
