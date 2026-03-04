# 🐝 hornet-radar

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-active-success)

Detection system for European hornets (*Vespa crabro*) and Asian hornets (*Vespa velutina nigrithorax*).

Please visit [Hornet-Radar](https://hornet-radar.com) for more details.

---

## 📦 Installation

### 🥧 Raspberry Pi setup

Go to the [Detailed view](https://hornet-radar.com/en/details) page and search for the highest PI-ID of all bait stations.  
Increment this by one (e.g. `PI-99`).  
This will be your PI_ID throughout the setup, referred to as `$PI_ID`.

Run the **Raspberry Pi Imager** from the official  
[Raspberry Pi website](https://www.raspberrypi.com/software/)  
and go through the setup:

- Select your Pi version → **64‑bit Raspberry Pi OS**
- Select your **microSD card**
- Choose the name `$PI_ID` and select your localization settings
- Select the username **`hornet`** and a password of your choice  
  (If you choose a different username, you must adjust `hornet-radar.service`)
- Enter your Wi‑Fi credentials and enable **SSH** if you want CLI access
- It is recommended to create a **Raspberry Pi Connect** account and enable it, which makes administration much easier

---

### 🐝 Hornet‑radar setup

Connect to your Raspberry Pi via  
https://connect.raspberrypi.com  
and open a terminal window to run the following commands:

| Command | Description |
|-------|------------|
| `sudo apt-get update && sudo apt-get upgrade` | Upgrade your Pi to the latest version |
| `git clone https://github.com/KarstenMS/hornet-radar.git` | Clone the latest hornet-radar repository |
| `pip install opencv-contrib-python --break-system-packages` | Install OpenCV Contrib |
| `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages` | Install Torch & Torchvision |
| `pip install yolov5 --break-system-packages` | Install YOLOv5 and Ultralytics |
| `pip install "urllib3>=2.6.0" --break-system-packages` | Install urllib3 to suppress YOLO errors |
| `pip uninstall opencv-python -y --break-system-packages` | Remove regular OpenCV (not needed) |
| `sudo apt-get install imx500-all` | Only required when using the Pi AI Camera |

---

### ⚙️ Hornet‑radar configuration

Edit the configuration file:

```bash
nano /home/hornet/hornet-radar/config.py
```

| Necessary Options                             |                                                                                                 Description    |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| PI_ID                                         | Unique identification number for your bait station                                                             |
| LATITUDE / LONGITUTE                          | Used to position the bait station on the https://hornet-radar.com/en/map                                       |
| SHOW_DEBUG_VIDEO                              | Useful for testing and troubleshooting; should be False in production                                          |
| ROOT                                          | Change only if you chose a different username                                                                  |
| CAMERA_TYPE                                   | Choose between Pi Camera or Webcam                                                                             |    
| CAMERA_WIDTH / CAMERA_HEIGHT / CAMERA_FPS     | Reduce values on older Raspberry Pi models                                                                     |
| CONFIDENCE_THRESHOLD                          | Increase if you get many false positives                                                                       |
| **SUPABASE_KEY**                              | Contact admin@hornet-radar.com to receive your personal upload key                                             |

All other settings can remain at their default values initially.

---

### ▶️ Running

```bash
python /home/hornet/hornet-radar/main.py
```
Optional flags:

- -v → Analyze videos from hornet-radar/detections/videos
- -i → Analyze images from hornet-radar/detections/images

### 🚀 Running in production

Once everything works as expected, create a systemd service to automatically start main.py after every reboot and every 12 hours.

Steps:
- Set SHOW_DEBUG_VIDEO = False in config.py
- Verify the service file:
  ```bash
  nano /home/hornet/hornet-radar/hornet-radar.service
  ```
- Move the service file:
  ```bash
  sudo mv /home/hornet/hornet-radar/hornet-radar.service /etc/systemd/system/hornet-radar.service
  ```
- Enable the service:
  ```bash
  sudo systemctl daemon-reload
  sudo systemctl enable hornet-radar.service
  sudo systemctl start hornet-radar.service
  ```

- Check status:
  ```bash
  systemctl status hornet-radar.service
  ```
- View live logs:
  ```bash
  journalctl -u hornet-radar.service -f
  ```
