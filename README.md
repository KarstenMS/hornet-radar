# hornet-radar
Detection system for European (Vespa crapro) and Asian hornets (Vespa velutina nigrithorax)
Please visit https://hornet-radar.com for more details.


Installation:

Raspberry PI setup:

Go to https://hornet-radar.com -> Detailed view or Map and search vor the latest PI-ID. 
Increment this by one (e.g PI-99) , which will be your PI_ID through the further setup refering as $PI-ID

Run the Raspberry Pi Imager from https://www.raspberrypi.com/software/ and go through the setup:
- Select your PI version -> 64bit Raspberry PI OS System -> your Micro-SD Card
- Choose the name $PI-ID an and select your localization settings
- Select the username "hornet" and a password of your choice. You can also choose a different username, but than you'll have to adjust the hornet-radar.service file.
- Enter the credentials for your Wifi and enable SSH if you want to access the PI through command line - otherwise disable it
- I recommend creating a Raspberry Pi Connect account and enable it on the PI, making the admiistration much more easier.

Hornet-radar setup:

Connect to your Raspberry Pi through connect.raspberrypi.com and open a terminal window to run the following commands:

| Command                                                                                                    |                                    Description   |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| sudo apt-get update && sudo apt-get upgrade                                                                | Upgrade your Pi to latest version                |
| git-clone https://github.com/KarstenMS/hornet-radar.git                                                    | Clone the latest version of hornet-radar.git     |
| pip install opencv-contrib-python --break-system-packages                                                  | Install OpenCV Contrib                           |
| pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages     | Install Torch & Torchvision                      |
| pip install yolov5 --break-system-packages                                                                 | Install Yolov5 and Ultralytics                   |
| pip install "urllib3>=2.6.0" --break-system-packages                                                       | Install urllib3 to suppress yolo errors          |
| pip uninstall opencv-python -y --break-system-packages                                                     | Uninstall regular opencv as we use opencv-contrib|
| sudo apt-get install imx500-all                                                                            | Only requiered if you use the Pi AI-Camera       |

Hornet-radar config:

Run nano /home/hornet/hornet-radar/config.py to modify the configuration file:

| Necessary Options                             |                                                                                                 Description   |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| PI_ID                                         | Unique identification number of your Bait-Station                                                             |
| LATITUDE / LONGITUTE                          | Used for positioning your Bait-Station on https://hornet-radar.com/maps                                       |
| SHOW_DEBUG_VIDEO                              | Good for testing and troubleshooting. Should be set to False in production                                    |
| ROOT                                          | Change only if you use different username                                                                     |
| CAMERA_TYPE                                   | Choose if you use a Pi Camera or Webcam                                                                       |
| CAMERA_WIDTH / CAMERA_HEIGHT / CAMERA_FPS     | If you use older generations of RaspberryPi you may wanna decrease this for performance                       |
| CONFIDENCE_THRESHOLD                          | If you get a lot of false positives you can increase this                                                     |
| SUPABASE_KEY                                  | Please write to admin@hornet-radar.com to receive your personal Key for uploading Informations to the webpage |

All other settings can remain default for the beginning or modified if needed.
