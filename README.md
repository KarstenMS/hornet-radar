# hornet-radar
Detection system for European (Vespa crapro) and Asian hornets (Vespa velutina nigrithorax)



Installation:

Raspberry PI setup:
Go to https://hornet-radar.com -> Detailed view or Map and search vor the latest PI-ID. 
Increment this by one (e.g PI-99) , which will be your PI_ID through the further setup refering as $PI-ID

- Run the Raspberry Pi Imager from ht
- tps://www.raspberrypi.com/software/ and go through the setup:
- Select your PI version -> 64bit Raspberry PI OS System -> your Micro-SD Card
- Choose the name $PI-ID an and select your localization settings
- Select the username hornet and a password you prefer. You can also select a different username, but than you'll have to adjust the hornet-radar.service file.
- Enter the credentials for your Wifi and enable SSH if you want to access the PI through command line - otherwise disable it
- I recommend creating a Raspberry Pi Connect account and enable it on the PI, making the admiistration much easier.
- 
