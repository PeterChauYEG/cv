

## ensure that your terminal has camera permissions but runing these

brew install ffmpeg
ffmpeg -f avfoundation -list_devices true -i ""
ffmpeg -f avfoundation -i "0:0" -vf  "crop=1024:768:400:800" -pix_fmt yuv420p -y -r 10 out.mov
sudo killall VDCAssistant
imagesnap -w 2 snapshot.png -d "HD Pro Webcam C920"

## need
- brew install wget
- conda
