# Theremin Controller (hand-tracking)

Hand-tracking theremin using MediaPipe Hands, OpenCV, and sounddevice.  

![image](https://github.com/otmojo/theremin/blob/main/adjusting-otmojo.gif)

## Requirements
- Python 3.9+ (Windows)
- Webcam and audio output device

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Choose output device
python - <<PY
import sounddevice as sd
for i, d in enumerate(sd.query_devices()):
    print(i, d['name'])
PY

Update `DEVICE_ID` in [main.py](cci:7://file:///c:/hal/d/th/venv/main.py:0:0-0:0) to your output device index.

## Run
python main.py

Esc to quit.

## Controls
- Right hand X controls pitch (smoothed)
- Left hand Y controls volume (with threshold)

## Contact
If you have any suggestions or questions, I’d be happy to hear from you:
- E-mail：ths50618@ths.hal.ac.jp
