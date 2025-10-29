import cv2
import numpy as np
import mediapipe as mp
import sounddevice as sd
import queue

# ===== Output Device =====
DEVICE_ID = 10  # Change this to your output device ID
device_info = sd.query_devices(DEVICE_ID)
SAMPLE_RATE = int(device_info['default_samplerate'])
print(f"Using device {DEVICE_ID} with samplerate {SAMPLE_RATE}")

# ===== Mediapipe Hands =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ===== Camera =====
cap = cv2.VideoCapture(0)

# ===== Audio Queue =====
audio_q = queue.Queue()
DURATION = 0.3
prev_freq = 440

# ===== Audio Functions =====
def generate_theremin_tone(freq, vol, duration=DURATION):
    t = np.linspace(0, duration, int(SAMPLE_RATE*duration), endpoint=False)
    # Subtle vibrato
    vibrato = 5 * np.sin(2*np.pi*5*t)  # 5 Hz LFO ±5Hz
    wave = np.sin(2*np.pi*(freq + vibrato)*t) * vol
    wave = np.clip(wave, -1, 1)
    return wave.astype(np.float32)

def apply_adsr(wave, attack=0.1, release=0.15):
    N = len(wave)
    a = int(N*attack)
    r = int(N*release)
    sustain = max(N - a - r, 0)
    env = np.concatenate([
        np.linspace(0, 1, a),
        np.ones(sustain),
        np.linspace(1, 0, r)
    ])
    return wave * env

def smooth_freq(freq, alpha=0.2):
    global prev_freq
    freq = alpha*freq + (1-alpha)*prev_freq
    prev_freq = freq
    return freq

# ===== Callback =====
def audio_callback(outdata, frames, time, status):
    try:
        wave = audio_q.get_nowait()
        if len(wave) < frames:
            outdata[:len(wave),0] = wave
            outdata[len(wave):,0] = 0
        else:
            outdata[:,0] = wave[:frames]
    except queue.Empty:
        outdata.fill(0)

# ===== Output Stream =====
stream = sd.OutputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    device=DEVICE_ID,
    callback=audio_callback
)
stream.start()

# ===== Main Loop =====
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        h, w, _ = frame.shape

        freq = 440
        vol = 0.0

        # Detect both hands
        if results.multi_hand_landmarks:
            for hand_idx, handLms in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                index_tip = handLms.landmark[8]
                ix, iy = int(index_tip.x*w), int(index_tip.y*h)

                cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)

                if hand_idx == 0:
                    # Assume right hand controls pitch
                    freq = 220 + index_tip.x*660
                    freq = smooth_freq(freq)
                elif hand_idx == 1:
                    # Assume left hand controls volume
                    vol = 1 - np.clip(index_tip.y, 0, 1)
                    if vol < 0.05:
                        vol = 0

        cv2.putText(frame, f"freq: {freq:.1f} Hz", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"vol: {vol:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Theremin Controller", frame)

        # Enqueue audio
        if vol > 0:
            wave = generate_theremin_tone(freq, vol)
            wave = apply_adsr(wave)
            audio_q.put(wave)
        else:
            audio_q.put(np.zeros(int(SAMPLE_RATE*DURATION), dtype=np.float32))

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    stream.stop()
    stream.close()
