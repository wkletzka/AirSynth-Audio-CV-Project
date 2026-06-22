import pyrealsense2 as rs
import numpy as np
import cv2
import sounddevice as sd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI(title="Backend Server")
app.add_middleware( # Allow  eact frontend to talk to this Python backend
    CORSMiddleware,
    allow_origins=["*"], # In production, we'd restrict this to localhost:5173
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Synthesizer class
class SimpleSynth:
    def __init__(self):
        self.sample_rate = 44100
        self.amp_ramp = 0.0005  # bigger ramp = faster fade

        self.phase1 = 0 # We track phase as a raw angle (0 to 2*PI) instead of frame counts
        self.target_freq1 = 261.63 # Using curr and target amps for an 'envelope' to fade the sounds in
        self.current_freq1 = 261.63
        self.target_amp1 = 0
        self.current_amp1 = 0

        # Adding a second frequency only changes one thing; we still calc it's wave but before sending it
        # to the speakers, we just add the two waveforms together
        self.phase2 = 0 
        self.target_freq2 = 261.63 
        self.current_freq2 = 261.63
        self.target_amp2 = 0
        self.current_amp2 = 0

        self.stream = sd.OutputStream(
            channels=1,
            callback=self.audio_callback,
            samplerate=self.sample_rate
        )
        self.stream.start()

    # Audio callback function is what it uses to call itself over and over
    def audio_callback(self, outdata, frames, time_info, status): 
        # Frequency 1 calculation
        # Calculate the targets for the end of this specific block
        next_freq1 = 0.9 * self.current_freq1 + 0.1 * self.target_freq1

        next_amp1 = self.current_amp1
        if self.current_amp1 < self.target_amp1:
            next_amp1 = min(self.current_amp1 + self.amp_ramp * frames, self.target_amp1)
        elif self.current_amp1 > self.target_amp1:
            next_amp1 = max(self.current_amp1 - self.amp_ramp * frames, self.target_amp1)

        # Interpolation:
        # Instead of 1 flat number, create an array of values that smoothly glide 
        # from the current state to the next state over the length of the buffer.
        freqs1 = np.linspace(self.current_freq1, next_freq1, frames, endpoint=False)
        amps1 = np.linspace(self.current_amp1, next_amp1, frames, endpoint=False)

        # Continuous phase accumulation:
        # Calculate exactly how much the angle changes per sample, then cumulatively add it
        phase_increments1 = 2 * np.pi * freqs1 / self.sample_rate
        phases1 = self.phase1 + np.cumsum(phase_increments1)
        
        # Generate the perfectly smooth waveform
        # First stack the overtones (harmonics)
        base_wave1 = np.sin(phases1)               # Fundamental note (loudest)
        base_wave1 += 0.5 * np.sin(phases1 * 2)    # 1st Overtone (Octave up, half volume)
        base_wave1 += 0.25 * np.sin(phases1 * 3)   # 2nd Overtone (Octave + 5th, quarter vol)
        base_wave1 += 0.125 * np.sin(phases1 * 4)  # 3rd Overtone (Two octaves up)

        # Normalize and apply volume. We divide by 1.875 (which is 1 + 0.5 + 0.25 + 0.125) so the added waves 
        # don't push the volume past 1.0 and cause speaker clipping after we mult by amps.
        wave1 = amps1 * (base_wave1 / 1.875)

        # (Optional) frequency 2 wave calculation
        next_freq2 = 0.9 * self.current_freq2 + 0.1 * self.target_freq2

        next_amp2 = self.current_amp2
        if self.current_amp2 < self.target_amp2:
            next_amp2 = min(self.current_amp2 + self.amp_ramp * frames, self.target_amp2)
        elif self.current_amp2 > self.target_amp2:
            next_amp2 = max(self.current_amp2 - self.amp_ramp * frames, self.target_amp2)

        freqs2 = np.linspace(self.current_freq2, next_freq2, frames, endpoint=False)
        amps2 = np.linspace(self.current_amp2, next_amp2, frames, endpoint=False)

        phase_increments2 = 2 * np.pi * freqs2 / self.sample_rate
        phases2 = self.phase2 + np.cumsum(phase_increments2)
        
        base_wave2 = np.sin(phases2)              
        base_wave2 += 0.5 * np.sin(phases2 * 2)    
        base_wave2+= 0.25 * np.sin(phases2 * 3)   
        base_wave2 += 0.125 * np.sin(phases2 * 4)  

        # Normalize and apply volume. We divide by 1.875 (which is 1 + 0.5 + 0.25 + 0.125) so the added waves 
        # don't push the volume past 1.0 and cause speaker clipping after we mult by amps.
        wave2 = amps2 * (base_wave2 / 1.875)


        # Save state(s) for the next callback
        # Use modulo 2*PI so the float doesn't eventually overflow after playing for 3 hours
        self.phase1 = phases1[-1] % (2 * np.pi) 
        self.current_freq1 = next_freq1
        self.current_amp1 = next_amp1
        self.phase2 = phases2[-1] % (2 * np.pi) 
        self.current_freq2 = next_freq2
        self.current_amp2 = next_amp2

        # Calculate the combined wave by adding the two independent waves together
        mixed_wave = wave1 + wave2

        outdata[:] = mixed_wave.reshape(-1, 1)

    def stop(self):
        self.stream.stop()
        self.stream.close()

def handIsClosed(hand):
    # Check if a hand is closed
    palm_width = math.dist( # Using values between 0 and 1 to calc dist
        (hand[5].x, hand[5].y),    # index MCP
        (hand[17].x, hand[17].y)   # pinky MCP
    )
    thumb_pinky = math.dist(
        (hand[4].x, hand[4].y),    # thumb tip
        (hand[20].x, hand[20].y)   # pinky tip
    )

    # Normalize based on how wide our palm is in the camera
    ratio = thumb_pinky / palm_width

    print(ratio)
    if ratio < 1.33:
        return True, ratio
    else:
        return False, ratio

# Start-up:
# Instantiate the class
synth = SimpleSynth()

notes = [ # Array of notes to play
    261.63,  # C4
    277.18,  # C#4/Db4
    293.66,  # D4
    311.13,  # D#4/Eb4
    329.63,  # E4
    349.23,  # F4
    369.99,  # F#4/Gb4
    392.00,  # G4
    415.30,  # G#4/Ab4
    440.00,  # A4
    466.16,  # A#4/Bb4
    493.88,  # B4
    523.25   # C5
]

key_width = 640 // 13 # Each key is 80 pixels wide on the screen, filling up entire screen

# Setting up Intel Realsense
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

print("Starting air piano... (Press 'q' or Esc to quit)")

# Setting up MediaPipe hand tracking
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_hand_detection_confidence=0.3, # Confidence threshold - affects how sensitive it is to picking up your hands
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.15,
    num_hands=2 # Starting with 1 hand for simplicity; eventually add both hands and superposition to this
)

detector = vision.HandLandmarker.create_from_options(options)

# This is the generator funciton that our StreamingResponse uses to yield over data frames from the camera
def generate_frames():
    pipeline.start(config) # Start the camera

    try: # Error handling
        while True:

            # Grab the depth and color frames from the cam pipeline
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # flip the colors so MediaPipe can 'see' properly, needs RGB not BGR
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Format the image to be a mediapipe image so that the hand detector works properly
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_image
            )

            timestamp_ms = int(time.time() * 1000)

            result = detector.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # If we detected hands, move onto the audio logic based on the landmarkers
            if result.hand_landmarks:
                # Calculate which key to play based on location of hand
                hand1 = result.hand_landmarks[0]
                hand2 = None 
                if len(result.hand_landmarks) > 1: # If we detected a second hand, use it
                    hand2 = result.hand_landmarks[1]
                
                # First hand
                middle_fing_x_pixel1 = int(hand1[9].x * 640) # Base of middle finger 
                curr_key1 = middle_fing_x_pixel1 // key_width # Integer round to the nearest note

                # Check if the 1st hand is closed based on pinky to thumb distance as a ratio to palm width 
                # We need to use the ratio because the hand could be any depth from the camera
                hand1_closed, ratio1 = handIsClosed(hand1)
                ratio1 /= 3.1 # Normalize based on it's max realistic val
                ratio1 = min(ratio1, 1) # Ceiling of 1

                # ADD HAND TILT THAT INTRODUCES VIBRATO/DIFFERENT SOUND?

                # Second hand, if it exists
                if hand2 is not None:
                    middle_fing_x_pixel2 = int(hand2[9].x * 640) # Base of middle finger 
                    curr_key2 = middle_fing_x_pixel2 // key_width # Integer round to the nearest note

                    hand2_closed, ratio2 = handIsClosed(hand2)
                    ratio2 /= 3.1
                    ratio2 = min(ratio2, 1)
                
                    # If we're in this block, we know there's a 2nd hand active, so set it's freq/amp
                    if (0 <= curr_key2 < 13) and (hand2_closed == False):
                        synth.target_freq2 = notes[curr_key2] 
                        synth.target_amp2 = 0.5 * ratio2 * (1.0 - hand2[9].y)
                    elif hand2_closed == True:
                        synth.target_amp2 = 0.0

                else: # If we didn't detect 2nd hand, mute it
                    synth.target_amp2 = 0.0

                # Potentially add depth rules to this later, but it's causing choppy audio currently
                # keys should be between idx 0 and 13
                if (0 <= curr_key1 < 13) and (hand1_closed == False):
                    synth.target_freq1 = notes[curr_key1]
                    synth.target_amp1 = 0.5 * ratio1 * (1.0 - hand1[9].y) # Volume is a function of how closed your hand is, and how high up it is
                elif hand1_closed == True:
                    synth.target_amp1 = 0.0
                
                # Draw the landmarks on screen for a visual 
                for hand in result.hand_landmarks:
                    for landmark in hand:

                        x = int(landmark.x * 640)
                        y = int(landmark.y * 480)

                        cv2.circle(
                            color_image,
                            (x, y),
                            3,
                            (0, 255, 0),
                            -1
                        )

            else: # Otherwise no hands were detected, keep it on mute
                synth.target_amp1 = 0.0
                synth.target_amp2 = 0.0

            # Display the image on screen using opencv
            # cv2.namedWindow(
            #     "RealSense Air Piano",
            #     cv2.WINDOW_AUTOSIZE
            # )

            # cv2.imshow(
            #     "RealSense Air Piano",
            #     color_image
            # )

            # Quit the loop if we hit esc or q
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q') or key == 27:
                break
            
            # Now that all the camera/hand logic is done, we can send it to the web server
            ret, buffer = cv2.imencode('.jpg', color_image) # Compress frame to JPG
            if not ret:
                continue

            # Conv to bytes so it can travel over the network
            frame_bytes = buffer.tobytes()

            # Yield hands over the current chunk of data to FastAPI, then keeps running
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# Finishing up 
    finally:
        synth.stop()
        pipeline.stop()
        cv2.destroyAllWindows()

        print("Camera and audio stopped safely.")
        pass

# Routing the backend data to our web server
@app.get("/video_feed")
def video_feed(): # Streams the generator to our frontend
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# Later on, add more routes to fetch database stuff, i.e. @app.get("/stats")

# Running the server this way instead of just running the script
if __name__ == "__main__":
    print("Starting AirSynth")
    uvicorn.run(app, host="0.0.0.0", port=8000)
