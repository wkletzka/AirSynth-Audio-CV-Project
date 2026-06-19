import pyrealsense2 as rs
import numpy as np
import cv2
import sounddevice as sd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time


# Synthesizer class
class SimpleSynth:
    def __init__(self):
        self.sample_rate = 44100
        self.phase = 0 # We track phase as a raw angle (0 to 2*PI) instead of frame counts
        self.frequency = 261.63
        self.current_freq = 261.63

        self.current_amp = 0.0 # Using curr and target amps for an 'envelope' to fade the sounds in
        self.target_amp = 0.0
        self.amp_ramp = 0.0005  # bigger ramp = faster fade

        self.stream = sd.OutputStream(
            channels=1,
            callback=self.audio_callback,
            samplerate=self.sample_rate
        )
        self.stream.start()

    # Audio callback function is what it uses to call itself over and over
    def audio_callback(self, outdata, frames, time_info, status): 
        # Calculate the targets for the end of this specific block
        next_freq = 0.9 * self.current_freq + 0.1 * self.frequency

        next_amp = self.current_amp
        if self.current_amp < self.target_amp:
            next_amp = min(self.current_amp + self.amp_ramp * frames, self.target_amp)
        elif self.current_amp > self.target_amp:
            next_amp = max(self.current_amp - self.amp_ramp * frames, self.target_amp)

        # Interpolation:
        # Instead of 1 flat number, create an array of values that smoothly glide 
        # from the current state to the next state over the length of the buffer.
        freqs = np.linspace(self.current_freq, next_freq, frames, endpoint=False)
        amps = np.linspace(self.current_amp, next_amp, frames, endpoint=False)

        # Continuous phase accumulation:
        # Calculate exactly how much the angle changes per sample, then cumulatively add it
        phase_increments = 2 * np.pi * freqs / self.sample_rate
        phases = self.phase + np.cumsum(phase_increments)
        
        # Generate the perfectly smooth waveform
        wave = amps * np.sin(phases)

        # Save state for the next callback
        # Use modulo 2*PI so the float doesn't eventually overflow after playing for 3 hours
        self.phase = phases[-1] % (2 * np.pi) 
        self.current_freq = next_freq
        self.current_amp = next_amp

        outdata[:] = wave.reshape(-1, 1)

    def stop(self):
        self.stream.stop()
        self.stream.close()

# Instantiate the class
synth = SimpleSynth()

notes = [ # Array of notes to play
    261.63,  # C4
    293.66,  # D4
    329.63,  # E4
    349.23,  # F4
    392.00,  # G4
    440.00,  # A4
    493.88,  # B4
    523.25   # C5
]

key_width = 640 // 8 # Each key is 80 pixels wide on the screen, filling up entire screen

# Setting up Intel Realsense
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

config.enable_stream(
    rs.stream.depth,
    640,
    480,
    rs.format.z16,
    30
)

print("Starting air piano... (Press 'q' or Esc to quit)")

# Setting up MediaPipe hand tracking
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_hand_detection_confidence=0.25, # Confidence threshold - affects how sensitive it is to picking up your hands
    min_hand_presence_confidence=0.25,
    min_tracking_confidence=0.1,
    num_hands=1 # Starting with 1 hand for simplicity; eventually add both hands and superposition to this
)

detector = vision.HandLandmarker.create_from_options(options)

try: # Error handling
    pipeline.start(config) # Start the camera

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
            middle_fing_x_pixel = int(hand1[12].x * 640) # Tip of middle finger 
            curr_key = middle_fing_x_pixel // key_width # Integer round to the nearest note

            x = int(hand1[9].x * 640) 
            y = int(hand1[9].y * 480) 
            x = max(0, min(x, 639)) # Clamp the coordinates so they can't go our of bounds and crash script 
            y = max(0, min(y, 479)) 
            depth_to_palm = depth_frame.get_distance(x,y)
            print(depth_to_palm)

             # Potentially add depth rules to this later, but it's causing choppy audio currently
            if (0 <= curr_key < 8): # key should be between idx 0 and 7
                synth.frequency = notes[curr_key]
                synth.target_amp = 0.5
            

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
            synth.target_amp = 0.0

        # Display the image on screen using opencv
        cv2.namedWindow(
            "RealSense Air Piano",
            cv2.WINDOW_AUTOSIZE
        )

        cv2.imshow(
            "RealSense Air Piano",
            color_image
        )

        # Quit the loop if we hit esc or q
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            break


# Finishing up 
finally:
    synth.stop()
    pipeline.stop()
    cv2.destroyAllWindows()

    print("Camera and audio stopped safely.")