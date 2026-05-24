import pyrealsense2 as rs
import numpy as np
import cv2
import sounddevice as sd
import mediapipe as mp
from mediapipe.tasks import python # Have to import newer version using mediapipe.tasks API
from mediapipe.tasks.python import vision
import time # Need timestamp for hand tracking video mode\
import math

# Setting up Synthesizer class
class SimpleSynth:
    def __init__(self):
        self.sample_rate = 44100
        self.phase = 0
        self.frequency = 261.63 # Starting at Middle C (C4)
        self.amplitude = 0.0    # 0.0 is muted, 0.5 is full volume
        
        # Start a background audio stream
        self.stream = sd.OutputStream(channels=1, callback=self.audio_callback, samplerate=self.sample_rate)
        self.stream.start()

        # For smoothing
        self.current_freq = self.frequency
        self.last_phase_reset_freq = self.frequency

    def audio_callback(self, outdata, frames, time, status):
        # Smooth frequency
        self.current_freq = 0.9 * self.current_freq + 0.1 * self.frequency

        # Reset phase on big jumps
        if abs(self.current_freq - self.last_phase_reset_freq) > 20:
            self.phase = 0
            self.last_phase_reset_freq = self.current_freq

        # Generate continuous sin wave
        t = (np.arange(frames) + self.phase) / self.sample_rate
        wave = self.amplitude * np.sin(2 * np.pi * self.current_freq * t)

        outdata[:] = wave.reshape(-1, 1)
        self.phase += frames

    def stop(self):
        self.stream.stop()
        self.stream.close()

synth = SimpleSynth()
last_update = 0
update_rate = 0.03

# Setting up Intel Realsense D435
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

print("Starting depth theremin... (Press 'q' or Esc to quit)")

# Setting up MediaPipe hand tracking 
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task") # Need to use landmark model .task file
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO, # Make sure to use video mode, rather than img mode for smoothness of detection
    min_hand_detection_confidence=0.25,
    min_hand_presence_confidence=0.25,
    min_tracking_confidence=0.1,
    num_hands=2
) 
detector = vision.HandLandmarker.create_from_options(options)

try:
    pipeline.start(config)

    while True:
        # Grab frames from Realsense - these are Realsense frame objects, NOT numpy arrays
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Displaying visuals
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Handtracking - might want to do this after the sound??
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)
        timestamp_ms = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)
        last_hand_reading = None # Display the last hand reading if we fail on one reading
        if result.hand_landmarks:
            # Music interfacing with hands
            both_closed = True # Default to closed
            avg_hand_depth = 0
            hand_counter = 0
            for hand in result.hand_landmarks:
                hand_counter += 1
                landmarks = [0, 5, 9, 13, 17] # Array to estimate hand location based on multiple landmarks
                depth_vals = []

                for i in landmarks:
                    x = int(hand[i].x * 640)
                    y = int(hand[i].y * 480)
                    x = max(0, min(x, 639)) # Clamp the coordinates so they can't go our of bounds and crash script
                    y = max(0, min(y, 479))
                    depth_vals.append(depth_frame.get_distance(x,y))

                hand_depth = sum(depth_vals) / len(depth_vals) # Divide total by num hands in frame to get avg depth
                avg_hand_depth += hand_depth

                # Only update closed if all hands are closed - est. this based on pinky-thumb distance and palm width
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
                if ratio > 1.6:
                    both_closed = False

            # Average out by how many hands we calculated depth for
            avg_hand_depth /= hand_counter

            # Set 0 inches as our base note (Middle C: 261.63 Hz)
            # Calculate how many 12-inch "chunks" further away the avg hand depth is
            avg_hand_depth_inches = avg_hand_depth * 39.3701
            half_steps_up = int((avg_hand_depth_inches) / 6.0)
            
            # Frequency
            target_freq = 261.63 * (2 ** (half_steps_up / 12.0)) # Every half step multiplies the frequency by 2^(1/12)

            # Amplitude 
            if both_closed == False: # only unmute if hands are open
                target_amp = 0.5 # Unmute
            else:   
                target_amp = 0.0   
            
            now = time.time()
            if now - last_update > update_rate:
                synth.frequency = target_freq
                synth.amplitude = target_amp
                last_update = now

            # Display text
            text = f"Dist: {avg_hand_depth:.1f} meters | +{half_steps_up} Steps | {synth.frequency:.1f} Hz"
            cv2.putText(color_image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display landmarks
            for hand in result.hand_landmarks: # for each hand
                for individual_landmark in hand: # for each landmark in that hand
                    x = int(individual_landmark.x * 640) # multiply landmark x and y by the width and height bc
                    y = int(individual_landmark.y * 480) # mp returns percentages of the screen, NOT pixel values
                    cv2.circle(
                        color_image,  # image to draw on
                        (x, y),       # center of circle
                        3,            # radius in pixels
                        (0, 255, 0),  # color (B, G, R)
                        -1            # thickness (-1 means fill in circle completely)
                    )

        else:
            synth.amplitude = 0.0 # Mute if hands aren't detected

        # Stack and show images on screen
        cv2.namedWindow('RealSense Theremin', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense Theremin', color_image)

        # Quit if you press q or esc
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break

finally:
    # Safely shut down the camera and the audio stream
    synth.stop()
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera and audio stopped safely.")