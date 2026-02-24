# Spatial Audio Instrument

## 🎵 Overview

This project is an interactive audio-visual installation that transforms real physical space into a playable instrument. By utilizing a depth camera, the system generates interactive "fields" in the air. Users can literally "play the air," manipulating audio and visuals simply by moving their hands and bodies through these generated zones.

## 🎯 Development Roadmap

The production of this project is divided into three core stages, scaling from flat interaction to complex volumetric sculpting.

### Phase 1: 2D Regions (The Foundation)

- **Goal:** Establish basic camera-to-audio communication.
- **Features:**
  - Track x/y movement using the depth camera.
  - Map flat, 2D planes/regions in the camera's view.
  - Trigger specific audio samples or MIDI notes when a user's hand enters these 2D zones.
  - Basic visual feedback indicating where the "invisible strings" are located.

### Phase 2: 3D Volumetric Fields (Adding Depth)

- **Goal:** Utilize the z-axis to create depth-based audio modulation.
- **Features:**
  - Incorporate depth (z-axis) data from the camera.
  - Create 3D bounding boxes or "cubes" of sound in the physical space.
  - Map depth to audio parameters (e.g., moving closer to the camera increases volume, adds reverb, or opens a synthesizer filter).
  - Visualizer updates to render 3D space and provide depth cues to the user.

### Phase 3: Intentionally Shaped Fields (Sculpting Sound)

- **Goal:** Move beyond basic geometric shapes into custom, intentional forms.
- **Features:**
  - Generate complex 3D shapes (spheres, winding tubes, abstract geometry) that act as the trigger zones.
  - Allow continuous, expressive playing (like a theremin on steroids) where the user slides their hands along the contours of invisible shapes to bend pitch or timbre.
  - Advanced visualizer mapping the exact shapes the user is interacting with in real-time.

---

## 🛠️ Tech Stack (To Be Determined)

_(Update this section as the project evolves)_

- **Hardware:** Depth Camera (e.g., Kinect, Intel RealSense, OAK-D)
- **Visuals / Tracking:** (e.g., TouchDesigner, Unity, openFrameworks)
- **Audio Engine:** (e.g., Ableton Live, Max/MSP, PureData, SuperCollider)

## 🚀 Setup and Installation

1. Connect the Depth Camera to your machine.
2. Install the necessary drivers (Link to drivers here).
3. Clone this repository: `git clone [your-repo-link]`
4. Open the main project file in [Software Name].
5. Route the MIDI/OSC output to your audio software.

## 🤝 Contributing

Feel free to fork this project, submit pull requests, or open issues to suggest improvements or new field shapes!
