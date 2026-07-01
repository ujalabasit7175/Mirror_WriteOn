# Mirror Camera with Fog Wipe Effect

## Overview

This project is an interactive computer vision application built with **Python**, **OpenCV**, and **MediaPipe**. It simulates a fogged mirror that users can wipe clean using their finger. The application also allows users to capture a photo by performing a two-hand **"L" gesture**, followed by a countdown and camera flash effect.

## Features

* Real-time webcam feed
* Realistic fog simulation
* Mouth detection to create a fogging effect
* Finger tracking to wipe away the fog
* Smooth drawing using motion smoothing
* Two-hand "L" gesture detection for taking photos
* 3-second countdown before capturing an image
* Camera flash animation
* Viewfinder overlay
* Automatically saves captured photos

## Technologies Used

* Python 3.10
* OpenCV
* MediaPipe
* NumPy

## Requirements

* Python 3.10
* Webcam
* Windows, macOS, or Linux

> If you are using a different Python version, you must create and use a Python 3.10 virtual environment to ensure compatibility with MediaPipe and other dependencies.

## Installation

1. Clone the repository.

```bash
git clone https://github.com/ujalabasit7175/Mirror_WriteOn.git
cd Mirror_WriteOn
```

2. If you are not using Python 3.10, create and activate a Python 3.10 virtual environment.

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages.

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file, install them manually:

```bash
pip install opencv-python mediapipe numpy
```

## Running the Project

Run the main Python file:

```bash
python mirror_writeon.py
```

## Output

Captured images are automatically saved in the project directory with filenames similar to:

```text
Snap_XXXXXXXX.jpg
```

## Project Structure

```text
├── mirror_writeon.py
├── README.md
└── requirements.txt
```

## Author

**Ujala Basit**

