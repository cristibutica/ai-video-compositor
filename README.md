# Context-Aware DSP Video Processor

A real-time, context-aware Digital Signal Processing (DSP) application that uses Artificial Intelligence to separate the subject from the background, allowing selective application of 2D convolution mathematical filters.

## Overview

Unlike standard convolution filters that apply blindly to the entire image, this system uses a hybrid AI-DSP architecture. It extracts a semantic mask of the user in real-time and routes the foreground and background through separate DSP pipelines. This allows for effects like Edge Detection on the subject while heavily blurring the background, mimicking professional video-conferencing software.

## Features

* **AI Segmentation:** Uses Google's MediaPipe (Selfie Segmentation) to generate a high-speed, binary semantic mask.
* **Custom DSP Engine:** Applies spatial 2D convolution filters via OpenCV (`cv2.filter2D`).
* **Available Kernels:** Identity (Bypass), Edge Detect (Laplacian), Sharpen, Emboss, and Box Blur (Low-Pass).
* **CPU Optimization (Bypass Logic):** Bypasses heavy matrix multiplication for layers set to "Normal", significantly reducing CPU load and maintaining ~30 FPS.
* **Interactive GUI:** Built with PyQt6, featuring a non-blocking `QTimer` asynchronous event loop for smooth video rendering.

## Architecture (Separation of Concerns)

* `main.py` — Application launcher and entry point.
* `ui/main_window.py` — PyQt6 interface, asynchronous 30ms video loop, and hardware connection.
* `core/ai_segmenter.py` — Neural network module for probability map extraction and thresholding.
* `core/dsp_engine.py` — Core mathematical matrix definitions and Alpha Compositing logic.

## Requirements & Installation

> **Note:** This project specifically requires **Python 3.11** for optimal compatibility with the specific MediaPipe build.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/cristibutica/ai-video-compositor.git
   cd ai-video-compositor
   ```

2. **Create a virtual environment:**

   ```bash
   python3.11 -m venv dsp_env
   source dsp_env/bin/activate  # On Windows use: dsp_env\Scripts\activate
   ```

3. **Install dependencies:**

   > **Note:** We strictly use MediaPipe version `0.10.14` to retain access to the legacy solutions API.

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application from the terminal:

```bash
python main.py
```

Use the interactive buttons on the right side of the screen to change the DSP kernels applied to your foreground and background independently.