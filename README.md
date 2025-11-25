# Adaptive Median Filter - Image Denoising Tool

A web-based application designed to demonstrate and compare image denoising algorithms. The tool focuses on removing salt-and-pepper noise using various filtering techniques, including a custom implementation of a Simple Adaptive Median Filter.

## Overview

This application allows users to upload an image, apply artificial salt-and-pepper noise at varying densities, and process the image using three different filters simultaneously. It calculates and displays quality metrics (PSNR and MSE) to objectively compare the performance of each filter.

## Features

* **Image Upload**: Supports PNG, JPG, JPEG, TIFF, and BMP formats.
* **Noise Simulation**: Add adjustable salt-and-pepper noise (10% to 90% density).
* **Multi-Filter Comparison**:
    * **Box Filter**: A 3x3 Weighted Average filter.
    * **Standard Median Filter**: Standard OpenCV implementation.
    * **Adaptive Median Filter (AMF)**: A proposed method that selectively filters only noisy pixels (values 0 or 255) while preserving original details.
* **Quality Metrics**: real-time calculation of:
    * **PSNR** (Peak Signal-to-Noise Ratio)
    * **MSE** (Mean Squared Error)
* **Visualization**: Side-by-side comparison of Original, Noisy, and Filtered images.
* **Export**: Download individual processed images or all results at once.
* **Responsive UI**: Dark-themed interface built with Bootstrap 5.

## Technical Details

### Algorithms Implemented

1.  **Box Filter (Weighted Average)**:
    Uses a fixed 3x3 kernel approximation of a Gaussian curve to blur noise.
    ```python
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]] / 16.0
    ```

2.  **Standard Median Filter**:
    Replaces each pixel with the median value of its neighborhood. Effective for salt-and-pepper noise but causes blurring in noise-free regions.

3.  **Simple Adaptive Median Filter**:
    A hybrid approach. It first identifies "noise" pixels (strictly 0 or 255 in grayscale).
    * If a pixel is noise -> Replace with Median value.
    * If a pixel is not noise -> Keep original value.
    This preserves edges better than the standard median filter.

*Note: All uploaded images are automatically resized to 512x512 pixels and converted to grayscale for processing.*

### Tech Stack

* **Backend**: Python, Flask
* **Image Processing**: OpenCV (`cv2`), NumPy, Scikit-image
* **Frontend**: HTML5, JavaScript, Bootstrap 5

## Installation

### Prerequisites

* Python 3.8 or higher
* pip (Python package manager)

### Setup Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd denoisingWeb
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    # On Linux/MacOS
    python3 -m venv .venv
    source .venv/bin/activate

    # On Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content or install directly:
    ```bash
    pip install flask opencv-python numpy scikit-image
    ```

## Usage

1.  **Run the application:**
    ```bash
    python app.py
    ```

2.  **Access the interface:**
    Open your web browser and navigate to:
    `http://127.0.0.1:5000`

3.  **Process an image:**
    * Click "Choose File" to upload an image.
    * Select the desired "Noise Density" (default is 30%).
    * Click "Process Image".
    * View metrics and download results.

## Project Structure

```text
/
├── app.py                 # Main Flask application and processing logic
├── .gitignore             # Git configuration
├── static/
│   └── uploads/           # Temporary storage for uploads (auto-generated)
└── templates/
    └── index.html         # Frontend interface
