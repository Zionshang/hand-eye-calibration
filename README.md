# Hand-Eye Calibration for Arx5_umi Robot

This repository provides a complete toolkit for performing **Eye-in-Hand** calibration between an `Arx5` robot arm and an Intel `RealSense D405` camera. The pipeline supports data collection, calibration computation using multiple OpenCV algorithms, and physical verification.

## 1. Environment Setup

Create a Python environment and install dependencies.

```bash

conda create -n handeye python=3.10

conda activate handeye

pip install -r requirements.txt

```
Ensure Arx5 SDK is in your PYTHONPATH(controlled via LCM)

```bash

export PYTHONPATH=/path/to/arx5-sdk/python:$PYTHONPATH

```

## 2. Data Collection

Teleoperate the robot to capture chessboard images paired with poses.

```bash

./run_collect.sh

```

- **Controls**: Keyboard keys to move robot .
- **H**: Save current image & pose.
- **Space**: Reset to Home.

## 3. Compute Calibration

Calculate the calibration matrix using captured data. Results are saved to `calibration_result.json`.

```bash

python compute_calibration.py

```

## 4. Verification (Touch Test)

Verify accuracy by commanding the robot to touch the chessboard corner using the computed calibration.

```bash

./run_verify.sh --method Horaud

# Options: Tsai, Park, Horaud, Daniilidis

```
- **H**: Detect board and execute touch test.

- **Space**: Reset to Home.  




