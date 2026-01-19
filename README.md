# Hand-Eye Calibration for Arx5 Robot# Hand-Eye Calibration for Arx5 Robot# Hand-Eye Calibration for Robot Arm# Hand-Eye Calibration



This toolkit performs **Eye-in-Hand** calibration between an `Arx5` robot arm and a `RealSense D405` camera.



## 1. Environment SetupThis repository provides a complete toolkit for performing **Eye-in-Hand** calibration between an `Arx5` robot arm and an Intel `RealSense D405` camera. The pipeline supports data collection, calibration computation using multiple OpenCV algorithms, and physical verification.



Create a Python environment and install dependencies.



```bash## 🛠️ PrerequisitesA comprehensive toolkit for performing **Eye-in-Hand** calibration between an `Arx5` robot arm and a `RealSense D405` camera. The pipeline includes data collection, multi-algorithm calibration computation, and closed-loop accuracy verification.## 1. Setup

conda create -n handeye python=3.10

conda activate handeye

pip install -r requirements.txt

### Hardware```bash

# Ensure Arx5 SDK is in your PYTHONPATH

export PYTHONPATH=/path/to/arx5-sdk/python:$PYTHONPATH- **Arx5 Robot Arm** (controlled via LCM)

```

- **Intel RealSense D405 Camera** (or compatible)## 🛠️ Prerequisitesconda create -n handeye python=3.10

## 2. Data Collection

- **Calibration Chessboard**: Specifies `11x8` inner corners with `20mm` (verification) or `15mm` (setup dependent) square size.

Teleoperate the robot to capture chessboard images paired with poses.

conda activate handeye

```bash

python collect_data_lcm.py --save_path ./data### Software

```

- **Controls**: Keyboard keys to move robot (Translation/Rotation).

- **H**: Save current image & pose.- **Arx5 SDK**: Ensure the SDK is in your `PYTHONPATH`.

- **Space**: Reset to Home.

## 3. Compute Calibration

## 📦 Installation

Calculate the calibration matrix using captured data. Results are saved to `calibration_result.json`.

  - Arx5 Robot Arm (controlled via LCM)```

```bash

python compute_calibration.py

```

## 4. Verification (Touch Test)

Verify accuracy by commanding the robot to touch the chessboard corner using the computed calibration.

# Verify using a specific method (Default: Tsai)   ```

python verify_calibration_lcm.py --method Tsai

# Options: Tsai, Park, Horaud, Daniilidis

```
- **H**: Detect board and execute touch test.

- **Space**: Reset to Home.  
   ``````



