#!/bin/bash
export PYTHONPATH=/home/jyx/python_ws/arx5-sdk/python:$PYTHONPATH
#python verify_calibration_lcm.py --method Tsai/Park/Horaud/Daniilidis "$@"
python verify_calibration_lcm.py --method Horaud "$@"