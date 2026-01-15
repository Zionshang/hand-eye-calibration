#!/bin/bash
export PYTHONPATH=/home/jyx/python_ws/arx5-sdk/python:$PYTHONPATH
python collect_data_lcm.py "$@"
