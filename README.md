# Hand-Eye Calibration

## 1. Setup
```bash
conda create -n handeye python=3.10
conda activate handeye
pip install -r requirements.txt
export PYTHONPATH=/home/zishang/py-workspace/arx5-sdk/python:$PYTHONPATH
```

## 2. Pipeline

**Step 1: Intrinsics**
```bash
python check_intrinsics.py
```

**Step 2: Collect Data**
```bash
bash run_collect.sh
```
> `[H]`: Save Frame | `[Space]`: Reset

**Step 3: Compute**
```bash
python compute_calibration.py
```

**Step 4: Verify**
```bash
bash run_verify.sh
```
> `[H]`: Touch Target | `[Space]`: Reset
