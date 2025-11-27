import sys
import subprocess

print("MARATHON PREDICTION MODEL")

# 1: prepare data
print("\n1. PREPARE DATA")
result1 = subprocess.run([sys.executable, 'code/prepare_data.py'])
if result1.returncode != 0:
    print("error in data preparation")
    sys.exit(1)

# 2: train model
print("\n2. TRAIN MODEL")
result2 = subprocess.run([sys.executable, 'code/train_model.py'])
if result2.returncode != 0:
    print("error in model training")
    sys.exit(1)

# 3: predict marathon
print("\n3. PREDICT MARATHON")
result3 = subprocess.run([sys.executable, 'code/predict_marathon.py'])
if result3.returncode != 0:
    print("error in predictor")
    sys.exit(1)

print("\nDONE")