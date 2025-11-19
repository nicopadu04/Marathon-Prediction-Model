# imports
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

print("3. PREDICTOR")

# load model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)
print(f"\nmodel used: {model_info['model_name']}")

# helper function for parsing input with ,/.
def parse_float_input(prompt):
    """parse input accepting both dot and comma as decimal separator"""
    while True:
        try:
            inp = input(prompt)
            inp = inp.replace(',', '.')
            return float(inp)
        except ValueError:
            print("error: please enter a valid number (ex: 23.5 or 23,5)")

# inputs: times
print("\nSECTION 1: TIMES")
print("\ninput times of runs you have done recently, more data, more accuracy")

# half marathon
print("\nHALF MARATHON (21km) - compulsory")
while True:
    hm_minutes = parse_float_input("time (total minutes, ex: 100.5): ")
    if hm_minutes <= 0 or hm_minutes > 180:
        print("input a reasonable time (< 180)")
        continue
    hm_seconds = hm_minutes * 60
    hm_pace_minkm = hm_minutes / 21.0975
    # converting h:mm:ss to display
    hm_h = int(hm_seconds // 3600)
    hm_m = int((hm_seconds % 3600) // 60)
    hm_s = int(hm_seconds % 60)
    break

# 5K
print("\n5K - optional")
has_5k = input("have you run a 5K recently? (y/n): ").lower().strip() == 'y'
k5_seconds = np.nan
k5_pace_minkm = np.nan
if has_5k:
    while True:
        k5_minutes = parse_float_input("time (total minutes, ex: 22.5): ")
        if k5_minutes <= 0 or k5_minutes > 60:
            print("input a reasonable time (< 60)")
            continue
        k5_seconds = k5_minutes * 60
        k5_pace_minkm = k5_minutes / 5.0
        break

# 10K
print("\n10K - optional")
has_10k = input("have you run a 10K recently? (y/n): ").lower().strip() == 'y'
k10_seconds = np.nan
k10_pace_minkm = np.nan
if has_10k:
    while True:
        k10_minutes = parse_float_input("time (total minutes, ex: 50.5): ")
        if k10_minutes <= 0 or k10_minutes > 120:
            print("input a reasonable time (< 120)")
            continue
        k10_seconds = k10_minutes * 60
        k10_pace_minkm = k10_minutes / 10.0
        break

# inputs: demographic
print("\nSECTION 2: DEMOGRAPHIC")

# sex
while True:
    sex_input = input("\nsex (M/F): ").strip().upper()
    if sex_input in ['M', 'F']:
        sex_M = 1 if sex_input == 'M' else 0
        sex_F = 1 if sex_input == 'F' else 0
        gender = 0 if sex_input == 'M' else 1
        break
    print("input M or F")

# age
while True:
    age = parse_float_input("\nage (yrs): ")
    if 15 <= age <= 99:
        break
    print("input a reasonable age (< 99)")

# weight and height
while True:
    weight = parse_float_input("\nweight (kg): ")
    if 30 <= weight <= 200:
        break
    print("input a reasonable weight (30< w < 200)")
while True:
    height = parse_float_input("\nheight (cm): ")
    if 130 <= height <= 230:
        break
    print("input a reasonable height (130< h < 230)")

# compute BMI
bmi = weight / ((height / 100) ** 2)

# inputs: training
print("\nSECTION 3: TRAINING")

# typic weekly KM
while True:
    typical_km = parse_float_input("\ntypic weekly KM: ")
    if 0 <= typical_km <= 300:
        break
    print("input a reasonable amount (< 300)")

# max weekly KM
while True:
    max_km = parse_float_input("\nmax weekly KM: ")
    if typical_km <= max_km <= 400:
        break
    print(f"has to be >= {typical_km:.0f} km (your typic)")

# sprint sessions
sprint_input = input("\ndo you usually have sprint/interval training sessions? (y/n): ").strip().lower()
has_sprint = 1 if sprint_input == 'y' else 0

# tempo sessions
tempo_input = input("\ndo you usually do tempo runs? (y/n): ").strip().lower()
has_tempo = 1 if tempo_input == 'y' else 0

# Categoria resistenza
print("\nendurance category:")
print("1 = beginner (little experience)")
print("2 = intermediate (moderate experience)")
print("3 = advanced (a lot of experience)")
print("4 = elite (competitive)")
while True:
    endurancecat = int(input("choose category (1-4): "))
    if 1 <= endurancecat <= 4:
        break
    print("choose between 1 and 4")

# injury
injury_input = input("\nhave you suffered from any injury recently? (y/n): ").strip().lower()
injury_raw = 2 if injury_input == 'y' else 1
injury_history = 1 if injury_raw > 1 else 0

# footwear
print("\nfootwear:")
print("1 = cushioned shoes")
print("2 = neutral/minimalist shoes")
print("3 = other")
while True:
    footwear_type = int(input("choose footwear (1-3): "))
    if footwear_type in [1, 2, 3]:
        break
    print("choose between 1, 2 or 3")

# features analysis
print("\nANALYZING YOUR DATA")

# progressive slowdown
slowdown_10k_to_hm = hm_pace_minkm / k10_pace_minkm if not np.isnan(k10_pace_minkm) else np.nan
slowdown_5k_to_hm = hm_pace_minkm / k5_pace_minkm if not np.isnan(k5_pace_minkm) else np.nan
# training volume ratio
training_volume_ratio = max_km / typical_km if typical_km > 0 else np.nan
# engineered features
k5_pace_squared = k5_pace_minkm ** 2 if not np.isnan(k5_pace_minkm) else np.nan
k10_pace_squared = k10_pace_minkm ** 2 if not np.isnan(k10_pace_minkm) else np.nan
mh_pace_squared = hm_pace_minkm ** 2
mh_pace_cubed = hm_pace_minkm ** 3
# interactions features
sex_pace_interaction = sex_M * hm_pace_minkm
age_pace_interaction = age * hm_pace_minkm
bmi_pace_interaction = bmi * hm_pace_minkm
training_pace_interaction = typical_km * hm_pace_minkm
endurance_pace_interaction = endurancecat * hm_pace_minkm
age_bmi_interaction = age * bmi
age_squared = age ** 2
# features dataframe
features = pd.DataFrame([{
    'k5_pace_minkm': k5_pace_minkm,
    'k10_pace_minkm': k10_pace_minkm,
    'mh_pace_minkm': hm_pace_minkm,
    'has_k5_data': 1 if has_5k else 0,
    'has_k10_data': 1 if has_10k else 0,
    'slowdown_10k_to_hm': slowdown_10k_to_hm,
    'slowdown_5k_to_hm': slowdown_5k_to_hm,
    'sex_M': sex_M,
    'sex_F': sex_F,
    'age_input': age,
    'age_squared': age_squared,
    'bmi_input': bmi,
    'endurancecat_input': endurancecat,
    'typical_km_week': typical_km,
    'max_km_week': max_km,
    'training_volume_ratio': training_volume_ratio,
    'has_sprint': has_sprint,
    'has_tempo': has_tempo,
    'injury_history': injury_history,
    'footwear_type': footwear_type,
    'k5_pace_squared': k5_pace_squared,
    'k10_pace_squared': k10_pace_squared,
    'mh_pace_squared': mh_pace_squared,
    'mh_pace_cubed': mh_pace_cubed,
    'sex_pace_interaction': sex_pace_interaction,
    'age_pace_interaction': age_pace_interaction,
    'bmi_pace_interaction': bmi_pace_interaction,
    'training_pace_interaction': training_pace_interaction,
    'endurance_pace_interaction': endurance_pace_interaction,
    'age_bmi_interaction': age_bmi_interaction
}])
features = features[model_info['feature_columns']]

# user data summary
print(f"\nTIMES:")
if has_5k:
    k5_h = int(k5_seconds // 3600)
    k5_m = int((k5_seconds % 3600) // 60)
    k5_s = int(k5_seconds % 60)
    print(f"   5K:  {k5_m}min {k5_s}s (pace: {k5_pace_minkm:.2f} min/km)")
else:
    print(f"   5K:  not given")

if has_10k:
    k10_h = int(k10_seconds // 3600)
    k10_m = int((k10_seconds % 3600) // 60)
    k10_s = int(k10_seconds % 60)
    print(f"   10K: {k10_m}min {k10_s}s (pace: {k10_pace_minkm:.2f} min/km)")
else:
    print(f"   10K: not given")

print(f"   half marathon:  {hm_h}h {hm_m}min {hm_s}s (pace: {hm_pace_minkm:.2f} min/km)")

print(f"\nDEMOGRAPHIC:")
print(f"   sex: {'male' if sex_input=='M' else 'female'}")
print(f"   age: {int(age)} yrs")
print(f"   BMI: {bmi:.1f}")

print(f"\nTRAINING:")
print(f"   typic weekly KM: {typical_km}")
print(f"   max weekly KM: {max_km}")
if not np.isnan(training_volume_ratio):
    print(f"   training volume ratio: {training_volume_ratio:.2f}")
print(f"   sprint sessions: {'yes' if has_sprint else 'no'}")
print(f"   tempo runs: {'yes' if has_tempo else 'no'}")
print(f"   endurance: {['beginner', 'intermediate', 'advanced', 'elite'][endurancecat-1]}")
print(f"   injury: {'yes' if injury_history else 'no'}")

if has_5k and not np.isnan(slowdown_5k_to_hm):
    print(f"\nslowdown from 5K to half marathon: {slowdown_5k_to_hm:.3f} ({(slowdown_5k_to_hm-1)*100:.1f}% slowdown)")
if has_10k and not np.isnan(slowdown_10k_to_hm):
    print(f"slowdown from 10K to half marathon: {slowdown_10k_to_hm:.3f} ({(slowdown_10k_to_hm-1)*100:.1f}% slowdown)")

# prediction

# handling missing values (5/10 K) with training set medians
if 'feature_medians' in model_info:
    feature_medians = model_info['feature_medians']
    for col in features.columns:
        if pd.isna(features[col].iloc[0]) and col in feature_medians:
            features.at[0, col] = feature_medians[col]

# predict slowdown
predicted_slowdown = model.predict(features)[0]

# if slowdown < 1.0 
if predicted_slowdown < 1.0:
    print(f"\nthe model predicted a slowdown factor less than 1.0 ({predicted_slowdown:.3f}) which is impossible, please try again with new data")
    exit(1)

# pace and marathon time
predicted_marathon_pace_minkm = predicted_slowdown * hm_pace_minkm
predicted_marathon_seconds = predicted_marathon_pace_minkm * 60 * 42.195

# converting in h:mm:ss
pred_h = int(predicted_marathon_seconds // 3600)
pred_m = int((predicted_marathon_seconds % 3600) // 60)
pred_s = int(predicted_marathon_seconds % 60)

# resluts
print("\nRESULTS")
print(f"\nslowdown factor: {predicted_slowdown:.3f}")
print(f"you will have a slowdown of {(predicted_slowdown-1)*100:.1f}%")
print(f"\nhalf marathon pace: {hm_pace_minkm:.2f} min/km")
print(f"predicted marathon pace: {predicted_marathon_pace_minkm:.2f} min/km")
print(f"\nslowdown: +{predicted_marathon_pace_minkm - hm_pace_minkm:.2f} min/km")
print("\nPREDICTED MARATHON TIME")
print(f"\n{pred_h}h {pred_m}min {pred_s}s")

# error
print("\nPOSSIBLE ERROR")
mae_seconds = model_info['test_mae_minutes'] * 60
# no negative values
lower_bound_seconds = max(0, predicted_marathon_seconds - mae_seconds)
upper_bound_seconds = predicted_marathon_seconds + mae_seconds
lower_h = int(lower_bound_seconds // 3600)
lower_m = int((lower_bound_seconds % 3600) // 60)
lower_s = int(lower_bound_seconds % 60)
upper_h = int(upper_bound_seconds // 3600)
upper_m = int((upper_bound_seconds % 3600) // 60)
upper_s = int(upper_bound_seconds % 60)

print(f"\npredicted range (±{model_info['test_mae_minutes']:.1f} min):")
print(f"{lower_h}h {lower_m}min {lower_s}s  -  {upper_h}h {upper_m}min {upper_s}s")

# prediction quality
print(f"\nPREDICTION QUALITY:")
if has_5k and has_10k:
    print(f"excellent, you provided 5K, 10K, and half marathon data")
    print(f"the prediction is very reliable")
elif has_5k or has_10k:
    print(f"good, you provided {'5K' if has_5k else '10K'} and half marathon data")
    print(f"the prediction is reliable but could improve with more data")
else:
    print(f"basic, you provided only half marathon data")
    print(f"consider adding 5K or 10K times for better accuracy")

# complete
print("\nPREDICTION COMPLETE!")
print(f"\ngood luck with your marathon!")