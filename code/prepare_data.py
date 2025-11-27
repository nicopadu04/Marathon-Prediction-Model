# imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# load dataset
df = pd.read_excel('data/raw/dataset_final.xlsx')
print("1. PREPARE DATA")

# keep only if has both half marathon, marathon and adjusted data
has_hm_and_marathon = df['mh_ti'].notna() & df['mf_ti'].notna()
df_filtered = df[has_hm_and_marathon].copy()
has_adjusted = df_filtered['mh_ti_adj'].notna() & df_filtered['mf_ti_adj'].notna()
df_filtered = df_filtered[has_adjusted].copy()
print(f"\nathletes with both half marathon, marathon and adjusted data: {len(df_filtered)}")

# pace (min/km)
df_filtered['k5_pace_minkm'] = (df_filtered['k5_ti'] / 60) / 5.0
df_filtered['k10_pace_minkm'] = (df_filtered['k10_ti'] / 60) / 10.0
df_filtered['mh_pace_minkm'] = (df_filtered['mh_ti'] / 60) / 21.0975
df_filtered['mf_pace_minkm'] = (df_filtered['mf_ti'] / 60) / 42.195

# slowdown features
print("\nSLOWDOW FEATURES")

# slowdown form half marathon to marathon
df_filtered['slowdown_hm_to_marathon'] = df_filtered['mf_pace_minkm'] / df_filtered['mh_pace_minkm']
print("\nstatistics for slowdown from half marathon to marathon:")
print(df_filtered['slowdown_hm_to_marathon'].describe())

# keep only if has reasonable slowdown
reasonable_slowdown = (
    (df_filtered['slowdown_hm_to_marathon'] >= 1.00) &
    (df_filtered['slowdown_hm_to_marathon'] <= 1.25)
)
df_clean = df_filtered[reasonable_slowdown].copy()
print(f"\nathletes before filter: {len(df_filtered)}")
print(f"athletes after filter: {len(df_clean)}")

# race times statistics
print("\nRACE TIMES STATISTICS")
df_clean['mh_time_minutes'] = df_clean['mh_ti'] / 60
df_clean['mf_time_minutes'] = df_clean['mf_ti'] / 60
print(f"\nhalf marathon time (minutes):")
print(f"min: {df_clean['mh_time_minutes'].min():.2f}")
print(f"max: {df_clean['mh_time_minutes'].max():.2f}")
print(f"mean: {df_clean['mh_time_minutes'].mean():.2f}")
print(f"median: {df_clean['mh_time_minutes'].median():.2f}")
print(f"\nmarathon time (minutes):")
print(f"min: {df_clean['mf_time_minutes'].min():.2f}")
print(f"max: {df_clean['mf_time_minutes'].max():.2f}")
print(f"mean: {df_clean['mf_time_minutes'].mean():.2f}")
print(f"median: {df_clean['mf_time_minutes'].median():.2f}")

# slowdown from 10k/5k to half marathon
df_clean['has_k5_data'] = df_clean['k5_pace_minkm'].notna().astype(int)
df_clean['has_k10_data'] = df_clean['k10_pace_minkm'].notna().astype(int)
print(f"\nhas_k5_data:  {df_clean['has_k5_data'].sum()} ")
print(f"has_k10_data: {df_clean['has_k10_data'].sum()} ")
df_clean['slowdown_10k_to_hm'] = df_clean['mh_pace_minkm'] / df_clean['k10_pace_minkm']
df_clean['slowdown_5k_to_hm'] = df_clean['mh_pace_minkm'] / df_clean['k5_pace_minkm']
print("\nstatistics for slowdown from 10k to half marathon:")
print(df_clean['slowdown_10k_to_hm'].describe())
print("\nstatistics for slowdown from 5k to half marathon:")
print(df_clean['slowdown_5k_to_hm'].describe())

# demographic features
print("\nDEMOGRAPHIC FEATURES")

# gender feature
df_clean['sex_M'] = (df_clean['gender'] == 0).astype(int)
df_clean['sex_F'] = (df_clean['gender'] == 1).astype(int)
print(f"\nmale (sex_M): {df_clean['sex_M'].sum()}")
print(f"female (sex_F): {df_clean['sex_F'].sum()}")

# age feature
df_clean['age_input'] = df_clean['age']
df_clean['age_squared'] = df_clean['age'] ** 2
print("\nstatistics for age:")
print(f"{df_clean['age_input'].describe()}")

# BMI ( weight(kg) / (height(m))^2 )
df_clean['bmi_input'] = df_clean['bmi']
print("\nstatistics for BMI:")
print(f"{df_clean['bmi_input'].describe()}")

# endurance category
df_clean['endurancecat_input'] = df_clean['endurancecat']
print("\nstatistics for endurance category:")
print(f"{df_clean['endurancecat_input'].value_counts().sort_index()}")

# training features
print("\nTRAINING FEATURES")

# keep only if max >= typical
df_clean['typical_km_week'] = df_clean['typical']
df_clean['max_km_week'] = df_clean['max']
n_fixed = (df_clean['max_km_week'] < df_clean['typical_km_week']).sum()
if n_fixed > 0:
    df_clean['max_km_week'] = df_clean[['max_km_week', 'typical_km_week']].max(axis=1)
print(f"\ntypical weekly training: {df_clean['typical_km_week'].describe()}")
print(f"\nmax weekly training: {df_clean['max_km_week'].describe()}")

# training volume ratio
df_clean['training_volume_ratio'] = df_clean['max_km_week'] / df_clean['typical_km_week'].replace(0, np.nan)
print(f"\ntraining volume ratio: {df_clean['training_volume_ratio'].describe()}")

# specific training (sprint runs and tempo runs)
df_clean['has_sprint'] = df_clean['sprint'].astype(int)
df_clean['has_tempo'] = df_clean['tempo'].astype(int)
print(f"\nsprint runs: {df_clean['has_sprint'].sum()} athletes")
print(f"tempo: {df_clean['has_tempo'].sum()} athletes")

# transform injury into dummy variable (0=no injury , 1=had injury)
df_clean['injury_history'] = (df_clean['injury'] > 1).astype(int)
print(f"\ndummy injury:")
print(f"no injury: {(df_clean['injury_history']==0).sum()}")
print(f"had injury: {(df_clean['injury_history']==1).sum()}")

# footwear
df_clean['footwear_type'] = df_clean['footwear'].astype(int)
print(f"\nfootwear:")
print(df_clean['footwear_type'].value_counts().sort_index())

# feature engineering
print("\nFEATURE ENGINEERING")

# pace ^2 and ^3
df_clean['k5_pace_squared'] = df_clean['k5_pace_minkm'] ** 2
df_clean['k10_pace_squared'] = df_clean['k10_pace_minkm'] ** 2
df_clean['mh_pace_squared'] = df_clean['mh_pace_minkm'] ** 2
df_clean['mh_pace_cubed'] = df_clean['mh_pace_minkm'] ** 3

# interactions
df_clean['sex_pace_interaction'] = df_clean['sex_M'] * df_clean['mh_pace_minkm']
df_clean['age_pace_interaction'] = df_clean['age_input'] * df_clean['mh_pace_minkm']
df_clean['bmi_pace_interaction'] = df_clean['bmi_input'] * df_clean['mh_pace_minkm']
df_clean['training_pace_interaction'] = df_clean['typical_km_week'] * df_clean['mh_pace_minkm']
df_clean['endurance_pace_interaction'] = df_clean['endurancecat_input'] * df_clean['mh_pace_minkm']
df_clean['age_bmi_interaction'] = df_clean['age_input'] * df_clean['bmi_input']

# print new feature
print(f"\nnew features:")
print(f"   - pace (squared and cubed)")
print(f"   - interactions (sex×pace, age×pace, bmi×pace, training×pace, endurance×pace, age×bmi)")

# features list
feature_columns = [
    'k5_pace_minkm', 'k10_pace_minkm', 'mh_pace_minkm',
    'has_k5_data', 'has_k10_data',
    'slowdown_10k_to_hm', 'slowdown_5k_to_hm',
    'sex_M', 'sex_F', 'age_input', 'age_squared', 'bmi_input', 'endurancecat_input',
    'typical_km_week', 'max_km_week', 'training_volume_ratio',
    'has_sprint', 'has_tempo', 'injury_history', 'footwear_type',
    'k5_pace_squared', 'k10_pace_squared', 'mh_pace_squared', 'mh_pace_cubed',
    'sex_pace_interaction', 'age_pace_interaction', 'bmi_pace_interaction',
    'training_pace_interaction', 'endurance_pace_interaction', 'age_bmi_interaction'
]
print("\nfeatures list:")
for i, feat in enumerate(feature_columns, 1):
    nan_count = df_clean[feat].isna().sum()
    nan_pct = (nan_count / len(df_clean)) * 100
    print(f"   {i:2d}. {feat:30s} (NaN: {nan_pct:5.1f}%)")

# split train/test
print("\nSPLIT TRAIN/TEST")

# split 70/30 (keeping same proportion of male and female)
train_idx, test_idx = train_test_split(
    df_clean.index,
    test_size=0.30,
    random_state=42,
    stratify=df_clean['sex_M']
)
df_train = df_clean.loc[train_idx].copy()
df_test = df_clean.loc[test_idx].copy()
print(f"\nTRAIN SET: {len(df_train)} athletes ({len(df_train)/len(df_clean)*100:.1f}%)")
print(f"male: {df_train['sex_M'].sum()}")
print(f"female: {df_train['sex_F'].sum()}")

print(f"\nTEST SET: {len(df_test)} athletes ({len(df_test)/len(df_clean)*100:.1f}%)")
print(f"male: {df_test['sex_M'].sum()}")
print(f"female: {df_test['sex_F'].sum()}")

# preparing data for model training
print("\nPREPARING DATA")

# X train/test and y train/test
X_train = df_train[feature_columns].copy()
y_train = df_train['slowdown_hm_to_marathon'].copy()
X_test = df_test[feature_columns].copy()
y_test = df_test['slowdown_hm_to_marathon'].copy()
print(f"\nX_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {len(y_train)}")
print(f"y_test: {len(y_test)}")

# info columns
info_columns = [
    'id', 'age', 'bmi', 'gender',
    'mh_ti', 'mf_ti',
    'mh_pace_minkm', 'mf_pace_minkm', 'slowdown_hm_to_marathon',
    'typical', 'max', 'sprint', 'tempo', 'endurancecat', 'injury', 'footwear'
]
train_info = df_train[info_columns].copy()
test_info = df_test[info_columns].copy()

# file saving
X_train.to_csv('data/processed_data/X_train.csv', index=False)
X_test.to_csv('data/processed_data/X_test.csv', index=False)
y_train.to_csv('data/processed_data/y_train.csv', index=False, header=['slowdown'])
y_test.to_csv('data/processed_data/y_test.csv', index=False, header=['slowdown'])
train_info.to_csv('data/processed_data/train_info.csv', index=False)
test_info.to_csv('data/processed_data/test_info.csv', index=False)
with open('data/processed_data/feature_columns.txt', 'w') as f:
    for col in feature_columns:
        f.write(f"{col}\n")

# statistics summary
print("\nSTATISTICS SUMMARY")
print(f"\ntarget slowdown from half marathon to marathon:")
print(f"train: {y_train.mean():.3f} ± {y_train.std():.3f}")
print(f"test:  {y_test.mean():.3f} ± {y_test.std():.3f}")
print(f"\npace half marathon (min/km):")
print(f"train: {X_train['mh_pace_minkm'].mean():.2f} ± {X_train['mh_pace_minkm'].std():.2f}")
print(f"test:  {X_test['mh_pace_minkm'].mean():.2f} ± {X_test['mh_pace_minkm'].std():.2f}")