# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle

print("2. TRAIN MODELS")

# load files
X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')['slowdown']
y_test = pd.read_csv('data/processed_data/y_test.csv')['slowdown']
train_info = pd.read_csv('data/processed_data/train_info.csv')
test_info = pd.read_csv('data/processed_data/test_info.csv')
with open('data/processed_data/feature_columns.txt', 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]

# NaN handling
print("\nNaN HANDLING")
nan_summary = []
for col in X_train.columns:
    train_nans = X_train[col].isna().sum()
    test_nans = X_test[col].isna().sum()
    
    if train_nans > 0 or test_nans > 0:
        nan_summary.append({
            'Feature': col,
            'Train_NaN': train_nans,
            'Train_%': f"{(train_nans/len(X_train))*100:.1f}%",
            'Test_NaN': test_nans,
            'Test_%': f"{(test_nans/len(X_test))*100:.1f}%"
        })
if nan_summary:
    print("\nfeatures with NaN:")
    nan_df = pd.DataFrame(nan_summary)
    print(nan_df.to_string(index=False))

# training models
print("\nTRAINING MODELS")
models = {}

# linear regression (imputation for NaN with median)
X_train_imputed = X_train.fillna(X_train.median())
X_test_imputed = X_test.fillna(X_train.median()) 
lr = LinearRegression()
lr.fit(X_train_imputed, y_train)
models['Linear Regression'] = {
    'model': lr,
    'X_train': X_train_imputed,
    'X_test': X_test_imputed
}
print("\nlinear regression done")

# ridge regression (with alpha tuning CV)
ridge_params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge_grid = GridSearchCV(
    Ridge(),
    ridge_params,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)
ridge_grid.fit(X_train_imputed, y_train)
models['Ridge Regression'] = {
    'model': ridge_grid.best_estimator_,
    'X_train': X_train_imputed,
    'X_test': X_test_imputed,
    'best_params': ridge_grid.best_params_,
    'cv_score': -ridge_grid.best_score_
}
print(f"\nbest alpha: {ridge_grid.best_params_['alpha']}")
print(f"CV MAE: {-ridge_grid.best_score_:.4f}")
print("\nridge regression done")

# random forest (imputation for NaN with median)
imputer_rf = SimpleImputer(strategy='median')
X_train_rf = pd.DataFrame(
    imputer_rf.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_rf = pd.DataFrame(
    imputer_rf.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)
# tuned to prevent overfitting
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_split=40,
    min_samples_leaf=15,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_rf, y_train)
models['Random Forest'] = {
    'model': rf,
    'X_train': X_train_rf,
    'X_test': X_test_rf
}
print("\nrandom forest done")

# models valuation
print("\nMODELS VALUATION")
results = {}
for name, model_dict in models.items():
    print(f"\n{name}")
    model = model_dict['model']
    X_tr = model_dict['X_train']
    X_te = model_dict['X_test']
    # train predictions
    y_train_pred = model.predict(X_tr)
    y_train_pred = np.maximum(y_train_pred, 1.0)
    # test predictions
    y_test_pred = model.predict(X_te)
    y_test_pred = np.maximum(y_test_pred, 1.0) 
    # train metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    # test metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    # MAE minutes convertion
    fm_pred_seconds = y_test_pred * (test_info['mh_ti'] / 21.0975) * 42.195
    fm_actual_seconds = test_info['mf_ti']
    test_mae_minutes = np.mean(np.abs(fm_actual_seconds - fm_pred_seconds)) / 60
    # MAPE
    fm_pred_seconds = y_test_pred * (test_info['mh_ti'] / 21.0975) * 42.195
    fm_actual_seconds = test_info['mf_ti']
    test_mape = np.mean(np.abs((fm_actual_seconds - fm_pred_seconds) / fm_actual_seconds)) * 100
    
    print(f"\nTRAIN SET:")
    print(f"MAE (slowdown): {train_mae:.4f}")
    print(f"RMSE (slowdown): {train_rmse:.4f}")
    print(f"R²: {train_r2:.4f}")
    print(f"\nTEST SET:")
    print(f"MAE (slowdown): {test_mae:.4f}")
    print(f"RMSE (slowdown): {test_rmse:.4f}")
    print(f"R²: {test_r2:.4f}")
    print(f"MAE (minutes): {test_mae_minutes:.2f} min")
    print(f"MAPE: {test_mape:.2f}%")
    
    results[name] = {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_mae_minutes': test_mae_minutes,
        'test_mape': test_mape,
        'y_test_pred': y_test_pred,
        'overfitting': train_r2 - test_r2
    }

# model comparison
print("\nMODEL COMPARISON")
comparison = []
for name, res in results.items():
    comparison.append({
        'model': name,
        'test R²': res['test_r2'],
        'test MAE (min)': res['test_mae_minutes'],
        'test MAPE (%)': res['test_mape'],
        'train R²': res['train_r2'],
        'overfitting': res['overfitting']
    })
comparison_df = pd.DataFrame(comparison).sort_values('test R²', ascending=False)
print("\n" + comparison_df.to_string(index=False))
best_model_name = comparison_df.iloc[0]['model']
best_model = models[best_model_name]['model']
print(f"\nbest model: {best_model_name}")
print(f"R²: {results[best_model_name]['test_r2']:.4f}")
print(f"MAE: {results[best_model_name]['test_mae_minutes']:.2f} minutes")
print(f"MAPE: {results[best_model_name]['test_mape']:.2f}%")

# feature importance
print(f"\nFEATURE IMPORTANCE - {best_model_name}")

# for random Forest
if hasattr(best_model, 'feature_importances_'):
    importance = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    print("\nfeature importance (from high to low):\n")
    print(importance_df.head(30).to_string(index=False))
    importance_df.to_csv('results/feature_importance.csv', index=False)
# for linear/ridge rergession
elif hasattr(best_model, 'coef_'):
    coef = best_model.coef_
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': coef
    }).sort_values('Coefficient', ascending=False, key=abs)
    print("\nlinear model coefficients:\n")
    print(importance_df.to_string(index=False))
    importance_df.to_csv('results/model_coefficients.csv', index=False)

# error analysis (for best model)
print("\nERROR ANALYSIS")
best_X_test = models[best_model_name]['X_test']
y_test_pred_best = results[best_model_name]['y_test_pred']
test_info_best = test_info.copy()
test_info_best['predicted_slowdown'] = y_test_pred_best
test_info_best['mh_pace_minkm'] = test_info_best['mh_ti'] / 60 / 21.0975
test_info_best['predicted_mf_seconds'] = y_test_pred_best * (test_info_best['mh_ti'] / 21.0975) * 42.195
test_info_best['actual_mf_seconds'] = test_info_best['mf_ti']
test_info_best['error_seconds'] = test_info_best['predicted_mf_seconds'] - test_info_best['actual_mf_seconds']
test_info_best['abs_error_minutes'] = abs(test_info_best['error_seconds']) / 60

# sex
print(f"\nfor sex:")
for sex, label in [(0, 'M'), (1, 'F')]:
    subset = test_info_best[test_info_best['gender'] == sex]
    if len(subset) > 0:
        mae = subset['abs_error_minutes'].mean()
        print(f"  {label}: MAE = {mae:.2f} min (n={len(subset)})")

# pace range half marathon
print(f"\nfor pace range half marathon (min/km):")
pace_bins = [0, 4.0, 4.5, 5.0, 5.5, 6.0, 100]
pace_labels = ['<4.0', '4.0-4.5', '4.5-5.0', '5.0-5.5', '5.5-6.0', '6.0+']
test_info_best['pace_bin'] = pd.cut(test_info_best['mh_pace_minkm'], bins=pace_bins, labels=pace_labels)
for pace_range in pace_labels:
    subset = test_info_best[test_info_best['pace_bin'] == pace_range]
    if len(subset) > 0:
        mae = subset['abs_error_minutes'].mean()
        print(f"  {pace_range}: MAE = {mae:.2f} min (n={len(subset)})")

# age
print(f"\nfor age:")
age_bins = [0, 30, 35, 40, 45, 100]
age_labels = ['<30', '30-35', '35-40', '40-45', '45+']
test_info_best['age_bin'] = pd.cut(test_info_best['age'], bins=age_bins, labels=age_labels)
for age_range in age_labels:
    subset = test_info_best[test_info_best['age_bin'] == age_range]
    if len(subset) > 0:
        mae = subset['abs_error_minutes'].mean()
        print(f"  {age_range}: MAE = {mae:.2f} min (n={len(subset)})")

# training volume
print(f"\nfor training volume:")
km_bins = [0, 30, 40, 50, 60, 1000]
km_labels = ['<30km', '30-40km', '40-50km', '50-60km', '60+km']
test_info_best['km_bin'] = pd.cut(test_info_best['typical'], bins=km_bins, labels=km_labels)
for km_range in km_labels:
    subset = test_info_best[test_info_best['km_bin'] == km_range]
    if len(subset) > 0:
        mae = subset['abs_error_minutes'].mean()
        print(f"  {km_range}: MAE = {mae:.2f} min (n={len(subset)})")

# 5/10K data available 
print(f"\nfor 5/10K data available:")
has_5k_test = best_X_test['has_k5_data']
has_10k_test = best_X_test['has_k10_data']
subset_5k = test_info_best[has_5k_test == 1]
if len(subset_5k) > 0:
    print(f"with 5K: MAE = {subset_5k['abs_error_minutes'].mean():.2f} min (n={len(subset_5k)})")
subset_no5k = test_info_best[has_5k_test == 0]
if len(subset_no5k) > 0:
    print(f"without 5K: MAE = {subset_no5k['abs_error_minutes'].mean():.2f} min (n={len(subset_no5k)})")
subset_10k = test_info_best[has_10k_test == 1]
if len(subset_10k) > 0:
    print(f"with 10K: MAE = {subset_10k['abs_error_minutes'].mean():.2f} min (n={len(subset_10k)})")
subset_no10k = test_info_best[has_10k_test == 0]
if len(subset_no10k) > 0:
    print(f"without 10K: MAE = {subset_no10k['abs_error_minutes'].mean():.2f} min (n={len(subset_no10k)})")

# graphs
print("\nGRAPHS")
fig = plt.figure(figsize=(16, 12))
model_names = list(results.keys())

# row 1: actual vs predicted for each model
for i, name in enumerate(model_names, 1):
    ax = plt.subplot(3, 3, i)
    y_pred = results[name]['y_test_pred']
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, c='steelblue')
    min_val = 1.0
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('actual Slowdown')
    ax.set_ylabel('predicted Slowdown')
    ax.set_title(f'{name}\nR²={results[name]["test_r2"]:.4f}')
    ax.grid(True, alpha=0.3)

# row 2: residuals for each model
for i, name in enumerate(model_names, 4):
    ax = plt.subplot(3, 3, i)
    y_pred = results[name]['y_test_pred']
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, c='coral')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('predicted Slowdown')
    ax.set_ylabel('residuals')
    ax.set_title(f'{name} - residuals')
    ax.grid(True, alpha=0.3)

# row 3: model camparison
ax7 = plt.subplot(3, 3, 7)
r2_vals = [results[m]['test_r2'] for m in model_names]
colors = ['steelblue', 'coral', 'lightgreen']
bars = ax7.bar(range(len(model_names)), r2_vals, color=colors, edgecolor='black')
ax7.set_xticks(range(len(model_names)))
ax7.set_xticklabels([n.replace(' regression', '').replace('random ', 'R') for n in model_names], 
                     rotation=0)
ax7.set_ylabel('R²')
ax7.set_title('R² comparison Test Set')

# allow negative y values
y_min = min(0, min(r2_vals))
y_max = max(r2_vals) * 1.2 if max(r2_vals) > 0 else 0.1
ax7.set_ylim([y_min, y_max])
ax7.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, r2_vals):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# error distribution
ax8 = plt.subplot(3, 3, 8)
ax8.hist(test_info_best['abs_error_minutes'], bins=30, 
         edgecolor='black', alpha=0.7, color='steelblue')
ax8.axvline(test_info_best['abs_error_minutes'].mean(), 
            color='red', linestyle='--', lw=2, 
            label=f'mean: {test_info_best["abs_error_minutes"].mean():.1f}min')
ax8.set_xlabel('absolute error (minutes)')
ax8.set_ylabel('frequency')
ax8.set_title(f'error distribution\n({best_model_name})')
ax8.legend()
ax8.grid(True, alpha=0.3)

# error vs pace half marathon
ax9 = plt.subplot(3, 3, 9)
colors_sex = ['blue' if g == 0 else 'red' for g in test_info_best['gender']]
ax9.scatter(test_info_best['mh_pace_minkm'], 
            test_info_best['abs_error_minutes'],
            alpha=0.5, s=20, c=colors_sex)
ax9.set_xlabel('half marathon pace (min/km)')
ax9.set_ylabel('absolute error (min)')
ax9.set_title('error vs pace half marathon')
ax9.grid(True, alpha=0.3)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', label='M'),
                   Patch(facecolor='red', label='F')]
ax9.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('results/model_results.png', dpi=150, bbox_inches='tight')
print("graph done - model results")

# feature importance plot
if hasattr(best_model, 'feature_importances_'):
    fig_feat, ax_feat = plt.subplots(figsize=(10, 8))
    top_features = importance_df.head(15)
    ax_feat.barh(range(len(top_features)), top_features['Importance'], 
                 color='steelblue', edgecolor='black')
    ax_feat.set_yticks(range(len(top_features)))
    ax_feat.set_yticklabels(top_features['Feature'])
    ax_feat.set_xlabel('importance')
    ax_feat.set_title(f'top 15 feature importance - {best_model_name}')
    ax_feat.invert_yaxis()
    ax_feat.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('results/feature_importance_plot.png', dpi=150, bbox_inches='tight')
    print("graph done - feature importance")
    plt.close(fig_feat)

# ridge regression learning curve
if 'Ridge Regression' in models:
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    train_scores = []
    cv_scores = []
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_imputed, y_train)
        train_pred = ridge.predict(X_train_imputed)
        train_mae = np.mean(np.abs(y_train - train_pred))
        train_scores.append(train_mae)
        cv_mae = -cross_val_score(ridge, X_train_imputed, y_train, 
                                   cv=5, scoring='neg_mean_absolute_error').mean()
        cv_scores.append(cv_mae)
    
    fig_curve, ax_curve = plt.subplots(figsize=(10, 6))
    ax_curve.plot(alphas, train_scores, 'o-', label='training MAE', 
                  linewidth=2, markersize=8, color='steelblue')
    ax_curve.plot(alphas, cv_scores, 's-', label='CV MAE', 
                  linewidth=2, markersize=8, color='coral')
    ax_curve.axvline(x=100, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_curve.set_xscale('log')
    ax_curve.set_xlabel('alpha (regularization strength)')
    ax_curve.set_ylabel('MAE (slowdown)')
    ax_curve.set_title('Ridge Regression: training vs cross-validation performance')
    ax_curve.legend()
    ax_curve.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/learning_curve_ridge.png', dpi=150, bbox_inches='tight')
    print("graph done - learning curve")
    plt.close(fig_curve)

# model saving
print("\nMODEL SAVING")
with open('best_model/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
model_info = {
    'model_name': best_model_name,
    'test_r2': results[best_model_name]['test_r2'],
    'test_mae_minutes': results[best_model_name]['test_mae_minutes'],
    'test_mape': results[best_model_name]['test_mape'],
    'feature_columns': list(X_train.columns),
    'feature_medians': X_train.median().to_dict(),
    'training_info': {
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X_train.shape[1]
    }
}
with open('best_model/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
test_info_best.to_csv('data/processed_data/test_predictions.csv', index=False)

print(f"model saved")

# trainig done
print("\nTRAINING DONE")