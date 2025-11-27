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

# riegel baseline
print("\nRIEGEL BASELINE")
D_hm = 21.0975
D_mar = 42.195
p_riegel = 1.06
riegel_factor = (D_mar / D_hm) ** p_riegel
train_info['mf_pred_riegel'] = train_info['mh_ti'] * riegel_factor
test_info['mf_pred_riegel'] = test_info['mh_ti'] * riegel_factor
print(f"riegel exponent: p = {p_riegel}")
print(f"riegel factor: {riegel_factor:.4f}")

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
# gridsearch 
rf_params = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [20, 30, 40, 50],
    'min_samples_leaf': [10, 15, 20, 25],
    'max_features': ['sqrt']
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_params,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1  
)
rf_grid.fit(X_train_rf, y_train)
print(f"\nbest parameters: {rf_grid.best_params_}")
print(f"CV MAE: {-rf_grid.best_score_:.4f}")
models['Random Forest'] = {
    'model': rf_grid.best_estimator_,
    'X_train': X_train_rf,
    'X_test': X_test_rf,
    'best_params': rf_grid.best_params_,
    'cv_score': -rf_grid.best_score_
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

# riegel baseline evaluation
print(f"\nRiegel Baseline (p={p_riegel})")
mf_pred_riegel_test = test_info['mf_pred_riegel'].values
mf_actual_test = test_info['mf_ti'].values
mh_pace_test = test_info['mh_ti'] / 60 / 21.0975
mf_pace_riegel = mf_pred_riegel_test / 60 / 42.195
slowdown_riegel = mf_pace_riegel / mh_pace_test
# calcola R² sullo slowdown (come gli altri modelli)
slowdown_actual = y_test.values
test_r2_riegel = r2_score(slowdown_actual, slowdown_riegel)
test_mae_riegel_slowdown = mean_absolute_error(slowdown_actual, slowdown_riegel)
test_mae_riegel_minutes = np.mean(np.abs(mf_actual_test - mf_pred_riegel_test)) / 60
test_mape_riegel = np.mean(np.abs((mf_actual_test - mf_pred_riegel_test) / mf_actual_test)) * 100
print(f"\nTEST SET:")
print(f"MAE (slowdown): {test_mae_riegel_slowdown:.4f}")
print(f"R² (slowdown): {test_r2_riegel:.4f}")
print(f"MAE (minutes): {test_mae_riegel_minutes:.2f} min")
print(f"MAPE: {test_mape_riegel:.2f}%")
results['Riegel Baseline'] = {
    'test_mae_minutes': test_mae_riegel_minutes,
    'test_mape': test_mape_riegel,
    'test_r2': test_r2_riegel,
    'test_mae': test_mae_riegel_slowdown,
    'y_test_pred': slowdown_riegel,
    'train_r2': np.nan,
    'train_mae': np.nan,
    'train_rmse': np.nan,
    'test_rmse': np.nan,
    'overfitting': np.nan
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
        'train R²': res.get('train_r2', np.nan),
        'overfitting': res.get('overfitting', np.nan)
    })
comparison_df = pd.DataFrame(comparison).sort_values('test R²', ascending=False)
print("\n" + comparison_df.to_string(index=False))

# best model (exclude baseline)
ml_models_only = comparison_df[~comparison_df['model'].str.contains('Baseline')]
best_model_name = ml_models_only.iloc[0]['model']
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
fig = plt.figure(figsize=(18, 14))

# ordine modelli: RF, Ridge, Linear (poi Riegel solo nelle barre)
ml_models = ['Random Forest', 'Ridge Regression', 'Linear Regression']

# row 1: actual vs predicted (solo ML models)
for i, name in enumerate(ml_models, 1):
    ax = plt.subplot(3, 3, i)
    y_pred = results[name]['y_test_pred']
    ax.scatter(y_test, y_pred, alpha=0.6, s=30, c='steelblue', edgecolors='none')
    min_val = 1.0
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5)
    ax.set_xlabel('Actual Slowdown', fontsize=10)
    ax.set_ylabel('Predicted Slowdown', fontsize=10)
    short_name = name.replace(' Regression', '')
    ax.set_title(f'{short_name}\nR²={results[name]["test_r2"]:.4f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

# row 2: residuals (solo ML models)
for i, name in enumerate(ml_models, 4):
    ax = plt.subplot(3, 3, i)
    y_pred = results[name]['y_test_pred']
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.6, s=30, c='coral', edgecolors='none')
    ax.axhline(y=0, color='r', linestyle='--', lw=2.5)
    ax.set_xlabel('Predicted Slowdown', fontsize=10)
    ax.set_ylabel('Residuals', fontsize=10)
    short_name = name.replace(' Regression', '')
    ax.set_title(f'{short_name} - Residuals', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

# row 3, col 1: R² comparison (tutti i 4 modelli)
ax7 = plt.subplot(3, 3, 7)
models_for_bar = ['Random Forest', 'Ridge Regression', 'Linear Regression', 'Riegel Baseline']
r2_vals = [results[m]['test_r2'] for m in models_for_bar]
colors_bar = ['lightgreen', 'coral', 'steelblue', 'orange']
x_pos = range(len(models_for_bar))
bars = ax7.bar(x_pos, r2_vals, color=colors_bar, edgecolor='black', linewidth=1.5)
ax7.set_xticks(x_pos)
short_labels = ['Random\nForest', 'Ridge', 'Linear', 'Riegel']
ax7.set_xticklabels(short_labels, fontsize=9)
ax7.set_ylabel('R² Score', fontsize=10)
ax7.set_title('R² Comparison', fontsize=11, fontweight='bold')
# limita y-axis per non far dominare riegel
ml_r2_vals = r2_vals[:3]  # solo ML models
y_min = min(min(ml_r2_vals), 0) - 0.05
y_max = max(ml_r2_vals) + 0.05
ax7.set_ylim([y_min, y_max])
ax7.grid(True, alpha=0.3, axis='y')
ax7.tick_params(labelsize=9)
# aggiungi valori sopra le barre
for bar, val in zip(bars, r2_vals):
    if val >= y_min and val <= y_max:  # solo se dentro range
        y_pos = val + 0.01 if val > 0 else val - 0.02
        va = 'bottom' if val > 0 else 'top'
        ax7.text(bar.get_x() + bar.get_width()/2, y_pos,
                 f'{val:.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

# row 3, col 2: MAE comparison (tutti i 4 modelli)
ax8 = plt.subplot(3, 3, 8)
mae_vals = [results[m]['test_mae_minutes'] for m in models_for_bar]
bars = ax8.bar(x_pos, mae_vals, color=colors_bar, edgecolor='black', linewidth=1.5)
ax8.set_xticks(x_pos)
ax8.set_xticklabels(short_labels, fontsize=9)
ax8.set_ylabel('MAE (minutes)', fontsize=10)
ax8.set_title('MAE Comparison', fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')
ax8.tick_params(labelsize=9)
for bar, val in zip(bars, mae_vals):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# row 3, col 3: error distribution (best model)
ax9 = plt.subplot(3, 3, 9)
ax9.hist(test_info_best['abs_error_minutes'], bins=25, 
         edgecolor='black', alpha=0.75, color='steelblue', linewidth=1)
mean_error = test_info_best['abs_error_minutes'].mean()
ax9.axvline(mean_error, color='red', linestyle='--', lw=2.5, 
            label=f'Mean: {mean_error:.1f}min')
ax9.set_xlabel('Absolute Error (min)', fontsize=10)
ax9.set_ylabel('Frequency', fontsize=10)
ax9.set_title(f'Error Distribution\n({best_model_name})', 
              fontsize=11, fontweight='bold')
ax9.legend(fontsize=9, loc='upper right')
ax9.grid(True, alpha=0.3)
ax9.tick_params(labelsize=9)

plt.subplots_adjust(hspace=0.40, wspace=0.35)
plt.savefig('results/model_results.png', dpi=300, bbox_inches='tight')
print("graph done - model results")
plt.close(fig)

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
with open('results/best_model/best_model.pkl', 'wb') as f:
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
with open('results/best_model/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
test_info_best.to_csv('data/processed_data/test_predictions.csv', index=False)

print(f"model saved")

# trainig done
print("\nTRAINING DONE")