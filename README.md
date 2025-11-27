# marathon prediction model
Machine learning system that predicts marathon finish times from half-marathon performance for recreational runners. The model predicts the "slowdown factor" so how much slower a runner's marathon pace will be compared to their half-marathon pace. It uses 30 engineered features including race times, demographics, and training characteristics. It implements a user friendly interface where you can input your data and predict your own amrathon finish time.

# repo structure
- `code/` --> python scripts: `prepare_data.py`, `train_model.py`, `predict_marathon.py`
- `data/` --> raw dataset and processed train/test splits
- `best_model/` --> trained random forest model
- `results/` --> best model, visualizations and feature importance plots
- `main.py` --> main script to run

# run
- install dependencies: `pip install -r requirements.txt`
- run main script: `main.py`

# results
The Random Forest model achieves R² = 0.1118 and MAE = 8.03 minutes, outperforming Linear and Ridge Regression. Model performs better for male runners, faster paces, and higher training volumes.
