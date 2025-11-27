# proposal
As an amateur runner, I regularly train and race over short and medium distances, but I have never gone beyond 20 kilometers. This made me wonder: if I can run a strong half-marathon, how fast could I finish a full marathon? It's from this personal curiosity that the idea for the project was born.

The project goal is to develop a machine learning model that predicts marathon finish times from half marathon performance for recreational runners. The model predicts the "slowdown factor" so how much slower a runner's marathon pace will be compared to their half-marathon pace.

The dataset is a survey-based dataset from Vickers & Vertosick (2016) and includes demographics (age, sex, BMI), training characteristics (weekly volume, training types), and actual race times.

The ML methods compare three regression models: Multiple Linear Regression, Ridge Regression, and Random Forest Regressor. Validation uses train/test split by athlete to ensure generalization. In addition, there is a baseline given by the classic Riegel formula to compare the results of the regressions with more traditional ones. 

The deliverable is a user-friendly interactive predictor where runners can input their data (race times, demographics, training) and receive personalized marathon time predictions with calibrated error bands.