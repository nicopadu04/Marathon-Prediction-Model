# AI usage

The main idea of the project, the dataset selection, the three regression models choice and the evaluation and interpretation were entirely done by me.

The prepare_data.py script was written almost entirely by me: data cleaning, feature engineering and construction of the slowdown target are my work. I used AI (mainly Claude) only for two things: the idea of using a stratified train/test split to preserve the male/female ratio, and the suggestion to export a more complete set of CSV files (X/y, info, feature list).

In train_model.py I designed the overall structure myself, with the three models, the GridSearchCV for Ridge and Random Forest and the conversion of slowdown errors into minutes to be more interpretable by end users. I used Claude only for the technical implementation of NaN handling and median imputation, and to improve the aesthetics and organization of the plots because I neededo to reduce space as much as possibile. Nevertheless, the choice of metrics, interpretation of the results and error analysis by subgroups are obviously all mine.

The predict_marathon.py script is also almost all my own work. I designed the interactive predictor, all the validation logic on the inputs ( such as range checks, retryng for invalid values ecc.), and I manually replicated the same feature engineering as in the training phase. The only part directly suggested by AI was the idea of a small helper function parse_float_input that accepts both comma and dot as decimal separators, which I didn't think about and wich I then integrated into the script.

For the report, I used ChatGPT only to revise grammar and improve the clarity of some sentences that were not easy to understand, but all the structure, content choices, interpretations of the results, and discussion points come from me.

Overall, these AI tools were used as debugging and editing aids, and as stated in the guidelines I did not submit any code I did not understand. All modelling decisions, experiments and final implementations are under my control, and I am happy about this because I learned a lot on the academic side, and I now also have a marathon predictor I can use for myself and my friends whenever I need it.