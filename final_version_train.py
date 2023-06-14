from data_transform import transform_function
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, FunctionTransformer
import xgboost as xgb

train_X, train_Y = transform_function()
test_ids = pd.read_csv('data/test.csv')['id']
################################################################### train
X, Y = train_X, train_Y.reshape(-1)

# Define the parameter grid for grid search
param_grid = {
    'max_depth': [7],
    'learning_rate': [0.1],
    'n_estimators': [200],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}
label_transform = MinMaxScaler
def custom_scorer(y_true, y_pred):
    y_pred = label_transform.inverse_transform([y_pred])[0]
    y_pred = np.clip(np.round(y_pred), 0, 9)
    y_true = label_transform.inverse_transform([y_true])[0]
    print(mean_absolute_error(y_true, y_pred))
    return -mean_absolute_error(y_true, y_pred)

# Create an instance of the HistGradientBoostingRegressor
model = xgb.XGBRegressor

# Perform grid search with cross-validation
scoring = make_scorer(custom_scorer)
grid_search = GridSearchCV(model(), param_grid, scoring=scoring, cv=5)
grid_search.fit(X, Y)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = -grid_search.best_score_

# Print the best parameters and score
print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Create a new instance of HistGradientBoostingRegressor with the best parameters
trained_model = model(**best_params)

# Fit the best HistGradientBoostingRegressor on the entire dataset
trained_model = trained_model.fit(X, Y)
# train loss
y_pred = trained_model.predict(X)
train_err = custom_scorer(Y, y_pred)

print(f'train loss: {train_err}')

################################################################### predict 
# Make predictions on the test set
test_predictions = trained_model.predict(train_X)
test_predictions = label_transform.inverse_transform([test_predictions])[0]
test_predictions = np.clip(np.round(test_predictions), 0, 9)
test_predictions = test_predictions.astype(int).reshape(-1)

# Prepare the submission dataframe
submission_df = pd.DataFrame({'id': test_ids, 'Danceability': test_predictions})

# Save the submission to a CSV file
submission_df.to_csv('submission.csv', index=False)

plt.hist(test_predictions, bins=10)
