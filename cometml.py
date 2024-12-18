import joblib
from comet_ml import Experiment
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import pandas as pd
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt


load_dotenv()

X_test_transformed = pd.read_csv('data/X_test_transformed.csv')
y_test = pd.read_csv('data/y_test.csv')

random_forest_model = joblib.load('models/random_forest_model.pkl')
ridge_model = joblib.load('models/ridge_model.pkl')
gradient_boosting_model = joblib.load('models/gradient_boosting_model.pkl')

experiment = Experiment(
  api_key=os.getenv("COMET_API_KEY"),
  project_name=os.getenv("COMET_PROJECT_NAME"),
  workspace=os.getenv("COMET_WORKSPACE")
)
experiment.set_name("imdb_rating_prediction")


# Metrics
rf_pred = random_forest_model.predict(X_test_transformed)
ridge_pred = ridge_model.predict(X_test_transformed)
gb_pred = gradient_boosting_model.predict(X_test_transformed)

# MSE
rf_mse = mean_squared_error(y_test, rf_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)
gb_mse = mean_squared_error(y_test, gb_pred)

experiment.log_metric("random_forest_mse", rf_mse)
experiment.log_metric("ridge_mse", ridge_mse)
experiment.log_metric("gradient_boosting_mse", gb_mse)

# R2
rf_r2 = r2_score(y_test, rf_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
gb_r2 = r2_score(y_test, gb_pred)

experiment.log_metric("random_forest_r2", rf_r2)
experiment.log_metric("ridge_r2", ridge_r2)
experiment.log_metric("gradient_boosting_r2", gb_r2)

# Explained Variance Score
rf_evs = explained_variance_score(y_test, rf_pred)
ridge_evs = explained_variance_score(y_test, ridge_pred)
gb_evs = explained_variance_score(y_test, gb_pred)

experiment.log_metric("random_forest_evs", rf_evs)
experiment.log_metric("ridge_evs", ridge_evs)
experiment.log_metric("gradient_boosting_evs", gb_evs)


# Graphs
models = ['Random Forest', 'Ridge', 'Gradient Boosting']
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# MSE Graph
mse_values = [rf_mse, ridge_mse, gb_mse]
ax[0].bar(models, mse_values, color=['blue', 'red', 'green'])
ax[0].set_title('MSE')
ax[0].set_xlabel('Model')
ax[0].set_ylabel('Mean Squared Error')

# R2 Graph
r2_values = [rf_r2, ridge_r2, gb_r2]
ax[1].bar(models, r2_values, color=['blue', 'red', 'green'])
ax[1].set_title('R2')
ax[1].set_xlabel('Model')
ax[1].set_ylabel('R-squared')

# Explained Variance Score Graph
evs_values = [rf_evs, ridge_evs, gb_evs]
ax[2].bar(models, evs_values, color=['blue', 'red', 'green'])
ax[2].set_title('Explained Variance')
ax[2].set_xlabel('Model')
ax[2].set_ylabel('Explained Variance Score')

experiment.log_figure(figure=fig)


experiment.end()

