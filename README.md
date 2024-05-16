# Hi, I'm Anil Kumar! 👋

## Fish Weight Prediction App
This Streamlit application predicts the weight of fish based on user inputs for species, height, width, and length dimensions using Random Forest Regression.

## Features
1. `Input Fields`: Users can select the fish species and enter the height, width, and length dimensions of the fish.
2. `Prediction`: Upon clicking the "Make Prediction" button, the app predicts the weight of the fish using a Random Forest Regression model.
3. `Error Handling`: The app checks for empty input fields and displays an error message if any field is left empty.
4. `Interactive Visualization`: The app includes interactive scatter plots to visualize the relationship between true and predicted fish weights.
5. `Accuracy Metrics`: MAE, MSE, and R-squared (R2) score are calculated and displayed for model evaluation.
6. `Responsive Design`: The app is designed to be responsive and user-friendly.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your_username/your_repository.git
```
2. Navigate to the project directory
```bash
cd your_repository
```
3. Install the required packages:
```bash
pip install streamlit pandas scikit-learn
```
or
```bash
pip install -r requirements.txt
```
## Usage
1. Run the Streamlit app:
```bash
streamlit run app.py
```
If the above command does not works then try the below code
```bash
python -m streamlit run app.py
```
2. Access the app in your web browser at http://localhost:8501.

3. Select the fish species and enter the height, width, and length dimensions in the input fields.

4. Click the "Make Prediction" button to see the predicted weight of the fish.

## Files Included
- `app.py`: The main Streamlit application script.
- `Fish_dataset.csv`: Dataset containing fish data for training and testing.
- `requirements.txt`: List of required Python packages.
- `Fish_Weigth_Prediction.ipynb`: Jupyter Notebook containing the data analysis and model training steps.

## Input Fields
1. Species: Select the species of the fish from the dropdown menu.
2. Height: Enter the height of the fish in centimeters.
3. Width: Enter the diagonal width of the fish in centimeters.
4. Length1: Enter the vertical length of the fish in centimeters.
5. Length2: Enter the diagonal length of the fish in centimeters.
6. Length3: Enter the cross length of the fish in centimeters.

## Dependencies

1. `Streamlit`: Streamlit is a Python library used for building interactive web applications. It provides simple APIs for creating user interfaces and visualizations.
```bash
import streamlit as st
```
2. `Pandas`: Pandas is a popular data manipulation and analysis library in Python. It is used for reading and processing data from various sources, such as CSV files.
```bash
import pandas as pd
```
3. `scikit-learn (sklearn)`: scikit-learn is a machine learning library in Python that provides various algorithms and tools for machine learning tasks such as classification, regression, clustering, and more.
```bash
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
```
4. `RandomForestRegressor`: RandomForestRegressor is a regression algorithm from the scikit-learn library. It is used to train a random forest model for predicting numerical outcomes based on input features.
5. `train_test_split`: train_test_split is a function from scikit-learn used for splitting the dataset into training and testing sets. It helps evaluate the model\'s performance on unseen data.

## Preview
[Checkout Here](https://fish-weight-prediction-by-anil79.streamlit.app/)

## 🔗 Follow us
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anilkumarkonathala/)

## Feedback
If you have any feedback, please reach out to us at konathalaanilkumar143@gmail.com
