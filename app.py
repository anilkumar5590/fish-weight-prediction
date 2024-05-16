import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(
        page_title="Fish Weight Prediction",
)
# Load the dataset
fish_df = pd.read_csv("Fish_dataset.csv")
option_values = fish_df.iloc[:, 1].unique()

category={
    "Bream":1,
    "Roach":5,
    "Whitefish":7,
    "Parkki":2,
    "Perch":3,
    "Pike":4,
    "Smelt":6,
}

st.title('Fish Weight Prediciton')
# Define the input fields using Streamlit's number_input
species=st.selectbox('Species',option_values)
height=st.number_input('Height',placeholder="Enter the height in cm",value=None,step=0.1)
width=st.number_input('Width',placeholder="Enter the diagonal width in cm",value=None,step=0.1)
length1=st.number_input('Length1',placeholder="Enter the vertical length in cm",value=None,step=0.1)
length2=st.number_input('Length2',placeholder="Enter the diagonal length in cm",value=None,step=0.1)
length3=st.number_input('Length3',placeholder="Enter the cross length in cm",value=None,step=0.1)

# Separate the features (X) and target variable (y)

y = fish_df['Weight']
X = fish_df[['Category','Height', 'Width', 'Length1', 'Length2', 'Length3']]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2529)

# Create a RandomForestRegressor model instance and fit it to the training data
random_forest_regressor_model = RandomForestRegressor(random_state=42)
random_forest_regressor_model.fit(X_train, y_train)

# Collect user input into a list
user_input = [category[species],height,width,length1,length2,length3]

# Make prediction when the button is clicked
if st.button('Make Prediction'):
    # Check if any input field is empty
    flag = 0
    for i in user_input:
        if i is None:
            flag = 1
            break
    if flag:
        # Display error message if any input field is empty
        st.error('⚠️ Please fill all the above fields ')
    else:  
        # Make prediction using the Random Forest model
        prediction = random_forest_regressor_model.predict([user_input])
        st.success(f"The predicted fish weight is  {int(prediction[0])} grams")


# Footer content
footer_html = """
<hr>
<div style="bottom: 0;  color: green; text-align: center;">
    <p style="font-weight: bold; ">Developed by Anil Kumar</p>
</div>
"""

# Display the footer using markdown
st.markdown(footer_html, unsafe_allow_html=True)