import streamlit as st
import pandas as pd
import pickle as pkl
import gzip
import numpy as np
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
#import scikit_learn
#from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Zomato App",layout="centered",initial_sidebar_state="expanded")

st.title('Zomato App')
 # front end elements of the web page 
html_temp = """ 
    <div style ="background-color:pink;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Predicting success of new restaurants</h1> 
    </div> 
    """
      
# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True) 
st.subheader('by Salma Reda ')

#take input from user     
# following lines create boxes in which user can enter data required to make prediction

location = st.selectbox('Name of Location', ('Banashankari', 'Basavanagudi', 'Mysore Road', 'Jayanagar',
       'Kumaraswamy Layout', 'Rajarajeshwari Nagar', 'Vijay Nagar',
       'Uttarahalli', 'JP Nagar', 'South Bangalore', 'City Market',
       'Bannerghatta Road', 'BTM', 'Kanakapura Road', 'Bommanahalli',
       'Electronic City', 'Sarjapur Road', 'Wilson Garden',
       'Shanti Nagar', 'Koramangala 5th Block', 'Richmond Road', 'HSR',
       'Koramangala 7th Block', 'Bellandur', 'Marathahalli', 'Whitefield',
       'East Bangalore', 'Old Airport Road', 'Indiranagar',
       'Koramangala 1st Block', 'Frazer Town', 'MG Road', 'Brigade Road',
       'Lavelle Road', 'Church Street', 'Ulsoor', 'Residency Road',
       'Shivajinagar', 'Infantry Road', 'St. Marks Road',
       'Cunningham Road', 'Race Course Road', 'Commercial Street',
       'Vasanth Nagar', 'Domlur', 'Koramangala 8th Block', 'Ejipura',
       'Jeevan Bhima Nagar', 'Old Madras Road', 'Seshadripuram',
       'Kammanahalli', 'Koramangala 6th Block', 'Majestic',
       'Langford Town', 'Central Bangalore', 'Brookefield',
       'ITPL Main Road, Whitefield', 'Varthur Main Road, Whitefield',
       'Koramangala 2nd Block', 'Koramangala 3rd Block',
       'Koramangala 4th Block', 'Koramangala', 'Hosur Road', 'RT Nagar',
       'Banaswadi', 'North Bangalore', 'Nagawara', 'Hennur',
       'Kalyan Nagar', 'HBR Layout', 'Rammurthy Nagar', 'Thippasandra',
       'CV Raman Nagar', 'Kaggadasapura', 'Kengeri', 'Sankey Road',
       'Malleshwaram', 'Sanjay Nagar', 'Sadashiv Nagar',
       'Basaveshwara Nagar', 'Rajajinagar', 'Yeshwantpur', 'New BEL Road',
       'West Bangalore', 'Magadi Road', 'Yelahanka', 'Sahakara Nagar',
       'Jalahalli', 'Hebbal', 'Nagarbhavi', 'Peenya', 'KR Puram'))

type_of_name = st.multiselect('Type', ('Restaurant', 'Cafe', 'Hotel'))


rest_type_ge = st.multiselect('type of restaurant',('Casual Dining', 'Cafe', 'Quick Bites', 'Delivery', 'Mess',
       'Dessert Parlor', 'Bakery', 'Pub', 'Fine Dining', 'Beverage Shop',
       'Sweet Shop', 'Bar', 'Kiosk', 'Food Truck', 'Microbrewery',
       'Lounge', 'Food Court', 'Dhaba', 'Club', 'Confectionery',
       'Bhojanalya'))


#options 
st.subheader('What are the services that you will provide? ')

online_order = st.radio("Online ordering is available?: ", ('yes', 'no'))
book_table = st.radio('Book Table is available',("yes","no")) 


# Transform selected options to numerical values
online_order_value = 1 if online_order == 'yes' else 0
book_table_value = 1 if book_table == 'yes' else 0

cost=st.slider('Cost For Two',1,10000,100)

count_cuisines = st.slider("Number Of Cuisines ",1,8)


st.sidebar.subheader("About App")

st.sidebar.info("This web app is helps you to find out whether your restaurant is going to successed or not.")
st.sidebar.info("Enter the required fields and click on the 'Predict' button to check whether your restaurant is going to successed or not")
st.sidebar.info("Don't forget to rate this app")



feedback = st.sidebar.slider('How much would you rate this app?',min_value=0,max_value=5,step=1)
if feedback:
    st.header("Thank you for rating the app!")
    st.info("Caution: This is just a prediction.") 


df_new = pd.DataFrame({'location': [location], 'type_of_name':[type_of_name], "rest_type_ge": [rest_type_ge], 'online_order': [online_order_value], 'book_table':[book_table_value], 'cost': [cost], 'count_cuisines': [count_cuisines]})
# load transformer
def decompress_file(input_file, output_file):
     with gzip.open(input_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                f_out.write(f_in.read())


# Usage example
input_file = 'zomato_transformer.pkl.gz'
decompressed_file = 'zomato_transformer.pkl'
decompress_file(input_file, decompressed_file)

# Load the transformer 
transformer = pkl.load(open(decompressed_file, 'rb'))

st.success(f"File '{input_file}' has been decompressed to '{decompressed_file}'")

# apply transformer on inputs
x_new = transformer.transform(df_new)
#transformed_data = transformer.transform(df_new)


# Define a custom transformer to handle columns with lists
def list_transformer(column):
    return np.array(column.tolist())



# Usage example
input_file = 'zomato.pkl.gz'
decompressed_file = 'zomato.pkl'
decompress_file(input_file, decompressed_file)
# load model 
st.success(f"File '{input_file}' has been decompressed to '{decompressed_file}'")

                     
loaded_model = pkl.load(open('decompressed_file.pkl', 'rb'))


#predict the output
predict= decompressed_file.predict(x_new)[0]


if st.button("Predict"):
    if pred[0] == 0:
        st.error('Warning! this restaurant is not probably going to succeed!')
    else:
        st.success('wow, good luck with this successeful resaurant!')
    
#predict output
success_proba= model.predict_proba(X_new)[0][1]*100
