# Importing the necessary packages
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Setting up the streamlit page
st.title(":green[INDUSTRIAL COPPER MODELLING]")
with st.sidebar:
    selected = option_menu('Menu',["About","Predict Selling Price","Predict Status"])

if selected=='About':
    st.write("Here, we have considered the data from a copper industry related to sales and pricing. The data is highly skewed and noisy, so some pre-processing is done to address them. With this data, we utilised a regression model (Random Forest Regressor) to make predictions on the selling price and a classification model (Extra Trees Classifier) to make predictions on the status.")

# Loading the required data
df_copper_data = pd.read_csv('Copper_Refined.csv',dtype={
 'id':'object',
 'quantity tons':'float64',
 'customer':'object',
 'country':'object',
 'status':'object',
 'item type':'object',
 'application':'object',
 'thickness':'float64',
 'width':'float64',
 'material_ref':'object',
 'product_ref':'object',
 'selling_price':'float64'},parse_dates=['item_date','delivery date'])
STATUS = tuple(df_copper_data.status.unique())
ITEM_TYPE = tuple(df_copper_data['item type'].unique())
COUNTRY = tuple(df_copper_data.country.unique())
APPLICATION = tuple(df_copper_data.application.unique())


# Loading regression model
with open('rf_reg.pkl','rb') as reg_fl:
    best_rf = pickle.load(reg_fl)

# Loading one hot encoder - regression
with open('ohe_cop.pkl','rb') as one_hot:
    ohe_copper = pickle.load(one_hot)
          
# Loading scaler - regression
with open('scale.pkl','rb') as scal:
    scale = pickle.load(scal)

# Loading classification model
with open('classfy_model.pkl','rb') as clasf:
    best_etc = pickle.load(clasf)

# Loading one hot encoder - classification
with open('ohe_class.pkl','rb') as one_hot_cl:
    ohe_copper_cl = pickle.load(one_hot_cl)
    
# Loading scaler - classification
with open('scale_class.pkl','rb') as scl_f:
    scale_cl = pickle.load(scl_f)

if selected=="Predict Selling Price":
    col1,col2 = st.columns(2)
    with col1:
        quantity_sp = st.text_input('**Quantity (in tonnes)**',key='quant_reg')
        thickness_sp = st.text_input('**Thickness**',key='thick_reg')
        width_sp = st.text_input('**Width**',key='width_reg')
        country_sp = st.selectbox('**Select Country**',COUNTRY,index=None,key='country_reg',placeholder='Select one')
    with col2:
        status_sp = st.selectbox('**Select Status**',STATUS,index=None,key='status_reg',placeholder='Select one')
        application_sp = st.selectbox('**Select Application**',APPLICATION,index=None,key='application_reg',placeholder='Select one')
        item_type_sp = st.selectbox('**Select Item type**',ITEM_TYPE,index=None,key='item_type_reg',placeholder='Select one')
        if st.button('Get Selling Price'):
            quant_log = np.log(float(quantity_sp))
            thick_log = np.log(float(thickness_sp))
            X_user = np.array([[quant_log,thick_log,float(width_sp),country_sp,status_sp,application_sp,item_type_sp]])
            X_user_oh = ohe_copper.transform(X_user[:,[3,4,5,6]])
            X_user1 = np.concatenate((X_user[:,[0,1,2]],X_user_oh),axis=1)
            X_user2 = scale.transform(X_user1)
            y_pred = best_rf.predict(X_user2)
            st.write(f'${np.around(np.exp(y_pred[0]),decimals=2)}')
            
if selected=="Predict Status":
    col1,col2=st.columns(2)
    with col1:
        quantity_st = st.text_input('**Quantity (in tonnes)**',key='quant_clas')
        thickness_st = st.text_input('**Thickness**',key='thick_clas')
        width_st = st.text_input('**Width**',key='width_clas')
        selling_pricest = st.text_input('**Selling Price**',key='sell_clas')
    with col2:
        country_st = st.selectbox('**Select Country**',COUNTRY,index=None,key='country_clas',placeholder='Select one')
        application_st = st.selectbox('**Select Application**',APPLICATION,index=None,key='application_clas',placeholder='Select one')
        item_type_st = st.selectbox('**Select Item type**',ITEM_TYPE,index=None,key='item_type_clas',placeholder='Select one')
        if st.button('Get Status'):
            quant_log_st = np.log(float(quantity_st))
            thick_log_st = np.log(float(thickness_st))
            sell_log = np.log(float(selling_pricest))
            X_val = np.array([[quant_log_st,thick_log_st,float(width_st),sell_log,country_st,application_st,item_type_st]])
            X_val_oh = ohe_copper_cl.transform(X_val[:,[4,5,6]])
            X_val1 = np.concatenate((X_val[:,[0,1,2,3]],X_val_oh),axis=1)
            X_val2 = scale_cl.transform(X_val1)
            Y_sp = best_etc.predict(X_val2)
            if Y_sp == 1:
                st.write("Won")
            else:
                st.write("Lost")
            
           
            
            
        
        
        
