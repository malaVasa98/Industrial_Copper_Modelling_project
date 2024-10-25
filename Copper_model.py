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
df_copper_ml = pd.read_csv('Copper_regress.csv')
STATUS = tuple(df_copper_ml.status.unique())
ITEM_TYPE = tuple(df_copper_ml['item type'].unique())
COUNTRY = tuple(df_copper_ml.country.unique())
APPLICATION = tuple(df_copper_ml.application.unique())
PRODUCT_REF = tuple(df_copper_ml.product_ref.unique())

# Loading regression model
with open('rf_reg.pkl','rb') as reg_fl:
    best_rf = pickle.load(reg_fl)

# Loading label encoder
with open('label_encode_fl.pkl','rb') as label:
    label_encode = pickle.load(label)
          
# Loading ordinal encoder
with open('ordinal_encode_fl.pkl','rb') as ordi:
    ordinal_encode = pickle.load(ordi)

# Loading scaler
with open('scale.pkl','rb') as scal:
    scale = pickle.load(scal)

# Loading classification model
with open('classfy_model.pkl','rb') as clasf:
    best_etc = pickle.load(clasf)

if selected=="Predict Selling Price":
    col1,col2 = st.columns(2)
    with col1:
        status_sp = st.selectbox('**Select Status**',STATUS,index=None,key='status_reg',placeholder='Select one')
        item_type_sp = st.selectbox('**Select Item type**',ITEM_TYPE,index=None,key='item_type_reg',placeholder='Select one')
        customer_sp = st.text_input('**Customer ID**',placeholder='Eg:30156308.0',key='customer_reg')
        country_sp = st.selectbox('**Select Country**',COUNTRY,index=None,key='country_reg',placeholder='Select one')
        application_sp = st.selectbox('**Select Application**',APPLICATION,index=None,key='application_reg',placeholder='Select one')
    with col2:
        product_refsp = st.selectbox('**Select Product Reference**',PRODUCT_REF,index=None,key='prod_reg',placeholder='Select one')
        quantity_sp = st.text_input('**Quantity (in tonnes)**',key='quant_reg')
        thickness_sp = st.text_input('**Thickness**',key='thick_reg')
        width_sp = st.text_input('**Width**',key='width_reg')
        if st.button('Get Selling Price'):
            quant_log = np.log(float(quantity_sp))
            thick_log = np.log(float(thickness_sp))
            X_sell = np.array([[status_sp,item_type_sp,float(customer_sp),float(country_sp),float(application_sp),float(product_refsp),quant_log,thick_log,float(width_sp)]])
            X_sell[:,[1]]=label_encode.transform(np.ravel(X_sell[:,[1]]))
            X_sell[:,[0]]=ordinal_encode.transform(X_sell[:,[0]])
            X_sell = scale.transform(X_sell)
            Y_sell = best_rf.predict(X_sell)
            st.write(f'${np.around(np.exp(Y_sell[0]),decimals=2)}')
            
if selected=="Predict Status":
    col1,col2=st.columns(2)
    with col1:
        customer_st = st.text_input('**Customer ID**',placeholder='Eg:30156308.0',key='customer_clas')
        country_st = st.selectbox('**Select Country**',COUNTRY,index=None,key='country_clas',placeholder='Select one')
        application_st = st.selectbox('**Select Application**',APPLICATION,index=None,key='application_clas',placeholder='Select one')
        product_refst = st.selectbox('**Select Product Reference**',PRODUCT_REF,index=None,key='prod_clas',placeholder='Select one')
        quantity_st = st.text_input('**Quantity (in tonnes)**',key='quant_clas')
    with col2:
        thickness_st = st.text_input('**Thickness**',key='thick_clas')
        width_st = st.text_input('**Width**',key='width_clas')
        selling_pricest = st.text_input('**Selling Price**',key='sell_clas')
        item_type_st = st.selectbox('**Select Item type**',ITEM_TYPE,index=None,key='item_type_clas',placeholder='Select one')
        if st.button('Get Status'):
            quant_log_st = np.log(float(quantity_st))
            thick_log_st = np.log(float(thickness_st))
            sell_log = np.log(float(selling_pricest))
            X_stat = np.array([[float(customer_st),float(country_st),float(application_st),float(product_refst),quant_log_st,thick_log_st,float(width_st),sell_log,item_type_st]])
            X_stat[:,[-1]]=label_encode.transform(np.ravel(X_stat[:,[-1]]))
            X_stat = scale.transform(X_stat)
            Y_stat = best_etc.predict(X_stat)
            if Y_stat == 1:
                st.write("Won")
            else:
                st.write("Lost")
            
            
        
        
        
