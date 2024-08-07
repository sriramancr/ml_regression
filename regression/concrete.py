# -*- coding: utf-8 -*-

'''
streamlit run D:/gitbuild/bosch/concrete.py
'''

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time,os

# ============
# page design
# ============
st.set_page_config(layout='wide')

if "concdata" not in st.session_state:
    st.session_state["concdata"] = None
    
if "predicted" not in st.session_state:
    st.session_state["predicted"] = False

if "predictions" not in st.session_state:
    st.session_state["predictions"] = None
    
if "mse" not in st.session_state:
    st.session_state["mse"] = -1

if "olsmodel" not in st.session_state:
    st.session_state["olsmodel"] = None

def homepage():
    # bg="D:/gitbuild/bosch/bg.jpg"
    # st.image(Image.open(bg))
    st.header("Demo : Linear Regression")
    st.divider()
    st.subheader("Concrete Compressive Strength prediction")

def dataset():
    st.session_state["predicted"] = False
    
    st.header("Dataset")
    # file="D:/gitbuild/bosch/concrete.csv"
    file = "concrete.csv"
    
    data = pd.read_csv(file)
    st.dataframe(data)

    tot = len(data)
    cols = len(data.columns) - 1
    
    st.success("Total Records = " + str(tot))
    st.success("Total Features = " + str(cols))
    
    st.session_state["concdata"] = data

def predictprice():
    
    c1,c2 = st.columns(2)
    
    if not st.session_state["predicted"]:
    
        with st.spinner("Building Regression Model and Predicting ..."):
        
            data = st.session_state["concdata"]
            data = data.rename(columns={'cementcomp':'ccomp', 'superplastisizer':'super', 'coraseaggr':'caggr','finraggr':'faggr'})
            
            if data is None:
                c1.error("Unable to load data ...")
                return
            
            trainx,testx,trainy,testy = train_test_split(data.drop("CCS",axis=1), data["CCS"],test_size=0.2)    
            trainx = sm.add_constant(trainx)
            testx = sm.add_constant(testx)
            
            m1 = sm.OLS(trainy,trainx).fit()
            p1 = np.round(m1.predict(testx),2)
            
            res = pd.DataFrame({'actual':testy,'predicted':p1})
            err = np.round(mean_squared_error(res.actual,res.predicted),2)
            
            st.session_state["olsmodel"] = m1
            st.session_state["predictions"] = res
            st.session_state["mse"] = err
            st.session_state["predicted"] = True
            time.sleep(2)
    
    res = st.session_state["predictions"]
    err = st.session_state["mse"]
    c1.subheader("Actual and Predicted values on the test dataset")
    c1.dataframe(res)
    c1.success("Mean Square Error of the Model is " + str(err))
        
    c2.subheader("Prediction on Unseen Data")
    unseen = c2.text_input("Enter the parameters, separated by a Comma",
                           placeholder="ccomp,slag,flyash,water,super,caggr,faggr,age",
                          help = "ccomp,slag,flyash,water,super,caggr,faggr,age")
    
    if c2.button("Predict"):
        m1 = st.session_state["olsmodel"]
        
        inv = unseen.split(",")
        inv = [float(i) for i in inv]
        inv.insert(0,1.0)
        inv = pd.DataFrame(inv).T
        p = m1.predict(inv)
        c2.success("Predicted CCS = " + str(np.round(p[0],2)))

    
# ==============================================
# calling each function based on the click value
# ==============================================
# main menu settings
options=[":house:",":memo:",":lower_left_fountain_pen:"]     # ,":red_circle:"]
captions=['Home','Dataset',"Regression Analysis"]            # ,"Close Application"]
nav = st.sidebar.radio("Select Option",options,captions=captions)
ndx = options.index(nav)

if (ndx==0):
    homepage()

if (ndx==1):
    dataset()
    
if (ndx==2):
    predictprice()
