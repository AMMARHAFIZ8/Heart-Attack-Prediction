# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:18:29 2022

@author: ACER
"""

import pickle
import os
import numpy as np
import streamlit as st


MODEL_PATH = os.path.join(os.getcwd(),'best_pipeline.pkl') 

with open(MODEL_PATH,'rb') as file:
    model= pickle.load(file)

# outcome_dict[outcome[0]]
with st.form("Patient's info"):
    st.title('Beware of Heart Attack!')
    st.write("This app is to predict chances of a person having heart attack")
    st.image("https://i.kym-cdn.com/photos/images/newsfeed/000/839/183/409.png",width=600)
    
    thalach = int(st.number_input('Your maximum heart rate achieved : '))
    oldpeak = int(st.number_input('Key in your Previous peak : '))
    thalassemia = int(st.radio(' Tick if  you have Thalassemia : ',(0,1,2,3)))
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Max Heart Rate",thalach,"Prev peak",oldpeak,"thalassemia",
                 thalassemia)
        temp = np.expand_dims([thalach,oldpeak,thalassemia], axis=0)
        outcome = model.predict(temp)
        
        outcome_dict = {0:'Less risk of having heart attack',
                        1:'Higher risk of having heart attack'}
        
        st.write(outcome_dict[outcome[0]])
        
        if outcome ==0:
            st.balloons()
            unsafe_html = '' 
        
st.markdown("![Alt Text](https://media0.giphy.com/media/T1TqR5TT62mVG/200.gif)")
st.write("Outside of the form")
st.audio("https://www.soundjay.com/human/sounds/heartbeat-02a.mp3")
