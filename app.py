import streamlit as st
import helper
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from streamlit_option_menu import option_menu
from home_page import show_home_page

img = Image.open("quora_logo.png")
page_config = {"page_title": "QUORA-Duplicate question pairs", "page_icon": img, "layout": "centered"}
st.set_page_config(**page_config)

page = option_menu(
    menu_title=None,
    options=["Home", "Classification", "Code"],
    icons=["house-fill", "motherboard", "file-earmark-code"],
    default_index=0,
    orientation="horizontal",
    styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "black", "font-size": "14px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "red"}
            }
)

# Home page
if page == "Home":
    show_home_page()


# Prediction page
if page == "Classification":

    st.text("")
    st.markdown("""***This Machine Learning application allows you to determine whether a given pair of questions are
                   duplicates or not.***""")
    st.text("")
    model = pickle.load(open('model.pkl', 'rb'))
    q1 = st.text_input('***Enter question 1 :***')
    q2 = st.text_input('***Enter question 2 :***')

    if st.button('Predict'):
        query = helper.query_point_creator(q1, q2)
        result = model.predict(query)[0]
        probabilities = model.predict_proba(query)[0]
        if result:
            st.markdown('The questions are **Duplicate**.')
        else:
            st.markdown('The questions are **Unique**.')
        st.text("")
        st.text("")
        probs = [np.round(x, 6) for x in probabilities]
        class_labels = [0, 1]
        ax = sns.barplot(probs, palette="winter", orient='h', width=0.25)
        ax.set_yticklabels(class_labels, rotation=0)
        plt.title("Probabilities of the datapoint belonging to each class")
        for index, value in enumerate(probs):
            plt.text(value, index, str(value))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


# Code page
if page == "Code":

    st.text("")
    st.write("###### If you are more interested in the code you can directly jump into these repositories :")
    st.text("")
    st.caption("**DEPLOYMENT** : ***[link](https://github.com/sangoleshubham20/dqp_DeploymentCode)***")
    st.caption("**MODELLING** : ***[link](https://github.com/sangoleshubham20/DuplicateQuestionPair_ModellingCode)***")
