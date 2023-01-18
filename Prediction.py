import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import io
import webbrowser
import streamlit_menu as menu
#import streamlit_authenticator as stauth
import streamlit.components.v1 as components
#from print_print import*
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import extra_streamlit_components as stx
# Create a connection to the database





logo = Image.open(r'C:/Users/dell/Desktop/logoUvs/uvs.JPEG')
#st.markdown('<style>' + open('./style/style.css').read() + '</style>', unsafe_allow_html=True)
with st.sidebar:
    img = Image.open("C:/Users/dell/Desktop/logoUvs/uvs.JPEG")
    st.sidebar.image(img, width=400)
    st.image("https://www.campus-teranga.com/site/images/actualite/20210804-610aa19bbdf57.jpg")
    choose = option_menu("Application de detection Paludisme", ["About", "Prediction Paludisme","Enregistrer Patient","Contact"],
                    icons=['house',
                    'bi bi-graph-down-arrow',
                    'bi bi-droplet-fill',
                    'bi bi-file-person-fill',
                    'bi bi-file-person-fill'],
                    menu_icon="app-indicator", default_index=0,
                    styles={
    "container": {"padding": "5!important", "background-color": "#FF9333"},
    "icon": {"color": "orange", "font-size": "30px"},
    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#33B8FF"},
    "nav-link-selected": {"background-color": "#02ab21"},
}
)
logo = Image.open(r'C:/Users/dell/Desktop/logoUvs/uvs.JPEG')
#profile = Image.open(r'C:\Users\13525\Desktop\medium_profile.png')
if (choose == "About"):
    col1, col2 = st.columns( [0.9, 0.4])

    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
    </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">A propos du createur</p>',
                    unsafe_allow_html=True)
    with col2:               # To display brand log
        st.image(logo, width=150)
    st.write("Abdoulaye BA etudiant en MASTER 2 BIG DATA UNIVERSITE DU SENEGAL, Aussi ingenieur des traveaux informatiques à l'hopaital aristide le dantec et Administrateur Reseaux et sytemes d'information et gestionnaire de parc informatique le lien du repos sur github est disponibles sur ce lien: https://github.com/lbfacto/A_BA_UVS_ProjetPaludisme")
    st.write("Ce projet est realiser avec Dr Oumy Niass de l'universite virtuelle du senegal")
    st.write("Dans le module de cas industrielle sur des données reels de patient dans une base de données avec les ")
    st.write("prelevements de differenetes sujet et diverses criteres sont eablies selon des cas de prevelemenet differentes sur des barometre divers")
    st.write("En outre il a ete fait et concue une application pour faire des prediction selon le type de donnes a notre disposition qui va afficher le resiltat de la personne")
#st.image(profile, width=700 )rue
#analyse des donnes

elif (choose =="Prediction Paludisme"):
    st.title("Predictioon Paludisme sur des sujets au senegal") #titre de l
    palu_pedict = pickle.load(open('trained_model.pkl','rb'))
# change the input_data to a numpy array
#Les colones
    col1,col2, col3,col4=st.columns(4)

    Age = st.text_input('Age')


    ratio =st.text_input('ratio')

    G6PD = st.text_input('G6PD')

    EP_6M_AVT = st.text_input('EP_6M_AVT')

    AcPf_6M_AVT = st.text_input('AcPf_6M_AVT')

    EP_1AN_AVT = st.text_input('EP_1AN_AVT')

    AcPf_1AN_AVT =st.text_input('AcPf_1AN_AVT')

    EP_6M_APR = st.text_input('EP_6M_APR')

    AcPf_6M_APR = st.text_input('AcPf_6M_APR')

    EP_1AN_APR = st.text_input('EP_1AN_APR')

    AcPf_1AN_APR = st.text_input('AcPf_1AN_APR')
    palu_diagnosis =''
    if st.button('Resultat Paludisme'):
        palu_pediction = palu_pedict.predict([[Age,ratio,G6PD,EP_6M_AVT,AcPf_6M_AVT,EP_1AN_AVT,AcPf_1AN_AVT,EP_6M_APR	,AcPf_6M_APR,EP_1AN_APR	,AcPf_1AN_APR]])
        if(palu_pediction[0]==1):
            palu_diagnosis = 'Antigene positif Personne atteint du paludisme'
        else:
            palu_diagnosis= 'Antigene negatif personne n_est pas atteint du paludisme'
        st.success(palu_diagnosis)
if choose == "Enregistrer Patient":

    conn = sqlite3.connect('dbuvs.db')
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Enregister Resultat</p>', unsafe_allow_html=True)

    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS Medecin
    ( Fonction text, service text,Nom text, Prenom text,Email text, Telephone numeric, date numeric)''')
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
    #st.write('Please help us improve!')
        st.title('Partie Medecin')
        Fonction = st.text_input(label='Fonction Docteur')
        service = st.text_input(label='Enter service')
        Nom=st.text_input(label='Entrer Name') #Collect user feedback
        Prenom=st.text_input(label=' Prenom') #Collect user feedback
        Email=st.text_input(label='Entrer Email') #Collect user feedback
        Telephone=st.text_input(label='Entrer Telephone') #Collect user feedback
        date=st.date_input("Entrer la date")
        submitted = st.form_submit_button('Submit')

        if submitted:
            c.execute("INSERT INTO Medecin VALUES(?,?,?,?,?,?,?)", (Fonction, service, Nom,Prenom, Email, Telephone, date))
            conn.commit()
            st.write('Donnees Medecins enregistrer')
    # Connect to SQLite3 database
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS malades
    (Nom text, Age integer,Prenom text, Email text, Telephone numeric,Adresse text, Resultat text, NomMedcin text,Avis text,date numeric)''')
    with st.form(key='columns_in_form3',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
    #st.write('Please help us improve!')
        st.title('Partie patient')

        Nom=st.text_input(label='Entrer Nom') #Collect user feedback
        Age=st.text_input(label='Entrer Age') #Collect user feedback
        Prenom=st.text_input(label=' Prenom') #Collect user feedback
        Email=st.text_input(label='Entrer Email') #Collect user feedback
        Telephone=st.text_input(label='Entrer Telephone') #Collect user feedback
        Adresse=st.text_input(label='Entrer Adresse') #Collect user feedback
        Resultat=st.text_input(label='Entrer Resultat') #Collect user feedback
        NomMedcin = st.text_input(label='Docteur Traitant')
        Avis=st.text_input(label='Recommandation Medcin') #Collect user feedback
        date=st.date_input("Entrer la date")
        submitted = st.form_submit_button('Submit')

        if submitted:
            c.execute("INSERT INTO malades VALUES(?,?,?,?,?,?,?,?,?,?)", (Nom, Age,Prenom, Email, Telephone,Adresse, Resultat, NomMedcin,Avis,date))
            conn.commit()
            st.success('Donner Patient enregistrer')
    conn.close()



elif choose == "Contact":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
    </style> """, unsafe_allow_html=True,)

    st.markdown('<p class="font">Votre Avis</p>', unsafe_allow_html=True,)
    st.write("Nous aimerions entendre vos commentaires et vos suggestions.")
    st.write("Veuillez nous contacter en utilisant les informations ci-dessous.")
    with st.form(key='columns_in_form3',clear_on_submit=True):
        name = st.text_input("Nom")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submitted = st.form_submit_button('envoyer')

        if submitted:
            if name and email and message:
                st.success("Merci pour votre message! Nous vous répondrons dès que possible.")
            else:
                st.error("Veuillez remplir tous les champs.")











