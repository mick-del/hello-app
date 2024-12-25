import streamlit as st
import numpy as np
import pandas as pd
import joblib as jb
from datetime import datetime
# Initialisation d'une liste pour stocker l'historique des prédictions
model_RF_ppv = jb.load('RF_PPV.pkl')
model_DT_ppv = jb.load('DT_PPV.pkl')
model_ANN_MLP_ppv = jb.load('ANN_MLP_PPV.pkl')
model_XGB_ppv = jb.load('XGB_PPV.pkl')
model_XGBRF_ppv = jb.load('XGBRF_PPV.pkl')

model_RF_AOP = jb.load('RF_AOP.pkl')
model_DT_AOP = jb.load('DT_AOP.pkl')
model_ANN_MLP_AOP = jb.load('ANN_MLP_AOP.pkl')
model_GDB_AOP = jb.load('GDB_AOP.pkl')
model_XGBRF_AOP = jb.load('XGBRF_AOP.pkl')
model_MLR_AOP = jb.load('LR_AOP.pkl')

model_ppv = jb.load('Stacked_model_PPV.pkl')
model_AOP = jb.load('Stacked_model_AOP (1).pkl')
if 'historique' not in st.session_state:
    st.session_state.historique = []


# Fonction pour afficher la page d'accueil
def afficher_accueil():

    st.markdown("<h1 style='text-align: left; color: red;'>MMG GV-AB Predictor</h1>",
        unsafe_allow_html=True)
    st.image("C:\\Users\claud\Desktop\Dossier_stage\Screenshot_20240331-223623_1.png", caption="",
             use_container_width=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Bienvenu(e) sur l'application MMG GV-AB Predictor !</h2>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: justify;'>Cette application a été développée dans le cadre du travail de recherche portant sur"
        " la pédiction numérique des vibrations et de la surpression d'air générée lors des opérations de minage dans la mine"
        " à ciel ouvert de MMG Kinsevere. Cette application permet donc de prédire le niveau des vibrations ainsi que le niveau"
        " sonore produit par le minage. L'utilisateur devra juster entrer les paramètre d'entrée tel que la quantité d'explosif"
        " qui détonne intantanement ainsi que la distance entre la station d'enregistrement et le lot de minage.</p>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center;'>Utilisez le menu pour naviguer vers de nouvelles prédictions ou consulter l'historique.</p>",
        unsafe_allow_html=True)
    st.image("C:\\Users\claud\Desktop\Dossier_stage\Pictures\Mine\DJI_0753.JPG", caption="G2-1~2.PNG", use_container_width=True)

# Fonction pour faire des prédictions
def faire_prediction(option):
    if option == 'Stations préenregistrées':
        st.markdown(
            "<h3 style='text-align: left; color: black;'>Etape 2. Choix de la mine :</h3>",
            unsafe_allow_html=True)
        option2 = st.radio("A partir de quelle mine voulez-vous prédire les vibrations et la surpression de l'air?", (
            "Mashi Pit", "Central Pit", "Kinsevere Hill Pit"
        ))
        st.markdown(
            "<h3 style='text-align: left; color: black;'>Etape 3. Spécification de la station :</h3>",
            unsafe_allow_html=True)
        if option2 == "Mashi Pit":
            # Sélection des stations
            # Checkbox pour sélectionner toutes les stations
            select_all = st.checkbox("Sélectionner toutes les stations")

            if select_all:
                selected_stations = list(stations_data.keys())
            else:
                selected_stations = st.multiselect("Sélectionnez les stations :", list(stations_data.keys()))

            st.markdown(
                "<h3 style='text-align: left; color: black;'>Etape 4. Spécification des variables d'entrée du modèle :</h3>",
                unsafe_allow_html=True)

            if selected_stations:
                # Obtenir les distances correspondantes
                distances = [stations_data[station] for station in selected_stations]
                Select_recep = [station for station in selected_stations]
                # Entrée de la quantité d'explosifs
                quantity_of_explosives = st.number_input("Entrez la quantité d'explosif (kg) :", min_value=0)
                Number_holes = st.number_input("Entrez le nombre des trous par lot de minage:", min_value=0)

                if quantity_of_explosives > 0:
                    # Calcul de la distance mise à l'échelle
                    scaled_distance = [distance / (quantity_of_explosives ** (1 / 2)) for distance in distances]
                    scaled_distance1 = [distance / (quantity_of_explosives ** (1 / 3)) for distance in distances]

                    # Afficher les données en entrée
                    Data_entry = pd.DataFrame({
                            'Receptors':Select_recep, 'distance (m)': distances,
                            'quantity_of_explosives (Kg)': [quantity_of_explosives] * len(distances), 'Number_of_holes':Number_holes,
                            'scaled_distance_PPV': scaled_distance, 'scaled_distance_AOP': scaled_distance1
                        })
                    st.markdown(
                        "<h3 style='text-align: left; color: black;'>Etape 5. Visualisation des données entrées :</h3>",
                        unsafe_allow_html=True)
                    st.dataframe(Data_entry)
                    # Bouton pour activer la prédiction

                    st.markdown(
                        "<h3 style='text-align: left; color: black;'>Etape 6. Prédire l'intensité des vibrations et de la"
                        " surpression de l'air :</h3>",
                        unsafe_allow_html=True)

                    if st.button("Cliquez pour faire la prédiction"):
                        # Préparer les données pour la prédiction
                        st.success("Prédiction faite avec succès !")
                        input_data = pd.DataFrame({
                            'Qmax':[quantity_of_explosives] * len(distances), 'Slope_Dist': distances,
                            'Nholes': Number_holes, 'Scaled_Dist': scaled_distance
                        })
                        input_data1 = pd.DataFrame({
                            'Qmax':[quantity_of_explosives] * len(distances), 'Slope_Dist': distances,
                            'Nholes': Number_holes, 'SD_': scaled_distance1
                        })

                      # Prédictions des deux modèles (remplacez par vos modèles)

                        predictions_RF_model_1 = model_RF_ppv.predict(input_data)
                        predictions_DT_model_1 = model_DT_ppv.predict(input_data)
                        predictions_XGBRF_model_1 = model_XGBRF_ppv.predict(input_data)
                        predictions_XGB_model_1 = model_XGB_ppv.predict(input_data)
                        predictions_ANN_MLP_model_1 = model_ANN_MLP_ppv.predict(input_data)


                        predictions_RF_model_2 = model_RF_AOP.predict(input_data1)
                        predictions_DT_model_2 = model_DT_AOP.predict(input_data1)
                        predictions_XGBRF_model_2 = model_XGBRF_AOP.predict(input_data1)
                        predictions_GDB_model_2 = model_GDB_AOP.predict(input_data1)
                        predictions_ANN_MLP_model_2 = model_ANN_MLP_AOP.predict(input_data1)
                        predictions_MLR_model_2 = model_MLR_AOP.predict(input_data1)

                        new_ID1 = pd.DataFrame({'XGB':predictions_XGB_model_1, 'RF':predictions_RF_model_1,
                                              'XGBRF':predictions_XGBRF_model_1, 'DT':predictions_DT_model_1,
                                              'ANN_MLP':predictions_ANN_MLP_model_1})

                        new_ID2 = pd.DataFrame({'MLR':predictions_MLR_model_2,'GDB': predictions_GDB_model_2, 'RF': predictions_RF_model_2,
                                                'XGBRF': predictions_XGBRF_model_2, 'DT': predictions_DT_model_2,
                                                'ANN_MLP': predictions_ANN_MLP_model_2})

                        predictions_model_1 = model_ppv.predict(new_ID1)
                        predictions_model_2 = model_AOP.predict(new_ID2)
                        result = pd.DataFrame({'Stations':Select_recep,'Mine' : option2, 'PPV_Predicted': predictions_model_1, 'AOP_Predicted':predictions_model_2,
                                               'Date/hour':[datetime.now().strftime("%Y-%m-%d   /  %H:%M:%S")]* len(distances)})

                        st.markdown(
                            "<h3 style='text-align: left; color: black;'>Etape 7. Affichage des résultats de prédiction :</h3>",
                            unsafe_allow_html=True)
                        st.dataframe(result)

                        # Afficher les résultats de prédiction

                        st.session_state.historique.append(result)
        elif option2 == "Central Pit":
            # Sélection des stations
            # Checkbox pour sélectionner toutes les stations
            select_all = st.checkbox("Sélectionner toutes les stations")

            if select_all:
                selected_stations = list(stations_data1.keys())
            else:
                selected_stations = st.multiselect("Sélectionnez les stations :", list(stations_data1.keys()))

            st.markdown(
                "<h3 style='text-align: left; color: black;'>Etape 4. Spécification des variables d'entrée du modèle :</h3>",
                unsafe_allow_html=True)
            if selected_stations:
                # Obtenir les distances correspondantes
                distances = [stations_data1[station] for station in selected_stations]
                Select_recep = [station for station in selected_stations]
                # Entrée de la quantité d'explosifs
                quantity_of_explosives = st.number_input("Entrez la quantité d'explosif (kg) :", min_value=0)
                Number_holes = st.number_input("Entrez le nombre des trous par lot de minage :", min_value=0)

                if quantity_of_explosives > 0:
                    # Calcul de la distance mise à l'échelle
                    scaled_distance = [distance / (quantity_of_explosives ** (1 / 2)) for distance in distances]
                    scaled_distance1 = [distance / (quantity_of_explosives ** (1 / 3)) for distance in distances]

                    # Afficher les données en entrée
                    Data_entry = pd.DataFrame({
                        'Receptors': Select_recep, 'distance (m)': distances,
                        'quantity_of_explosives (Kg)': [quantity_of_explosives] * len(distances),
                        'Number_of_holes': Number_holes,
                        'scaled_distance_PPV': scaled_distance, 'scaled_distance_AOP': scaled_distance1
                    })

                    st.markdown(
                        "<h3 style='text-align: left; color: black;'>Etape 5. Visualisation des données entrées :</h3>",
                        unsafe_allow_html=True)

                    st.dataframe(Data_entry)

                    st.markdown(
                        "<h3 style='text-align: left; color: black;'>Etape 6. Prédire l'intensité des vibrations et de la"
                        " surpression de l'air :</h3>",
                        unsafe_allow_html=True)

                    # Bouton pour activer la prédiction
                    if st.button("Cliquez pour faire la prédiction"):
                        # Préparer les données pour la prédiction
                        st.success("Prédiction faite avec succès !")
                        input_data = pd.DataFrame({
                            'Qmax': [quantity_of_explosives] * len(distances), 'Slope_Dist': distances,
                            'Nholes': Number_holes, 'Scaled_Dist': scaled_distance
                        })
                        input_data1 = pd.DataFrame({
                            'Qmax': [quantity_of_explosives] * len(distances), 'Slope_Dist': distances,
                            'Nholes': Number_holes, 'SD_': scaled_distance1
                        })

                        # Prédictions des deux modèles (remplacez par vos modèles)

                        predictions_RF_model_1 = model_RF_ppv.predict(input_data)
                        predictions_DT_model_1 = model_DT_ppv.predict(input_data)
                        predictions_XGBRF_model_1 = model_XGBRF_ppv.predict(input_data)
                        predictions_XGB_model_1 = model_XGB_ppv.predict(input_data)
                        predictions_ANN_MLP_model_1 = model_ANN_MLP_ppv.predict(input_data)

                        predictions_RF_model_2 = model_RF_AOP.predict(input_data1)
                        predictions_DT_model_2 = model_DT_AOP.predict(input_data1)
                        predictions_XGBRF_model_2 = model_XGBRF_AOP.predict(input_data1)
                        predictions_GDB_model_2 = model_GDB_AOP.predict(input_data1)
                        predictions_ANN_MLP_model_2 = model_ANN_MLP_AOP.predict(input_data1)
                        predictions_MLR_model_2 = model_MLR_AOP.predict(input_data1)

                        new_ID1 = pd.DataFrame({'XGB': predictions_XGB_model_1, 'RF': predictions_RF_model_1,
                                                'XGBRF': predictions_XGBRF_model_1, 'DT': predictions_DT_model_1,
                                                'ANN_MLP': predictions_ANN_MLP_model_1})

                        new_ID2 = pd.DataFrame({'MLR': predictions_MLR_model_2, 'GDB': predictions_GDB_model_2,
                                                'RF': predictions_RF_model_2,
                                                'XGBRF': predictions_XGBRF_model_2, 'DT': predictions_DT_model_2,
                                                'ANN_MLP': predictions_ANN_MLP_model_2})

                        predictions_model_1 = model_ppv.predict(new_ID1)
                        predictions_model_2 = model_AOP.predict(new_ID2)
                        result = pd.DataFrame({'Stations': Select_recep, 'Mine' : option2,'PPV_Predicted': predictions_model_1,
                                               'AOP_Predicted': predictions_model_2,
                                               'Date/hour': [datetime.now().strftime("%Y-%m-%d   /  %H:%M:%S")] * len(
                                                   distances)})
                        st.dataframe(result)

                        # Afficher les résultats de prédiction

                        st.session_state.historique.append(result)
        else:
            # Sélection des stations
            # Checkbox pour sélectionner toutes les stations
            select_all = st.checkbox("Sélectionner toutes les stations")

            if select_all:
                selected_stations = list(stations_data2.keys())
            else:
                selected_stations = st.multiselect("Sélectionnez les stations :", list(stations_data2.keys()))

            st.markdown(
                "<h3 style='text-align: left; color: black;'>Etape 4. Spécification des variables d'entrée du modèle :</h3>",
                unsafe_allow_html=True)

            if selected_stations:
                # Obtenir les distances correspondantes
                distances = [stations_data2[station] for station in selected_stations]
                Select_recep = [station for station in selected_stations]
                # Entrée de la quantité d'explosifs
                quantity_of_explosives = st.number_input("Entrez la quantité d'explosif (kg) :", min_value=0)
                Number_holes = st.number_input("Entrez le nombre des trous par lot de minage :", min_value=0)

                if quantity_of_explosives > 0:
                    # Calcul de la distance mise à l'échelle
                    scaled_distance = [distance / (quantity_of_explosives ** (1 / 2)) for distance in distances]
                    scaled_distance1 = [distance / (quantity_of_explosives ** (1 / 3)) for distance in distances]

                    # Afficher les données en entrée
                    Data_entry = pd.DataFrame({
                        'Receptors': Select_recep, 'distance (m)': distances,
                        'quantity_of_explosives (Kg)': [quantity_of_explosives] * len(distances),
                        'Number_of_holes': Number_holes,
                        'scaled_distance_PPV': scaled_distance, 'scaled_distance_AOP': scaled_distance1
                    })

                    st.markdown(
                        "<h3 style='text-align: left; color: black;'>Etape 5. Visualisation des données entrées :</h3>",
                        unsafe_allow_html=True)

                    st.dataframe(Data_entry)

                    st.markdown(
                        "<h3 style='text-align: left; color: black;'>Etape 6. Prédire l'intensité des vibrations et de la"
                        " surpression de l'air :</h3>",
                        unsafe_allow_html=True)

                    # Bouton pour activer la prédiction
                    if st.button("Cliquez pour faire la prédiction"):
                        st.success("Prédiction faite avec succès !")
                        # Préparer les données pour la prédiction
                        input_data = pd.DataFrame({
                            'Qmax': [quantity_of_explosives] * len(distances), 'Slope_Dist': distances,
                            'Nholes': Number_holes, 'Scaled_Dist': scaled_distance
                        })
                        input_data1 = pd.DataFrame({
                            'Qmax': [quantity_of_explosives] * len(distances), 'Slope_Dist': distances,
                            'Nholes': Number_holes, 'SD_': scaled_distance1
                        })

                        # Prédictions des deux modèles (remplacez par vos modèles)

                        predictions_RF_model_1 = model_RF_ppv.predict(input_data)
                        predictions_DT_model_1 = model_DT_ppv.predict(input_data)
                        predictions_XGBRF_model_1 = model_XGBRF_ppv.predict(input_data)
                        predictions_XGB_model_1 = model_XGB_ppv.predict(input_data)
                        predictions_ANN_MLP_model_1 = model_ANN_MLP_ppv.predict(input_data)

                        predictions_RF_model_2 = model_RF_AOP.predict(input_data1)
                        predictions_DT_model_2 = model_DT_AOP.predict(input_data1)
                        predictions_XGBRF_model_2 = model_XGBRF_AOP.predict(input_data1)
                        predictions_GDB_model_2 = model_GDB_AOP.predict(input_data1)
                        predictions_ANN_MLP_model_2 = model_ANN_MLP_AOP.predict(input_data1)
                        predictions_MLR_model_2 = model_MLR_AOP.predict(input_data1)

                        new_ID1 = pd.DataFrame({'XGB': predictions_XGB_model_1, 'RF': predictions_RF_model_1,
                                                'XGBRF': predictions_XGBRF_model_1, 'DT': predictions_DT_model_1,
                                                'ANN_MLP': predictions_ANN_MLP_model_1})

                        new_ID2 = pd.DataFrame({'MLR': predictions_MLR_model_2, 'GDB': predictions_GDB_model_2,
                                                'RF': predictions_RF_model_2,
                                                'XGBRF': predictions_XGBRF_model_2, 'DT': predictions_DT_model_2,
                                                'ANN_MLP': predictions_ANN_MLP_model_2})

                        predictions_model_1 = model_ppv.predict(new_ID1)
                        predictions_model_2 = model_AOP.predict(new_ID2)
                        result = pd.DataFrame({'Stations': Select_recep, 'Mine' : option2, 'PPV_Predicted': predictions_model_1,
                                               'AOP_Predicted': predictions_model_2,
                                               'Date/hour': [datetime.now().strftime("%Y-%m-%d   /  %H:%M:%S")] * len(
                                                   distances)})
                        st.dataframe(result)

                        # Afficher les résultats de prédiction

                        st.session_state.historique.append(result)
    else:
        # Créer une nouvelle station

        st.markdown(
            "<h3 style='text-align: left; color: black;'>Etape 2. Spécification des variables d'entrée du modèle :</h3>",
            unsafe_allow_html=True)

        new_station_name = st.text_input("Nom de la nouvelle station :")
        Pit = st.radio("A partir de quelle mine voulez-vous prédire les vibrations et la surpression de l'air?", (
            "Mashi Pit", "Central Pit", "Kinsevere Hill Pit"
        ))
        new_station_distance = st.number_input("Entrez la distance entre la nouvelle station et le lot de minage (m) :", min_value=0)
        quantity_of_explosives = st.number_input("Entrez la quantité d'explosif (kg) :", min_value=0)
        Number_holes = st.number_input("Entrez le nombre des trous par lot de minage :", min_value=0)

        if new_station_name and new_station_distance > 0 and quantity_of_explosives > 0:
            # Calcul de la distance mise à l'échelle
            scaled_distance = new_station_distance / (quantity_of_explosives ** (1 / 2))
            scaled_distance1 = new_station_distance / (quantity_of_explosives ** (1 / 3))


            # Afficher les données en entrée
            Data_entry = pd.DataFrame({
                'Receptors': [new_station_name], 'distance (m)': [new_station_distance],
                'quantity_of_explosives (Kg)': [quantity_of_explosives],
                'Number_of_holes': [Number_holes],
                'scaled_distance_PPV': [scaled_distance], 'scaled_distance_AOP': [scaled_distance1]
            })

            st.markdown(
                "<h3 style='text-align: left; color: black;'>Etape 3. Visualisation des données entrées :</h3>",
                unsafe_allow_html=True)

            st.dataframe(Data_entry)

            st.markdown(
                "<h3 style='text-align: left; color: black;'>Etape 4. Prédire l'intensité des vibrations et de la"
                " surpression de l'air :</h3>",
                unsafe_allow_html=True)

            # Bouton pour activer la prédiction
            if st.button("Cliquez ici pour faire la prédiction"):
                st.success("Prédiction faite avec succès !")
                # Préparer les données pour la prédiction

                input_data = pd.DataFrame({
                    'Q_max': [quantity_of_explosives],
                    'Slope_Dist': [new_station_distance],
                    'Nholes': [Number_holes],
                    'scaled_distance': [scaled_distance]
                })
                input_data1 = pd.DataFrame({
                    'Q_max': [quantity_of_explosives],
                    'Slope_Dist': [new_station_distance],
                    'Nholes': [Number_holes],
                    'SD_': [scaled_distance1]
                })

                # Prédictions des deux modèles (remplacez par vos modèles)

                predictions_RF_model_1 = model_RF_ppv.predict(input_data)
                predictions_DT_model_1 = model_DT_ppv.predict(input_data)
                predictions_XGBRF_model_1 = model_XGBRF_ppv.predict(input_data)
                predictions_XGB_model_1 = model_XGB_ppv.predict(input_data)
                predictions_ANN_MLP_model_1 = model_ANN_MLP_ppv.predict(input_data)

                predictions_RF_model_2 = model_RF_AOP.predict(input_data1)
                predictions_DT_model_2 = model_DT_AOP.predict(input_data1)
                predictions_XGBRF_model_2 = model_XGBRF_AOP.predict(input_data1)
                predictions_GDB_model_2 = model_GDB_AOP.predict(input_data1)
                predictions_ANN_MLP_model_2 = model_ANN_MLP_AOP.predict(input_data1)
                predictions_MLR_model_2 = model_MLR_AOP.predict(input_data1)

                new_ID1 = pd.DataFrame({'XGB': predictions_XGB_model_1, 'RF': predictions_RF_model_1,
                                        'XGBRF': predictions_XGBRF_model_1, 'DT': predictions_DT_model_1,
                                        'ANN_MLP': predictions_ANN_MLP_model_1})

                new_ID2 = pd.DataFrame({'MLR': predictions_MLR_model_2, 'GDB': predictions_GDB_model_2,
                                        'RF': predictions_RF_model_2,
                                        'XGBRF': predictions_XGBRF_model_2, 'DT': predictions_DT_model_2,
                                        'ANN_MLP': predictions_ANN_MLP_model_2})

                predictions_model_1 = model_ppv.predict(new_ID1)
                predictions_model_2 = model_AOP.predict(new_ID2)
                result = pd.DataFrame({'Stations': new_station_name, 'Mine':Pit, 'PPV_Predicted': predictions_model_1,
                                       'AOP_Predicted': predictions_model_2,
                                       'Date/hour': [datetime.now().strftime("%Y-%m-%d   /  %H:%M:%S")]})
                st.dataframe(result)

                # Afficher les résultats de prédiction

                st.session_state.historique.append(result)


# Fonction pour afficher l'historique des prédictions
def afficher_historique():
    st.markdown("<h2>Historique des prédictions</h2>", unsafe_allow_html=True)
    if st.session_state.historique:
        hist = pd.concat(st.session_state.historique, ignore_index=True)
        st.dataframe(hist)

    else:
        st.markdown("<p>Aucune prédiction n'a été faite jusqu'à présent.</p>", unsafe_allow_html=True)

# Fonction pour effacer l'historique
def effacer_historique():
    st.session_state.historique = []
    st.success("Historique effacé avec succès !")

# Données des stations (exemple)
stations_data = {
    'Mikanda': 6420,
    'Denis': 6846,
    'Sela': 3746,
    'Ernest': 3240,
    'Mpundu': 1789,
    'Kalianda': 947,
    'Katumba': 3480,
    'Lutenge': 5284,
    'Kilongo': 927,
    'Poto 93': 3345,
}

stations_data1 = {
    'Mikanda': 6348,
    'Denis': 6686,
    'Sela': 3780,
    'Ernest': 3665,
    'Mpundu': 2426,
    'Kalianda': 1650,
    'Katumba': 3941,
    'Lutenge': 5995,
    'Kilongo': 1429,
    'Poto 93': 2637,
}

stations_data2 = {
    'Mikanda': 6034,
    'Denis': 6259,
    'Sela': 3623,
    'Ernest': 3848,
    'Mpundu': 3040,
    'Kalianda': 2484,
    'Katumba': 4736,
    'Lutenge': 6622,
    'Kilongo': 2239,
    'Poto 93': 2131,
}


# Menu de navigation
menu = st.sidebar.selectbox("Menu", ("Accueil", "Nouvelle Prédiction", "Historique"))

# Afficher la page appropriée en fonction de la sélection
if menu == "Accueil":
    afficher_accueil()
    st.markdown(
        "<footer style='text-align: center; padding: 50px;'><small>&copy;2024 Travail de fin d'étude MBN: Prédiction numérique des "
        " vibrations et la surpression d'air lors des opérations de minage dans une mine à ciel ouvert : Cas de la mine MMG Kinsevere."
        "</small></footer>",
        unsafe_allow_html=True)
elif menu == "Nouvelle Prédiction":
    st.markdown("<h1 style='text-align: left; color: red;'>MMG GV-AB Predictor</h1>",
                unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Bienvenu(e) dans le menu de la  nouvelle prediction</h2>",
                unsafe_allow_html=True)
    st.image("E:\DOSSIER MICK\COURS\Mick_TFE\Images TFE\G2-1~2.PNG", caption="",
             use_container_width=True)
    st.markdown(
        "<h3 style='text-align: left; color: black;'>1. Choisissez la station de prédiction :</h3>",
        unsafe_allow_html=True)
    prediction_option = st.radio("Voulez-vous faire la prédiction sur :",
                                 ('Stations préenregistrées', 'Créer une nouvelle station'))
    faire_prediction(prediction_option)
    st.markdown(
        "<footer style='text-align: center; padding: 50px;'><small>&copy;2024 Travail de fin d'étude MBN: Prédiction numérique des "
        " vibrations et la surpression d'air lors des opérations de minage dans une mine à ciel ouvert : Cas de la mine MMG Kinsevere."
        "</small></footer>",
        unsafe_allow_html=True)
elif menu == "Historique":
    st.markdown("<h1 style='text-align: left; color: red;'>MMG GV-AB Predictor</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<h2 style='text-align: center; color: black;'>Bienvenu(e) dans le menu d'historiques de prédiction</h2>",
        unsafe_allow_html=True)
    afficher_historique()
    if st.button("Effacer l'historique"):
        effacer_historique()
    st.markdown(
        "<footer style='text-align: center; padding: 50px;'><small>&copy;2024 Travail de fin d'étude MBN: Prédiction numérique des "
        " vibrations et la surpression d'air lors des opérations de minage dans une mine à ciel ouvert : Cas de la mine MMG Kinsevere."
        "</small></footer>",
        unsafe_allow_html=True)
# Ajout d'une image
#st.image("path/to/your/image.png", caption="Votre image", use_column_width=True)


