# -*- coding: utf-8 -*-
""" Librairie personnelle pour manipulation les modèles de machine learning
"""

# ====================================================================
# Outils ML -  projet 4 Openclassrooms
# ====================================================================

import datetime
#import jyquickhelper
import numpy as np
import pandas as pd
import pycaret as pyc
import sys
from math import sqrt
import pickle
from pprint import pprint
import mlflow
import pickle
import time
import shap

# Librairies personnelles
import fonctions_data
import fonctions_models

# Création de pipelines
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer

# Feature selection
from sklearn.feature_selection import RFECV
#import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn import decomposition
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer

# Scoring - cross-validation
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, \
    cross_validate, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, \
    KFold, learning_curve

# Data pré-processing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, LabelEncoder, LabelBinarizer
from category_encoders import LeaveOneOutEncoder, TargetEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import FeatureHasher


# Modélisation
from sklearn import tree
from pycaret.regression import *
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, \
    BayesianRidge, HuberRegressor, OrthogonalMatchingPursuit, Lars
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, \
    ExtraTreesRegressor, GradientBoostingRegressor, BaggingRegressor, \
    StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Warnings
import warnings
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from dtreeviz.trees import dtreeviz
import seaborn as sns

# Métriques
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.0'



def sort_array(array_to_sort):
    '''
    TRI un NUMPY ARRAY par ordre descendant
    Parameters
    ----------
    array_to_sort : l'array à trier, obligatoire.
    Returns
    -------
    array_to_sort : l'array trié par ordre descendant.
    '''
    for i in range(len(array_to_sort)):
        swap = i + np.argmin(array_to_sort[i:])
        (array_to_sort[i], array_to_sort[swap]) = (
            array_to_sort[swap], array_to_sort[i])
    return array_to_sort


############

PropertyGFABuilding_ix, NumberofFloors_ix = 4, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin) :
    
    def __init__(self, add_GFA_per_floor_per_building=True) :
        self.add_GFA_per_floor_per_building = add_GFA_per_floor_per_building
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.add_GFA_per_floor_per_building :
            BuildingGFA_per_floor = X[:, PropertyGFABuilding_ix]*X[:, NumberofFloors_ix]
            return np.c_[X, BuildingGFA_per_floor]
        else : 
            return X 


#############

class AddBooleanEnergyType(BaseEstimator, TransformerMixin) :
    
    def __init__(self, add_Boolean_EnergyType=True) :
        self.add_Boolean_EnergyType = add_Boolean_EnergyType
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.add_Boolean_EnergyType :
            
            return np.array(X > 0) * 1
        else : 
            return X 


############################

def display_scores(scores, scoring=["neg_mean_squared_error", "neg_mean_absolute_error"]) :
    # Résultats, moyenne et écart-type

    print("Resultats de la cross validation :")
     
    for metric in scoring : 
    
        results = -scores["test_" + metric]
        if metric == "neg_mean_squared_error" :
            print("Métrique utilisé : RMSE")
            results = np.sqrt(results)
        else :
            print("Métrique utilisé : MAE")
        print("\t - Moyenne : {:.3f}".format(results.mean()))
        print("\t - Ecart-type : {:.3f}".format(results.std()))
        print("\t - Coefficient de variation : {:.2f} %".format(results.std()/results.mean()*100))
        print("\n")


####################################

def mean_absolute_percentage_error(y_true, y_pred): 
    # MAPE : Mean aboslute percentage error

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


####################################

def define_pipeline(preprocessor, regressor):
    
    #global preprocessor, regressor 
    # Assembly of preprocessor and regressor
    pipe = Pipeline([("preprocess", preprocessor),
                      ("regressor", regressor),
                       ])
    return pipe

###################################

# Fonction qui permet d'afficher un barplot des résultats : moyenne, écart-type et coefficient de variation
def graphical_display(results, title) :
    
    x = list(results)
    y = np.array(list(results.values()))
    rmse_mean = [score.mean() for score in np.array(list(results.values()))]
    rmse_std = [score.std() for score in np.array(list(results.values()))]
    
    # Liste des couleurs pour chaque segment
    palette = sns.color_palette()
    colors = palette.as_hex()[0: len(results)]

    fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10,5))
    fig.suptitle(title, fontsize=18)
    # Barplot de la moyenne et son ecart-type
    axs[0].bar(x=x, height=rmse_mean, color=colors)
    axs[0].set_ylabel("Mean")
    # Ajout de la barre ecart-type
    axs[0].errorbar(range(len(rmse_mean)), rmse_mean, yerr=rmse_std, fmt='none', ecolor='black')
    # Titre du graphique
    axs[1].bar(x=x, height=rmse_std, color=colors)
    axs[1].set_ylabel("Variance")

    plt.show()
    
#####################################

class AddEnergyTypePredicted(BaseEstimator, TransformerMixin) :
    
    def __init__(self, list_energy_type) :
        self.list_energy_type = list_energy_type
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        Y = []
        for energy_type in self.list_energy_type:
            Y.append(final_model[energy_type].predict(X))
        return np.array(Y).T
    
    
######################################


    





# --------------------------------------------------------------------
# -- Entrainer/predire modele de regression de base avec cross-validation
# --------------------------------------------------------------------

from matplotlib.gridspec import GridSpec

def process_regression(
        model_reg,
        X_train,
        X_test,
        y_train,
        y_test,
        df_resultats,
        titre,
        affiche_tableau=True,
        affiche_comp=True,
        affiche_erreur=True,
        xlim_sup=130000000):
    """
    Lance un modele de régression, effectue cross-validation et sauvegarde les
    performances
    Parameters
    ----------
    model_reg : modèle de régression initialisé, obligatoire.
    X_train : train set matrice X, obligatoire.
    X_test : test set matrice X, obligatoire.
    y_train : train set vecteur y, obligatoire.
    y_test : test set, vecteur y, obligatoire.
    df_resultats : dataframe sauvegardant les traces, obligatoire
    titre : titre à inscrire dans le tableau de sauvegarde, obligatoire.
    affiche_tableau : booleen affiche le tableau de résultat, facultatif.
    affiche_comp : booleen affiche le graphique comparant y_test/y_pres,
                   facultatif.
    affiche_erreur : booleen affiche le graphique des erreurs, facultatif.
    xlim_sup : limite supérieure de x, facultatif.
    Returns
    -------
    df_resultats : Le dataframe de sauvegarde des performances.
    y_pred : Les prédictions pour le modèle
    """
    # Top début d'exécution
    time_start = time.time()

    # Entraînement du modèle
    model_reg.fit(X_train, y_train)
    
    # Sauvegarde du modèle de régression entaîné
    #with open('modeles/modele_' + titre + '.pickle', 'wb') as f:
    #    pickle.dump(model_reg, f, pickle.HIGHEST_PROTOCOL)
        
    # Prédictions avec le test set
    y_pred = model_reg.predict(X_test)

    # Top fin d'exécution
    time_end = time.time()

    # Calcul des métriques
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    errors = abs(y_pred - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape

    # durée d'exécution
    time_execution = time_end - time_start

    # cross validation
    scoring = ['r2', 'neg_mean_squared_error']
    scores = cross_validate(model_reg, X_train, y_train, cv=10,
                            scoring=scoring, return_train_score=True)

    # Sauvegarde des performances
    df_resultats = df_resultats.append(pd.DataFrame({
         'Modèle': [titre],
         'R2': [r2],
         'MSE': [mse],
         'RMSE': [rmse],
         'MAE': [mae],
         'Erreur moy': [np.mean(errors)],
         'Précision': [accuracy],
         'Durée': [time_execution],
         'Test R2 CV': [scores['test_r2'].mean()],
         'Test R2 +/-': [scores['test_r2'].std()],
         'Test MSE CV': [-(scores['test_neg_mean_squared_error'].mean())],
         'Train R2 CV': [scores['train_r2'].mean()],
         'Train R2 +/-': [scores['train_r2'].std()],
         'Train MSE CV': [-(scores['train_neg_mean_squared_error'].mean())]
     }), ignore_index=True)

    if affiche_tableau:
        display(df_resultats.style.hide_index())

    if affiche_comp:
        # retour aux valeurs d'origine
        
        test = (10 ** y_test) + 1
        predict = (10 ** y_pred) + 1

        sns.regplot(x = test , y = predict,
                scatter_kws = {"color": "black", "alpha": 0.5},
            line_kws = {"color": "red"})
        
        plt.xlabel('y_test')
        plt.ylabel('y_predicted')
        plt.suptitle(t='Tests /Predictions pour : '
                       + str(titre),
                       y=0,
                       fontsize=16,
                       alpha=0.75,
                       weight='bold',
                       ha='center')
        plt.xlim([0, xlim_sup])
        plt.show()



        # Affichage Test vs Predictions
        

    if affiche_erreur:
        # retour aux valeurs d'origine
        test = (10 ** y_test) + 1
        predict = (10 ** y_pred) + 1
        # affichage des erreurs
        df_res = pd.DataFrame({'true': test, 'pred': predict})
        df_res = df_res.sort_values('true')

        plt.plot(df_res['pred'].values, label='pred')
        plt.plot(df_res['true'].values, label='true')
        plt.xlabel('Test set')
        plt.ylabel("Consommation energie totale")
        plt.suptitle(t='Erreurs pour : '
                     + str(titre),
                     y=0,
                     fontsize=16,
                     alpha=0.75,
                     weight='bold',
                     ha='center')
        plt.legend()
        plt.show()


    return df_resultats, y_pred

############################################"
#EVALUATION DU MODELE
############################################


def evaluer_hyperparametre(models, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre n_estimators de ExraTreesRegressor
    Parameters
    ----------
    models : liste des modèles instanciés avec des valeurs différentes
             d'hyperparamètre', obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    # sauvegarde des performances
    results, names = list(), list()

    print('Hyperparam', 'Test R2 +/- std', 'Train R2 +/- std')
    for name, model in models.items():
        # evaluate the model
        # scores = cross_val_score(model, X, y, scoring='r2', cv=10, n_jobs=-1)
        scores = pd.DataFrame(
            cross_validate(
                model,
                X,
                y,
                cv=10,
                scoring='r2',
                return_train_score=True))

        # store the results
        results.append(scores['test_score'])
        names.append(name)
        test_mean = scores['test_score'].mean()
        test_std = scores['test_score'].std()
        train_mean = scores['train_score'].mean()
        train_std = scores['train_score'].std()
        # sAffiche le R2 pour le nombre d'arbres
        print('>%s %.5f (%.5f) %.5f (%.5f)' %
              (name, test_mean, test_std, train_mean, train_std))

    if affiche_boxplot:
        plt.boxplot(results, labels=names, showmeans=True)
        plt.show()



# --------------------------------------------------------------------
# GB- GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre n_estimators
# Le nombre d'étapes de boosting à exécuter
# --------------------------------------------------------------------

def regle_gradboost_nestimators(n_estimators, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre ccp_alpha de GradientBoostRegressor
    Parameters
    ----------
    n_estimators : Le nombre d'étapes de boosting à exécuter, obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in n_estimators:
        models[i] = GradientBoostingRegressor(n_estimators=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# GB- GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre learning_rate
#  Le taux d'apprentissage
# --------------------------------------------------------------------


def regle_gradboost_learningrate(learning_rate, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre learning_rate de GradientBoostRegressor
    Parameters
    ----------
    learning_rate :  Le taux d'apprentissage, obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in learning_rate:
        models[i] = GradientBoostingRegressor(learning_rate=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# GB- GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre max_features
# NOMBRE DE CARACTERISTIQUES
# --------------------------------------------------------------------    
    
    
def regle_gradboost_maxfeatures(max_features, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre max_features de RandomForestRegressor
    Parameters
    ----------
    max_features : NOMBRE DE CARACTERISTIQUES, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_features:
        models[str(i)] = GradientBoostingRegressor(max_features=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)    
 

 # --------------------------------------------------------------------
# GB- GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre criterion
# La fonction permettant de mesurer la qualité d'un fractionnement
# --------------------------------------------------------------------


def regle_gradboost_criterion(criterion, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre criterion de GradientBoostRegressor
    Parameters
    ----------
    criterion : La fonction permettant de mesurer la qualité d'un
    fractionnement, obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for s in criterion:
        models[s] = GradientBoostingRegressor(criterion=s, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# GB- GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre max_depth
# Profondeur maximale des estimateurs de régression individuels
# --------------------------------------------------------------------

def regle_gradboost_maxdepth(max_depth, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre criterion de GradientBoostRegressor
    Parameters
    ----------
    max_depth : Profondeur maximale des estimateurs de régression individuels,
    obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_depth:
        models[i] = GradientBoostingRegressor(max_depth=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# GB- GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre min_samples_split
# Le nombre minimum d'échantillons requis pour diviser un nœud interne.
# --------------------------------------------------------------------

def regle_gradboost_minsamplessplit(
        min_samples_split, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre min_samples_leaf de GradientBoostRegressor
    Parameters
    ----------
    min_samples_split : Le nombre minimum d'échantillons requis pour diviser un
    nœud interne., obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in min_samples_split:
        models[i] = GradientBoostingRegressor(
            min_samples_split=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# GB- GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre min_samples_leaf
# Le nombre minimum d'échantillons requis pour se trouver à un nœud de feuille.
# --------------------------------------------------------------------


def regle_gradboost_minsamplesleaf(
        min_samples_leaf, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre min_samples_leaf de GradientBoostRegressor
    Parameters
    ----------
    min_samples_leaf : Le nombre minimum d'échantillons requis pour se trouver
    à un nœud de feuille., obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in min_samples_leaf:
        models[i] = GradientBoostingRegressor(
            min_samples_leaf=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)
    
    
#--------------------------------------------------------------------
# ET - ExtaTreesRegressor - règle l'hyper-paramètre max_features
# nombre d'arbres
# --------------------------------------------------------------------


def regle_extratrees_nestimators(n_estimators, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre n_estimators de ExraTreesRegressor
    Parameters
    ----------
    n_estimators : nombre d'arbres, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    for n in n_estimators:
        models[str(n)] = ExtraTreesRegressor(n_estimators=n, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre max_features
# NOMBRE DE CARACTERISTIQUES
# --------------------------------------------------------------------


def regle_extratrees_maxfeatures(max_features, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre max_features de ExraTreesRegressor
    Parameters
    ----------
    max_features : NOMBRE DE CARACTERISTIQUES, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_features:
        models[str(i)] = ExtraTreesRegressor(max_features=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre min_samples_split
# nombre minimum d'échantillons requis pour diviser un nœud interne
# --------------------------------------------------------------------

def regle_extratrees_minsamplessplit(
        min_samples_split, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre min_samples_split de ExraTreesRegressor
    Parameters
    ----------
    min_samples_split : nombre minimum d'échantillons requis pour diviser un
    nœud interne, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in min_samples_split:
        models[str(i)] = ExtraTreesRegressor(
            min_samples_split=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre max_depth
# profondeur maximale de l'arbre
# --------------------------------------------------------------------

def regle_extratrees_maxdepth(max_depth, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre max_depth de ExraTreesRegressor
    Parameters
    ----------
    max_depth : profondeur maximale de l'arbre, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_depth:
        models[i] = ExtraTreesRegressor(max_depth=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre criterion
# mesurer la qualité d'un fractionnement
# --------------------------------------------------------------------

def regle_extratrees_criterion(criterion, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre criterion de ExraTreesRegressor
    Parameters
    ----------
    criterion : mesurer la qualité d'un fractionnement, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for s in criterion:
        models[s] = ExtraTreesRegressor(criterion=s, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre min_samples_leaf
# nombre minimum d'échantillons requis pour se trouver à un nœud de feuille
# --------------------------------------------------------------------

def regle_extratrees_minsamplesleaf(
        min_samples_leaf, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre min_samples_leaf de ExraTreesRegressor
    Parameters
    ----------
    min_samples_leaf : nombre minimum d'échantillons requis pour se trouver à
                       un nœud de feuille, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in min_samples_leaf:
        models[i] = ExtraTreesRegressor(min_samples_leaf=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre max_leaf_nodes
# Un nœud sera divisé si cette division induit une diminution de l'impureté
# supérieure ou égale à cette valeur
# --------------------------------------------------------------------

def regle_extratrees_maxleafnodes(max_leaf_nodes, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre max_leaf_nodes de ExraTreesRegressor
    Parameters
    ----------
    max_leaf_nodes : Un nœud sera divisé si cette division induit une
    diminution de l'impureté supérieure ou égale à cette valeur.
    Faire croître les arbres avec max_leaf_nodes de la manière la plus
    efficace possible. Les meilleurs nœuds sont définis comme une réduction
    relative de l'impureté. Si None, le nombre de nœuds feuilles est illimité,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_leaf_nodes:
        models[i] = ExtraTreesRegressor(max_leaf_nodes=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre min_impurity_decrease
# Un nœud sera divisé si cette division induit une diminution de l'impureté
# supérieure ou égale à cette valeur
# --------------------------------------------------------------------


def regle_extratrees_minimpuritydecrease(
        min_impurity_decrease, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre min_impurity_decrease de ExraTreesRegressor
    Parameters
    ----------
    min_impurity_decrease : Un nœud sera divisé si cette division induit une
    diminution de l'impureté plus grande ou égale à cette valeur., obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in min_impurity_decrease:
        models[i] = ExtraTreesRegressor(
            min_impurity_decrease=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre bootstrap
# Si les échantillons bootstrap sont utilisés lors de la construction
# des arbres. Si Faux, l'ensemble des données est utilisé pour construire chaque arbre.
# --------------------------------------------------------------------


def regle_extratrees_bootstrap(bootstrap, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre bootstrap de ExraTreesRegressor
    Parameters
    ----------
    bootstrap : Si les échantillons bootstrap sont utilisés lors de la construction des arbres.
    Si Faux, l'ensemble des données est utilisé pour construire chaque arbre,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for b in bootstrap:
        models[b] = ExtraTreesRegressor(bootstrap=b, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre warm_start
# Lorsqu'elle est définie sur True, la solution de l'appel précédent à
# l'ajustement est réutilisée et d'autres estimateurs sont ajoutés à l'ensemble,
# sinon, une nouvelle forêt est ajustée.
# --------------------------------------------------------------------


def regle_extratrees_warm_start(warm_start, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre warm_start de ExraTreesRegressor
    Parameters
    ----------
    warm_start : Lorsqu'elle est définie sur True, la solution de l'appel
    précédent à l'ajustement est réutilisée et d'autres estimateurs sont
    ajoutés à l'ensemble, sinon, une nouvelle forêt est ajustée., obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for b in warm_start:
        models[b] = ExtraTreesRegressor(warm_start=b, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre max_samples
# Si bootstrap est True, le nombre d'échantillons à tirer de X pour entraîner
# chaque estimateur de base
# --------------------------------------------------------------------


def regle_extratrees_maxsamples(max_samples, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre warm_start de ExraTreesRegressor
    Parameters
    ----------
    max_samples :  Si bootstrap est True, le nombre d'échantillons à tirer
    de X pour entraîner chaque estimateur de base, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_samples:
        models[i] = ExtraTreesRegressor(max_samples=i, bootstrap=True,
                                        random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre ccp_alpha
# Paramètre de complexité utilisé pour l'élagage minimal de complexité-coût.
# Le sous-arbre avec la plus grande complexité de coût qui est plus petite que
 # ccp_alpha sera choisi. Par défaut, aucun élagage n'est effectué
# --------------------------------------------------------------------

def regle_extratrees_ccpalpha(ccp_alpha, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre ccp_alpha de ExraTreesRegressor
    Parameters
    ----------
    ccp_alpha : Paramètre de complexité utilisé pour l'élagage minimal de
    complexité-coût. Le sous-arbre avec la plus grande complexité de coût qui
    est plus petite que ccp_alpha sera choisi. Par défaut, aucun élagage n'est
    effectué, obligatoire.
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in ccp_alpha:
        models[i] = ExtraTreesRegressor(ccp_alpha=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)
    

# --------------------------------------------------------------------
# XGB- EXTREME GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre n_estimators
# Le nombre d'étapes de boosting à exécuter
# --------------------------------------------------------------------

def regle_xgradboost_nestimators(n_estimators, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre ccp_alpha de Extreme Gradient Boosting Regressor
    Parameters
    ----------
    n_estimators : Le nombre d'étapes de boosting à exécuter, obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in n_estimators:
        models[i] = XGBRegressor(n_estimators=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)
    

# --------------------------------------------------------------------
# XGB- EXTREME GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre learning_rate
#  Le taux d'apprentissage
# --------------------------------------------------------------------


def regle_xgradboost_learningrate(learning_rate, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre learning_rate de Extreme Gradient Boosting Regressor
    Parameters
    ----------
    learning_rate :  Le taux d'apprentissage, obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in learning_rate:
        models[i] = XGBRegressor(learning_rate=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)
    
    
# --------------------------------------------------------------------
# XGB- EXTREME GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre max_depth
# Profondeur maximale des estimateurs de régression individuels
# --------------------------------------------------------------------

def regle_xgradboost_maxdepth(max_depth, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre criterion de GradientBoostRegressor
    Parameters
    ----------
    max_depth : Profondeur maximale des estimateurs de régression individuels,
    obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_depth:
        models[i] = XGBRegressor(max_depth=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)
    
    
# --------------------------------------------------------------------
# XGB- EXTREME GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre max_leaves
# Profondeur maximale des feuilles de régression individuels
# --------------------------------------------------------------------

def regle_xgradboost_maxleaves(max_depth, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre criterion de GradientBoostRegressor
    Parameters
    ----------
    max_depth : Profondeur maximale des estimateurs de régression individuels,
    obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_depth:
        models[i] = XGBRegressor(max_leaves=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)    
        
        

def regle_randomforest_nestimators(n_estimators, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre max_features de RandomForestRegressor
    Parameters
    ----------
    max_features : NOMBRE DE CARACTERISTIQUES, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in n_estimators:
        models[str(i)] = RandomForestRegressor(n_estimators=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


def regle_randomforest_maxfeatures(max_features, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre max_features de RandomForestRegressor
    Parameters
    ----------
    max_features : NOMBRE DE CARACTERISTIQUES, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_features:
        models[str(i)] = RandomForestRegressor(max_features=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

    
def regle_randomforest_maxdepth(max_depth, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre max_features de RandomForestRegressor
    Parameters
    ----------
    max_features : NOMBRE DE CARACTERISTIQUES, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_depth:
        models[str(i)] = RandomForestRegressor(max_depth=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)
    

# --------------------------------------------------------------------
# RANDOM FOREST REGRESSOR - règle l'hyper-paramètre min_samples_leaf
# nombre minimum d'échantillons requis pour se trouver à un nœud de feuille
# --------------------------------------------------------------------

def regle_randomforest_minsamplesleaf(
        min_samples_leaf, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre min_samples_leaf de RandomForestRegressor
    Parameters
    ----------
    min_samples_leaf : nombre minimum d'échantillons requis pour se trouver à
                       un nœud de feuille, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in min_samples_leaf:
        models[i] = RandomForestRegressor(min_samples_leaf=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)  
    
    
# --------------------------------------------------------------------
# RANDOM FOREST REGRESSOR - règle l'hyper-paramètre min_samples_split
# nombre minimum d'échantillons requis pour diviser un nœud interne
# --------------------------------------------------------------------

def regle_randomforest_minsamplessplit(
        min_samples_split, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre min_samples_split de RandomForestRegressor
    Parameters
    ----------
    min_samples_split : nombre minimum d'échantillons requis pour diviser un
    nœud interne, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in min_samples_split:
        models[str(i)] = RandomForestRegressor(
            min_samples_split=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

    
  

# --------------------------------------------------------------------
# RANDOM FOREST REGRESSOR - règle l'hyper-paramètre criterion
# mesurer la qualité d'un fractionnement
# --------------------------------------------------------------------

def regle_randomforest_criterion(criterion, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre criterion de RandomForestRegressor
    Parameters
    ----------
    criterion : mesurer la qualité d'un fractionnement, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for s in criterion:
        models[s] = RandomForestRegressor(criterion=s, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)
    

# --------------------------------------------------------------------
# RANDOM FOREST REGRESSOR - règle l'hyper-paramètre warm_start
# Lorsqu'elle est définie sur True, la solution de l'appel précédent à
# l'ajustement est réutilisée et d'autres estimateurs sont ajoutés à l'ensemble,
# sinon, une nouvelle forêt est ajustée.
# --------------------------------------------------------------------


def regle_randomforest_warm_start(warm_start, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre warm_start de RandomForestRegressor
    Parameters
    ----------
    warm_start : Lorsqu'elle est définie sur True, la solution de l'appel
    précédent à l'ajustement est réutilisée et d'autres estimateurs sont
    ajoutés à l'ensemble, sinon, une nouvelle forêt est ajustée., obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for b in warm_start:
        models[b] = RandomForestRegressor(warm_start=b, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)    
    

# --------------------------------------------------------------------
# RADNOM FOREST REGRESSOR - règle l'hyper-paramètre min_impurity_decrease
# Un nœud sera divisé si cette division induit une diminution de l'impureté
# supérieure ou égale à cette valeur
# --------------------------------------------------------------------


def regle_randomforest_minimpuritydecrease(
        min_impurity_decrease, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre min_impurity_decrease de RandomForestRegressor
    Parameters
    ----------
    min_impurity_decrease : Un nœud sera divisé si cette division induit une
    diminution de l'impureté plus grande ou égale à cette valeur., obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in min_impurity_decrease:
        models[i] = RandomForestRegressor(
            min_impurity_decrease=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)    


# --------------------------------------------------------------------
# RANDOM FOREST REGRESSOR - règle l'hyper-paramètre max_leaf_nodes
# Un nœud sera divisé si cette division induit une diminution de l'impureté
# supérieure ou égale à cette valeur
# --------------------------------------------------------------------

def regle_randomforest_maxleafnodes(max_leaf_nodes, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre max_leaf_nodes de RandomForestRegressor
    Parameters
    ----------
    max_leaf_nodes : Un nœud sera divisé si cette division induit une
    diminution de l'impureté supérieure ou égale à cette valeur.
    Faire croître les arbres avec max_leaf_nodes de la manière la plus
    efficace possible. Les meilleurs nœuds sont définis comme une réduction
    relative de l'impureté. Si None, le nombre de nœuds feuilles est illimité,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_leaf_nodes:
        models[i] = RandomForestRegressor(max_leaf_nodes=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)    
    
    
   
# --------------------------------------------------------------------
# RANDOM FOREST REGRESSOR - règle l'hyper-paramètre ccp_alpha
# Paramètre de complexité utilisé pour l'élagage minimal de complexité-coût.
# Le sous-arbre avec la plus grande complexité de coût qui est plus petite que
# ccp_alpha sera choisi. Par défaut, aucun élagage n'est effectué
# --------------------------------------------------------------------

def regle_randomforest_ccpalpha(ccp_alpha, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre ccp_alpha de RandomForestRegressor
    Parameters
    ----------
    ccp_alpha : Paramètre de complexité utilisé pour l'élagage minimal de
    complexité-coût. Le sous-arbre avec la plus grande complexité de coût qui
    est plus petite que ccp_alpha sera choisi. Par défaut, aucun élagage n'est
    effectué, obligatoire.
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in ccp_alpha:
        models[i] = RandomForestRegressor(ccp_alpha=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

 

# --------------------------------------------------------------------
# RANDOM FOREST REGRESSOR - règle l'hyper-paramètre bootstrap
# Si les échantillons bootstrap sont utilisés lors de la construction
# des arbres. Si Faux, l'ensemble des données est utilisé pour construire chaque arbre.
# --------------------------------------------------------------------


def regle_randomforest_bootstrap(bootstrap, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre bootstrap de RandomForestRegressor
    Parameters
    ----------
    bootstrap : Si les échantillons bootstrap sont utilisés lors de la construction des arbres.
    Si Faux, l'ensemble des données est utilisé pour construire chaque arbre,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for b in bootstrap:
        models[b] = RandomForestRegressor(bootstrap=b, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)
    

# --------------------------------------------------------------------
# RANDOM FOREST REGRESSOR - règle l'hyper-paramètre max_samples
# Si bootstrap est True, le nombre d'échantillons à tirer de X pour entraîner
# chaque estimateur de base
# --------------------------------------------------------------------


def regle_randomforest_maxsamples(max_samples, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre warm_start de ExraTreesRegressor
    Parameters
    ----------
    max_samples :  Si bootstrap est True, le nombre d'échantillons à tirer
    de X pour entraîner chaque estimateur de base, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_samples:
        models[i] = RandomForestRegressor(max_samples=i, bootstrap=True,
                                        random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)            
    
    
# --------------------------------------------------------------------
# -- Validation Croisée  - RANDOMIZED SEARCH CV 
# --------------------------------------------------------------------


def randomCV_search(regressors, pipeline, param_grid, X, y, n_splits=5, display=True) :
    
    """
    Optimise une liste d'estimateurs et retourne le meilleur
    regressors(dict): Dictionnaire d'estimateurs, {"name_estimator_1" : estimator_1}
    pipeline : pipeline de transformation de données
    param_grid(dict) : dictionnaire de dictionnaires de paramètres, {"name_estimator_1": dict_params}
    X(np.array): jeux de données
    y(np.array): prédiction
    
    
    """
    
    # Distribution des indices train/validation pour obtenir des jeux train/validation.
    # Divise le jeu de données en k partitions
    kf = KFold(n_splits=n_splits)
    
    rnd_search_dict = {}
    
    # Meilleur score
    best_score_rnd = float('inf')
    # Dictionnaire des modèles hyperparamétrés
    rmse_mean_rnd, rmse_std_rnd = {}, {}
    fit_time_mean_rnd, fit_time_std_rnd = {}, {}

    for name_reg, regressor in regressors.items() :
    
        print(name_reg)
        
        #Définition du pipeline pour chaque algorithmes (regresseurs)
        prepare_select_and_predict_pipeline = fonctions_models.define_pipeline(pipeline, regressor)

         
        # Instanciation des méthodes d'hyperparamétrisation (Randomized Search CV)
        rnd_search_prep = RandomizedSearchCV(prepare_select_and_predict_pipeline,
                                             param_grid[name_reg],
                                             n_iter=100,
                                             scoring='neg_mean_squared_error',
                                             cv=kf,
                                             random_state=42)
        
        
        # Entrainement du modèle (avec la méthode Randomized Search)
        rnd_search_prep.fit(X, y)
        
        
        # Sauvegarde des modèles
        
        rnd_search_dict[name_reg] = rnd_search_prep
        
        
        best_model_index_rnd = rnd_search_prep.best_index_
        score_rnd = [np.sqrt(-rnd_search_prep.cv_results_["split" + str(i) + "_test_score"][best_model_index_rnd])
                 for i in range(n_splits)]
        
        
        # Temp d'entrainement du modèle (avec la méthode Randomized Search)
        fit_time_mean_rnd[name_reg] = rnd_search_prep.cv_results_['mean_fit_time'][best_model_index_rnd]
        fit_time_std_rnd[name_reg] = rnd_search_prep.cv_results_['std_fit_time'][best_model_index_rnd]
        
        
        # Moyenne et écart-type de la RMSE des 2 méthodes
        mean_score_rnd, std_score_rnd = np.mean(score_rnd), np.std(score_rnd)
        rmse_mean_rnd[name_reg], rmse_std_rnd[name_reg] = mean_score_rnd, std_score_rnd
        score_rnd = rnd_search_prep.best_estimator_.score(X, y)
        
        
        print("Paramètres du meilleur modèle (RandomizedSearchCV) : \n")
        for (params_rnd, values_rnd) in rnd_search_prep.best_params_.items():
             print("\t - ", params_rnd, ":", values_rnd)   
        print("\nRésultats :")
        print("\n\t - r2 score (RandomizedSearchCV) sur le jeu d'entrainement : {:.3f}".format(score_rnd))
        print("\t - Mean of RMSE(RandomizedSearchCV) : {:.3f}".format(mean_score_rnd))
        print("\t - Ecart-type(RandomizedSearchCV) : {:.3f}".format(std_score_rnd))
        print("\t - Coefficient de variation(RandomizedSearchCV) : {:.2f} % \n".format(std_score_rnd/mean_score_rnd*100))
        
        if (mean_score_rnd < best_score_rnd):
            best_score_rnd = mean_score_rnd
            best_model_rnd = rnd_search_prep.best_estimator_
               
            
    if display:
            
        # Liste des couleurs pour chaque segment
        palette = sns.color_palette()
        colors = palette.as_hex()[0: len(regressors)]

        # Graphique des résultats pour la RMSE de l'hyper-paramétrisation
        fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 5))
        fig.suptitle("RMSE sur les jeux de validation", fontsize=18)

        axs[0].set_ylabel("Mean of RMSE", fontsize=13)
        axs[0].bar(x=list(rmse_mean_rnd.keys()), height=list(rmse_mean_rnd.values()), color=colors)
        axs[0].tick_params(axis='x', labelsize=13)

        axs[1].set_ylabel("Variance of RMSE", fontsize=13)
        axs[1].bar(x=list(rmse_std_rnd.keys()), height=list(rmse_std_rnd.values()), color=colors)
        axs[1].tick_params(axis='x', labelsize=13)
        plt.show()

        # Graphique des résultats pour la RMSE de l'hyper-paramétrisation
        fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 5))
        fig.suptitle("Temps d'entrainement des modèles", fontsize=18)

        axs[0].set_ylabel("Mean of Fit Time (s)", fontsize=13)
        axs[0].bar(x=list(fit_time_mean_rnd.keys()), height=list(fit_time_mean_rnd.values()), color=colors)
        axs[0].tick_params(axis='x', labelsize=13)


        axs[1].set_ylabel("Variance of Fit Time (s)", size=13)
        axs[1].bar(x=list(fit_time_std_rnd.keys()), height=list(fit_time_std_rnd.values()), color=colors)
        axs[1].tick_params(axis='x', labelsize=13)
        plt.show()
        
        
            
    return rnd_search_dict


# --------------------------------------------------------------------
# -- Validation Croisée  - GRID SEARCH CV 
# --------------------------------------------------------------------

def gridCV_search(regressors, pipeline, param_grid, X, y, n_splits=5, display=True) :
    
    """
    Optimise une liste d'estimateurs et retourne le meilleur
    regressors(dict): Dictionnaire d'estimateurs, {"name_estimator_1" : estimator_1}
    pipeline : pipeline de transformation de données
    param_grid(dict) : dictionnaire de dictionnaires de paramètres, {"name_estimator_1": dict_params}
    X(np.array): jeux de données
    y(np.array): prédiction
    
    
    """
    
    # Distribution des indices train/validation pour obtenir des jeux train/validation.
    # Divise le jeu de données en k partitions
    kf = KFold(n_splits=n_splits)
    
    grd_search_dict = {}
    
    # Meilleur score
    best_score_grd= float('inf')
    # Dictionnaire des modèles hyperparamétrés
    rmse_mean_grd, rmse_std_grd = {}, {}
    fit_time_mean_grd, fit_time_std_grd = {}, {}

    for name_reg, regressor in regressors.items() :
    
        print(name_reg)
        
        #Définition du pipeline pour chaque algorithmes (regresseurs)
        prepare_select_and_predict_pipeline = fonctions_models.define_pipeline(pipeline, regressor)

        
        grd_search_prep = GridSearchCV(prepare_select_and_predict_pipeline,
                                      param_grid[name_reg],
                                      n_jobs=-1,
                                      scoring='neg_mean_squared_error',
                                      cv=kf)
        
        
        # Entrainement du modèle (avec la méthode Grid Search)
        grd_search_prep.fit(X, y)
        
        # Sauvegarde des modèles
        
        grd_search_dict[name_reg] = grd_search_prep
        
        
        best_model_index_grd = grd_search_prep.best_index_
        score_grd = [np.sqrt(-grd_search_prep.cv_results_["split" + str(i) + "_test_score"][best_model_index_grd])
                   for i in range(n_splits)]
        
        
        # Temp d'entrainement du modèle (avec la méthode Grid Search)
        fit_time_mean_grd[name_reg] = grd_search_prep.cv_results_['mean_fit_time'][best_model_index_grd]
        fit_time_std_grd[name_reg] = grd_search_prep.cv_results_['std_fit_time'][best_model_index_grd]
        
        # Moyenne et écart-type de la RMSE des 2 méthodes
        mean_score_grd, std_score_grd = np.mean(score_grd), np.std(score_grd)
        rmse_mean_grd[name_reg], rmse_std_grd[name_reg] = mean_score_grd, std_score_grd
        score_grd = grd_search_prep.best_estimator_.score(X, y)
        
        
        print("Paramètres du meilleur modèle (GridSearchCV) : \n")
        for (params_grd, values_grd) in grd_search_prep.best_params_.items():
             print("\t - ", params_grd, ":", values_grd)   
        print("\nRésultats :")
        print("\n\t - r2 score (GridSearchCV) sur le jeu d'entrainement : {:.3f}".format(score_grd))
        print("\t - Mean of RMSE(GridSearchCV) : {:.3f}".format(mean_score_grd))
        print("\t - Ecart-type(GridSearchCV) : {:.3f}".format(std_score_grd))
        print("\t - Coefficient de variation(GridSearchCV) : {:.2f} % \n".format(std_score_grd/mean_score_grd*100))
        
            
    if display:
            
        # Liste des couleurs pour chaque segment
        palette = sns.color_palette()
        colors = palette.as_hex()[0: len(regressors)]

        # Graphique des résultats pour la RMSE de l'hyper-paramétrisation
        fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 5))
        fig.suptitle("RMSE sur les jeux de validation", fontsize=18)

        axs[0].set_ylabel("Mean of RMSE", fontsize=13)
        axs[0].bar(x=list(rmse_mean_grd.keys()), height=list(rmse_mean_grd.values()), color=colors)
        axs[0].tick_params(axis='x', labelsize=13)

        axs[1].set_ylabel("Variance of RMSE", fontsize=13)
        axs[1].bar(x=list(rmse_std_grd.keys()), height=list(rmse_std_grd.values()), color=colors)
        axs[1].tick_params(axis='x', labelsize=13)
        plt.show()

        # Graphique des résultats pour la RMSE de l'hyper-paramétrisation
        fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 5))
        fig.suptitle("Temps d'entrainement des modèles", fontsize=18)

        axs[0].set_ylabel("Mean of Fit Time (s)", fontsize=13)
        axs[0].bar(x=list(fit_time_mean_grd.keys()), height=list(fit_time_mean_grd.values()), color=colors)
        axs[0].tick_params(axis='x', labelsize=13)


        axs[1].set_ylabel("Variance of Fit Time (s)", size=13)
        axs[1].bar(x=list(fit_time_std_grd.keys()), height=list(fit_time_std_grd.values()), color=colors)
        axs[1].tick_params(axis='x', labelsize=13)
        plt.show()
        
            
    return grd_search_dict




# --------------------------------------------------------------------
# -- Modèles de régression - entraîner le modèle de base et scores
# --------------------------------------------------------------------


def comparer_baseline_regressors(
        X,
        y,
        cv=10,
        metrics=[
            'r2',
            'neg_mean_squared_error'],
        seed=21):
    """Comparaison rapide des modèles de régression de base.
    Parameters
    ----------
    X: Matrice x, obligatoire
    y: Target vecteur, obligatoire
    cv: le nombre de k-folds pour la cross validation, optionnel
    metrics: liste des scores à appliquer, optionnel
    seed: nombre aléatoire pour garantir la reproductibilité des données.
    Returns
    -------
    La liste des modèles avec les scores
    """
    # Les listes des modèles de régression de base (à enrichir)
    models = []
    models.append(('dum_mean', DummyRegressor(strategy='mean')))
    models.append(('dum_med', DummyRegressor(strategy='median')))
    models.append(('lin', LinearRegression()))
    models.append(
        ('ridge',
         Ridge(
             alpha=10,
             solver='cholesky',
             random_state=seed)))
    models.append(('lasso', Lasso(random_state=seed)))
    models.append(('en', ElasticNet(random_state=seed)))
    models.append(('svr', SVR()))
    models.append(('br', BayesianRidge()))
    models.append(('hr', HuberRegressor()))
    models.append(('omp', OrthogonalMatchingPursuit()))
    models.append(('lars', Lars(random_state=seed)))
    models.append(('knr', KNeighborsRegressor()))
    models.append(('dt', DecisionTreeRegressor(random_state=seed)))
    models.append(('ada', AdaBoostRegressor(random_state=seed)))
    models.append(('xgb', XGBRegressor(seed=seed)))
    models.append(('sgd', SGDRegressor(random_state=seed)))
    #models.append(('lgbm', LGBMRegressor(random_state=seed)))
    models.append(('rfr', RandomForestRegressor(random_state=seed)))
    models.append(('etr', ExtraTreesRegressor(random_state=seed)))
    models.append(('cat', CatBoostRegressor(random_state=seed, verbose=False)))
    models.append(('gbr', GradientBoostingRegressor(random_state=seed)))
    models.append(('bag', BaggingRegressor(random_state=seed)))

    # Création d'un dataframe stockant les résultats des différents algorithmes
    df_resultats = pd.DataFrame(dtype='object')
    for name, model in models:

        # Cross validation d'entraînement du modèle
        scores = pd.DataFrame(
            cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=metrics,
                return_train_score=True))

        # Sauvegarde des performances
        df_resultats = df_resultats.append(pd.DataFrame({
            'Modèle': [name],
            'Fit time': [scores['fit_time'].mean()],
            'Durée': [scores['score_time'].mean()],
            'Test R2 CV': [scores['test_r2'].mean()],
            'Test R2 +/-': [scores['test_r2'].std()],
            'Test MSE CV': [-(scores['test_neg_mean_squared_error'].mean())],
            'Train R2 CV': [scores['train_r2'].mean()],
            'Train R2 +/-': [scores['train_r2'].std()],
            'Train MSE CV': [-(scores['train_neg_mean_squared_error'].mean())]
        }), ignore_index=True)
        print(f'Exécution terminée - Modèle : {name}')

    return df_resultats.sort_values(
        by=['Test R2 CV', 'Test MSE CV', 'Durée'], ascending=False)







# -----------------------------------------------------------------------
# -- PLOT LES FEATURES IMPORTANCES
# -----------------------------------------------------------------------

def plot_features_importance(features_importance, nom_variables):
    '''
    Affiche le liste des variables avec leurs importances par ordre décroissant.
    Parameters
    ----------
    features_importance: les features importances, obligatoire
    nom_variables : nom des variables, obligatoire
    Returns
    -------
    None.
    '''
    # BarGraph de visalisation
    plt.figure(figsize=(6, 5))
    plt.barh(nom_variables, features_importance)
    plt.xlabel('Feature Importances (%)')
    plt.ylabel('Variables')
    plt.title('Comparison des Features Importances')
    plt.show()
    
    
# def plot_feature_importances(feature_importances, attributes):
    
#     # Les features sont triés par importance pour l'affichage graphique
#     features = np.array([[feature, attrib] for feature, attrib in sorted(zip(feature_importances, attributes), reverse=True)])
#     features_labels = [attrib for value, attrib in sorted(zip(feature_importances, attributes), reverse=True)]
#     features_values = [value for value, attrib in sorted(zip(feature_importances, attributes), reverse=True)]

#     # Affichage Bar Plot
#     fig = plt.figure(1, figsize=(25, 5))
#     plt.subplot(121)
#     sns.barplot(x=features_labels[0:20], y=[100*v for v in features_values[0:20]], orient='v')
#     plt.ylabel("%")
#     plt.title("Features Importances")
#     plt.xticks(rotation=90)
    
#     # Affichage Pieplot ENERGYSTARScore
#     plt.subplot(122)
#     feat_imp_energystarscore = features_values[features_labels.index('SiteEnergyUse(kBtu)')]
#     values = [1-feat_imp_energystarscore, feat_imp_energystarscore]
#     plt.pie(values, labels=["Other Features", "SiteEnergyUse(kBtu)"],
#             autopct='%1.1f%%')
#     plt.title("Importance relative de l'ENERGYSTARScore par rapport à l'ensemble des autres features")
#     plt.show()
    

# -----------------------------------------------------------------------
# -- PLOT LES SHAP VALUES
# -----------------------------------------------------------------------


def plot_shape_values(model, x_test):
    '''
    Affiche les SHAPE VALUES.
    Parameters
    ----------
    model: le modèle de machine learning, obligatoire
    x_test :le jeu de test de la matrice X, obligatoire
    Returns
    -------
    None.
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)

    shap.summary_plot(shap_values, x_test, plot_type="bar")

    shap.summary_plot(shap_values, x_test)

    # shap.initjs()
    # shap.force_plot(explainer.expected_value, shap_values[1,:], X_test_log.iloc[1,:])


# -----------------------------------------------------------------------
# -- EVALUE LE RESULTAT D'UN MODELE
# -----------------------------------------------------------------------

def evaluate(model, X_test, y_test):
    '''
    Evalue le résultat d'un modèle, MAE, RMSE, R2, accuracy'
    Parameters
    ----------
    model : modèle de machine learning à évaluer, obligatoire
    X_test : jeu de test matrice X, obligatoire
    y_test : jeu de test target y, obligatoire
    Returns
    -------
    accuracy : précision du modèle par rapport aux erreurs.

    '''
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print('\nPerformance du modèle :\n')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, predictions)
    print(f'mae={mae}')
    print(f'mse={mse}')
    print(f'rmse={rmse}')
    print(f'r2={r2}')

    return accuracy

# -----------------------------------------------------------------------
# -- TRACE LEARNING CURVE POUR VOIR L'OVER ou UNDER FITTING
# -----------------------------------------------------------------------


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Affiche la  learning curve pour je jeu de données de test et d'entraînement
    Parameters
    ----------
    estimator : object qui implemente les méthodes "fit" and "predict"
    title : string, titre du graphique
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training exemples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()



