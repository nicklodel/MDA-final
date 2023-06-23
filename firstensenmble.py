from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn import tree, svm
from sklearn.model_selection import cross_validate
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn_evaluation import plot
import seaborn as sns
import time


data = arff.loadarff(fp)
df_iris = pd.DataFrame(data[0])
print(df_iris)
##Factorizamos el atributo nominal
df_iris['class'] = pd.factorize(df_iris['class'])[0]
print(df_iris)
#Conjuntos de entrenamiento
trainIris, testIris = train_test_split(df_iris, test_size=.25)

X_trainIris = trainIris.drop('class', axis=1)
y_trainIris = trainIris['class']
X_testIris = testIris.drop('class', axis=1)
y_testIris = testIris['class']


#Declaración de Bagging con Svm
clf_svm = svm.SVC()
clf_BaggingSVM = BaggingClassifier(estimator=clf_svm)


#Entrenamiento con validación cruzada de 10
Datasets = []
TimeTrain = []
TimeScore = []
TimeCV = []
ScoreBaggingSVM = []
init = time.time()
cv = cross_validate(clf_BaggingSVM,X_trainIris, y=y_trainIris,cv=10, n_jobs=-1)
end = time.time()
timeCV = end - init
print(f"\n--------- Iris ---------")
print(f"Score de la Validación Cruzada:\n   score = {np.mean(cv['test_score'])} +- {np.std(cv['test_score'])}")
print(f"Tiempo medio en ejecutarse el método (train): {np.mean(cv['fit_time'])} +- {np.mean(np.mean(cv['fit_time']))}s")
print(f"Tiempo medio en ejecutarse el método (score): {np.mean(cv['score_time'])} +- {np.mean(np.mean(cv['score_time']))}s")
print(f"Tiempo en ejecutarse la búsqueda {timeCV}s, ({timeCV/60} min)")
Datasets.append('Iris')
TimeTrain.append(np.mean(cv['fit_time']))
TimeScore.append(np.mean(cv['score_time']))
TimeCV.append(timeCV)
ScoreBaggingSVM.append(np.mean(cv['test_score']))

parametersSVMBagging = [
    {
    'n_estimators':[10], # Numero de estimators = 10 porque si no tarda demasiado
    "estimator__kernel": ["rbf"],
    "estimator__C": [0.01, 0.1],
    'estimator__gamma': [0.01, 0.1],
    #'max_samples': [0.75, 1]
    # Saltan warnings si aleatoriamente solo seleccionamos instancias de una clase. Podríamos ignorar dichos fits o capturar los warning. Eliminamos el problema directamente
    'max_features': [0.5, 0.75],
    'bootstrap': [True, False]
    }
]



SVMBag = GridSearchCV(estimator=clf_BaggingSVM, cv=10, param_grid=parametersSVMBagging, n_jobs=-1)



Datasets = []
TimeSearch = []
TimeMethod = []
ScoreBaggingSVM = []
for i in data:
    init = time.time()
    SVMBag.fit(X_trainIris, y_trainIris)
    end = time.time()
    timeSearch = end - init
    print(f"\n--------- Iris ---------")
    print(f"La mejor accuracy se obtuvo con el siguiente SVMBag:")
    print(f'    Best params -> {SVMBag.best_params_}')
    print(f'    Best score -> {SVMBag.best_score_}')

    print(f"Si usamos el dataset de test, obtenemos el siguiente resultado:")
    print(f"    score = {SVMBag.score(X_trainIris, y_trainIris)}")
    print(f"Tiempo medio en ejecutarse el método: {np.mean(SVMBag.cv_results_.get('mean_fit_time'))} +- {np.mean(SVMBag.cv_results_.get('std_fit_time'))}s")
    print(f"Tiempo en ejecutarse la búsqueda {timeSearch}s, ({timeSearch/60} min)")
    Datasets.append('Iris')
    TimeMethod.append(np.mean(SVMBag.cv_results_.get('mean_fit_time')))
    TimeSearch.append(timeSearch)
    ScoreBaggingSVM.append(SVMBag.score(X_testIris, y_testIris))



