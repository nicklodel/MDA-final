#Disable warning of Ripper implementation
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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


#lectura de los arffs
data = arff.loadarff('ionosphere.arff')
df_iono = pd.DataFrame(data[0])

data = arff.loadarff('diabetes.arff')
df_diabe = pd.DataFrame(data[0])

data = arff.loadarff('vehicle.arff')
df_Vehicle = pd.DataFrame(data[0])

data = arff.loadarff('vowel.arff')
df_vowel = pd.DataFrame(data[0])

data = arff.loadarff('iris.arff')
df_iris = pd.DataFrame(data[0])

data = arff.loadarff('letter.arff')
df_letter = pd.DataFrame(data[0])


# Changing the last categorical class value into a numerical value
df_iono['class'] = pd.factorize(df_iono['class'])[0]

# Changing the last categorical class value into a numerical value
df_diabe['class'] = pd.factorize(df_diabe['class'])[0]

# Changing the last categorical class value into a numerical value
df_Vehicle['Class'] = pd.factorize(df_Vehicle['Class'])[0]

# Changing the last categorical class value into a numerical value
df_vowel['Class'] = pd.factorize(df_vowel['Class'])[0]

# Changing the last categorical class value into a numerical value
df_iris['class'] = pd.factorize(df_iris['class'])[0]

# Changing the last categorical class value into a numerical value
df_letter['class'] = pd.factorize(df_letter['class'])[0]

#Entrenamientos
trainIono, testIono = train_test_split(df_iono, test_size=.25)
X_trainIono = trainIono.drop('class', axis=1)
y_trainIono = trainIono['class']
X_testIono = testIono.drop('class', axis=1)
y_testIono = testIono['class']

trainDiabe, testDiabe = train_test_split(df_diabe, test_size=.25)
X_trainDiabe = trainDiabe.drop('class', axis=1)
y_trainDiabe = trainDiabe['class']
X_testDiabe = testDiabe.drop('class', axis=1)
y_testDiabe = testDiabe['class']

trainVehicle, testVehicle = train_test_split(df_Vehicle, test_size=.25)
X_trainVehicle = trainVehicle.drop('Class', axis=1)
y_trainVehicle = trainVehicle['Class']
X_testVehicle = testVehicle.drop('Class', axis=1)
y_testVehicle = testVehicle['Class']

trainVowel, testVowel = train_test_split(df_vowel, test_size=.25)
X_trainVowel = trainVowel.drop('Class', axis=1)
y_trainVowel = trainVowel['Class']
X_testVowel = testVowel.drop('Class', axis=1)
y_testVowel = testVowel['Class']

trainIris, testIris = train_test_split(df_iris, test_size=.25)
X_trainIris = trainIris.drop('class', axis=1)
y_trainIris = trainIris['class']
X_testIris = testIris.drop('class', axis=1)
y_testIris = testIris['class']

trainLetter, testLetter = train_test_split(df_letter, test_size=.25)
X_trainLetter = trainLetter.drop('class', axis=1)
y_trainLetter = trainLetter['class']
X_testLetter = testLetter.drop('class', axis=1)
y_testLetter = testLetter['class']

#¿Diccionario de datasets?
data = [
    ('Ionosphere', X_trainIono, y_trainIono, X_testIono, y_testIono),
    ('Diabetes', X_trainDiabe, y_trainDiabe,X_testDiabe, y_testDiabe),
    ('Vehicle', X_trainVehicle, y_trainVehicle, X_testVehicle, y_testVehicle),
    ('Vowel', X_trainVowel, y_trainVowel, X_testVowel, y_testVowel),
    ('Iris', X_trainIris, y_trainIris, X_testIris, y_testIris)
]

#Instancias de los métodos
clf_tree = tree.DecisionTreeClassifier()
clf_svm = svm.SVC()
clf_BaggingTree = BaggingClassifier(estimator=clf_tree)
clf_AdaBoostSAMMETree = AdaBoostClassifier(estimator=clf_tree,algorithm='SAMME')
clf_AdaBoostSAMMERTree = AdaBoostClassifier(estimator=clf_tree,algorithm='SAMME.R')
clf_BaggingSVM = BaggingClassifier(estimator=clf_svm)
clf_AdaBoostSAMMESVM = AdaBoostClassifier(estimator=clf_svm,algorithm='SAMME')
clf_AdaBoostSAMMERSVM = AdaBoostClassifier(estimator=clf_svm,algorithm='SAMME.R')
clf_GradBoost = GradientBoostingClassifier()



#bucle de ejecución de la validación cruzada
Datasets = []
TimeTrain = []
TimeScore = []
TimeCV = []
ScoreTree = []
for i in data:
    init = time.time()
    cv = cross_validate(clf_tree,i[1], y=i[2],cv=10, n_jobs=-1)
    end = time.time()
    timeCV = end - init
    print(f"\n--------- {i[0]} ---------")
    print(f"Score de la Validación Cruzada:\n   score = {np.mean(cv['test_score'])} +- {np.std(cv['test_score'])}")
    print(f"Tiempo medio en ejecutarse el método (train): {np.mean(cv['fit_time'])} +- {np.mean(np.mean(cv['fit_time']))}s")
    print(f"Tiempo medio en ejecutarse el método (score): {np.mean(cv['score_time'])} +- {np.mean(np.mean(cv['score_time']))}s")
    print(f"Tiempo en ejecutarse la búsqueda {timeCV}s, ({timeCV/60} min)")
    Datasets.append(i[0])
    TimeTrain.append(np.mean(cv['fit_time']))
    TimeScore.append(np.mean(cv['score_time']))
    TimeCV.append(timeCV)
    ScoreTree.append(np.mean(cv['test_score']))

my_dict = dict(Dataset=Datasets,TimeTrain=TimeTrain, TimeScore=TimeScore,TimeCV=TimeCV, Score=ScoreTree)
SVMDF = pd.DataFrame (my_dict)
print(SVMDF.to_latex())

