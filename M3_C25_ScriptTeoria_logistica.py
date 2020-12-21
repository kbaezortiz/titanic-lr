# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:53:14 2019

@author: Karina
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os

# Extra: No forma parte del pipeline del modelado de datos
# una cosa mas del apply
table=np.arange(8).reshape((4,2))
df = pd.DataFrame(table, columns=['A', 'B'])
# apply aplica la funcion dada a cada elemento
df.apply(np.sqrt)
# si especificamos los ejes, nos lo hará por columnas o por filas
# axis=1 filas
df.apply(np.sum, axis=1)
# axis = 0 columnas
df.apply(np.sum, axis=0)

# esta lambda tiene como entrada un parametro cualquiera y devuelve una lista
# ahora tenemos una columna con una lista en cada elemento
df.apply(lambda x: [1, 2], axis=1)
# con result_type = 'expand' expandimos esta lista en diferentes columnas
df.apply(lambda x: [1, 2], axis=1, result_type='expand')




path='C:/Users/Usuario/Desktop/Bootcamp/Clase 25 11_02'
os.chdir(path)


train=pd.read_csv("train.csv")

# nombre columnas
train.columns

#info devuelve la información basica de las columnas
train.info()

# seleccionamos dos columnas para estudiar los odds ratio i las proporciones
dt=train[['Survived','Sex']]


# Miramos los datos
100 * dt['Sex'].value_counts() / len(dt['Sex'])

# podemos hacer un bar plot
dt['Sex'].value_counts().plot(kind='bar', title='Pasajeros del Titanic')

(100 * dt['Sex'].value_counts()/len(dt['Sex'])) .plot(kind='bar', title='Pasajeros del Titanic')


# tabla de contingencia
cross_tab=pd.crosstab(index=dt['Survived'], columns=dt['Sex'], margins=True)
print(cross_tab)

# tabla de contingencia % Proporciones/probabilidades de cada clase
pd.crosstab(index=dt['Survived'], columns=dt['Sex'], margins=True).apply(lambda r: r/len(dt)*100)

# tabla de contingencia % relativo por cada fila
pd.crosstab(index=dt['Survived'], columns=dt['Sex'] ).apply(lambda r: r/r.sum()*100, axis=1)
# de los que sobreviven cuantos son hombres
109/342*100
print(cross_tab)

# tabla de contingencia % relativo por cada fila
pd.crosstab(index=dt['Survived'], columns=dt['Sex'] ).apply(lambda r: r/r.sum()*100, axis=0)
# de los hombres, cuantos sobreviven
109/577*100

plot = pd.crosstab(index=dt['Survived'], columns=dt['Sex'] ).apply(lambda r: r/r.sum()*100, axis=0).plot(kind='bar')



pd.crosstab(index=dt['Survived'], columns=dt['Sex'], margins=True)

p_mujer_vive=233/314
p_mujer_muere=1-p_mujer_vive
p_hombre_vive=109/577
p_hombre_muere=1-p_hombre_vive
# odds mujer
odds_mujer=p_mujer_vive/p_mujer_muere
# Viven 2.87 mujeres por cada 1 que muere
odds_hombre=p_hombre_vive/p_hombre_muere
# Viven 0.23 hombres por cada uno que muere


# Odds ratio
odds_hombre/odds_mujer
# la probabilidad de encontrar un hombre (entre los hombres) que sobreviva es 0.08 veces la probabilidad de encontrar una mujer
odds_mujer/odds_hombre
# La probabilidad de encontrar una mujer (entre las mujeres) que sobreviva es 12.35 veces la probabilidad de encontrar a un hombre

# Hay funciones que calculan los odds ratio
import scipy.stats as stats
table=pd.crosstab(index=dt['Survived'], columns=dt['Sex'])
oddsratio, pvalue = stats.fisher_exact(table)
# no entraremos demasiado

# vamos al Modelo!!
# la funcion get_dummies nos transforma los strings en dummies
dt=pd.get_dummies(dt)
#dt=pd.get_dummies(dt) # drop_first=True

# al tirar Sex_female estamos escojiendo como base a las mujeres
dt=dt.drop('Sex_female', axis = 1)

# creamos un solver
clf = LogisticRegression(random_state=0, solver='lbfgs')

# parametros del modelo
clf.get_params()
X = dt.drop('Survived', axis=1)
Y = dt['Survived']
clf=clf.fit(X, Y)

# coeficientes del modelo
clf.coef_
clf.intercept_

from math import e
from math import log

# odds de la mujer
e**(clf.intercept_)
odds_mujer

# odds del hombre
e**(clf.coef_)*e**(clf.intercept_)
e**(clf.coef_+clf.intercept_)
odds_hombre

# odds ratio
e**(clf.coef_)
odds_hombre/odds_mujer


# Predecimos
Y_pred= clf.predict(X)
# confussion matrix

from sklearn.metrics import confusion_matrix
cfm = confusion_matrix(Y_pred, Y)
dt['Survived'].value_counts()

import seaborn as sns

g = sns.lmplot(x="Age", y="Survived", data=train, y_jitter=.02, logistic=True, truncate=False)
g = sns.lmplot(x="Age", y="Survived", col="Sex", row="Pclass", hue="Pclass", data=train, y_jitter=.02, logistic=True, truncate=False)

train.columns

from sklearn.metrics import accuracy_score

# la acuracy de un modelo lo podemos obtener a partir de dos funciones
# .score nos evalua unos datos sobre un modelos y calcula el error
clf.score(X,Y)
# accuracy_score calcula la accuracy entre dos vector uno de Y reales y uno de Y predecidas
accuracy_score(Y,Y_pred)


##############################
# un ejemplo con mas variables
##############################
dt=train[['Survived','Sex','Age','Pclass']]

dt.info()
# lo primero que hago es trnasformar la columna Pclass en una string
# lo hago porque asi me lo tratara como una clase
dt.Pclass=dt.Pclass.apply(lambda x: str(x))
# transformo las variables string en variables binarias
dt=pd.get_dummies(dt, drop_first=True)
# miro cuantos na hay en el dataset
np.sum(dt.isnull())

# filtramos los datos incompletos
dt=dt.dropna()

# creamos un solver
clf = LogisticRegression(random_state=0, solver='lbfgs')

# parametros del modelo
clf.get_params()

# filtramos las clases base que escojemos (la Pclass_3 y el genero mujer)
X=dt.drop('Survived', axis=1)
X=X.drop('Pclass_3', axis=1)
X=X.drop('Sex_female', axis=1)
Y = dt['Survived']
# Ajustamos el modelo segun los datos dados
clf=clf.fit(X, Y)
# Predecimos 
# la Y predecidaviene de las probabilidades y escogiendo directamente el threshold como 0.5
Y_pred=clf.predict(X)
# en lugar de querer la variable target como 0,1 lo queremos como probabilidades
# utilizamos la función predict_proba les objeto que hemos ajustado
probs=clf.predict_proba(X)


# nos devuelve la acuracy media del modelo
clf.score(X, Y)

# coeficientes del modelo
clf.coef_

clf.intercept_

# función para conseguir la accuracy media
accuracy_score(Y_pred, Y)


# Modulo propio LogisticReg
from LogisticReg import *
logreg=LogisticReg()
logreg.fit(X,Y)
logreg.z_scores
logreg.p_values

# Conseguir facilmente los p_values
import statsmodels.api as sm
X=sm.add_constant(X)
logit_model=sm.Logit(Y, X)
result=logit_model.fit()
print(result.summary())


# para conocer mejor el significado de Z value
# http://logisticregressionanalysis.com/1577-what-are-z-values-in-logistic-regression/

# roc curve
from sklearn import metrics

# devuelve la evolucion de:
# false-positive-rate de forma creciente (1-sensitivity)
# True-positive-rate de forma creciente (Sensitivity)
# threshold de forma decreciente
fpr, tpr, threshold = metrics.roc_curve(Y, probs[:,1])
roc_auc = metrics.auc(fpr, tpr)



# plots
import matplotlib.pyplot as plt
t = np.arange(0., 5., 0.2)
# podemos indicar al plot dibujar por tripletas
# primer elemento (eje X)
# segundo elemento (eje Y)
# tercer elemento (tipo de dibujo)
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()

plt.plot(t, t, 'r--', label='red')
plt.plot(t, t**2, 'bs', label='blue')
plt.plot(t, t**3, 'g^', label='green')
plt.legend(loc = 'upper left')
plt.show()


# plot curve roc
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# =============================================================================
# Multilabel model
# =============================================================================

from sklearn.datasets import load_iris


data = load_iris()
data.keys()
class_names =data['target_names']
data['target']
data['feature_names']
X=pd.DataFrame(data['data'],columns=data['feature_names'])
Y=data['target']


clf = LogisticRegression(random_state=0, solver='lbfgs',
                          multi_class='multinomial',max_iter=500).fit(X, Y)

y_pred = clf.fit(X, Y).predict(X)

pd.crosstab(index=Y, columns=y_pred, margins=True)



plot_confusion_matrix(Y, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')
