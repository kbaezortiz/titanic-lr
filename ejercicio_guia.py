# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:37:39 2020

@author: Usuario
"""

#1. Entra en Kaggle.com y busca por el dataset Titanic. Acepta las reglas y descárgate el dataset. Carga en Python, a través de pandas, el csv llamado “train” y realiza un head para comprobar que has cargado correctamente los datos.

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os

directorio = "C:/Users/Usuario/Desktop/Bootcamp/Clase 25 11_02 OK/"
os.chdir(directorio)


train=pd.read_csv("train.csv")

train.head()
train.info()

#a. Cuenta cuantos pasajeros de cada clase hay en el dataset. Haz un barplot


train['Pclass'].count()
dt = train[['Pclass','Survived']]
dt.info()

#dt['Sex'].value_counts().plot(kind='bar', title='Pasajeros del Titanic')

#(100 * dt['Sex'].value_counts()/len(dt['Sex'])) .plot(kind='bar', title='Pasajeros del Titanic')

dt['Pclass'].value_counts().plot(kind='bar', title='Pasajeros del Titanic')

cross_tab=pd.crosstab(index=dt['Pclass'], columns=dt['Survived'], margins=True)
print(cross_tab)


#b. Calcula el tanto % de pasajeros de cada clase hay en el dataset. Haz un barplot.

(100 * dt['Pclass'].value_counts()/len(dt['Pclass'])) .plot(kind='bar', title='Pasajeros del Titanic')

#c. Haz una tabla de contingencia con la relación ​Pclass/Survived​, calcula el porcentaje de personas que están en cada celda.

cross_tab=pd.crosstab(index=dt['Pclass'], columns=dt['Survived'], margins=True)
print(cross_tab)


pd.crosstab(index=dt['Survived'], columns=dt['Pclass'], margins=True).apply(lambda r: r/len(dt)*100)


#d. Haz una tabla de porcentajes relativa, según la ​Pclass ​y según la variable Survived ​. De los pasajeros que sobrevivieron que % eran de la clase 2? De los pasajeros de la clase 3 que % sobrevivió? Haz un barplot de cada tabla de contingencia. Que nos dicen estos gráficos? Que nos dicen sobre las ​Pclass​?


plot = pd.crosstab(index=dt['Survived'], columns=dt['Pclass'] ).apply(lambda r: r/r.sum()*100, axis=0).plot(kind='bar')

#e. Calcula los odds de cada clase, interpreta los resultados

pd.crosstab(index=dt['Survived'], columns=dt['Pclass'], margins=True)


p_class1_vive=136/216 #total de clase a por el total de soprevivientes
p_class1_muere = 1-p_class1_vive
p_class2_vive=87/184
p_class2_muere = 1-p_class2_vive
p_class3_vive=119/491
p_class3_muere = 1-p_class3_vive


odd_clase1 = p_class1_vive/p_class1_muere
#tienen la posibilidad de 1,7 de sobrevivir si perteneces a la clase 1

odd_clase2 = p_class2_vive/p_class2_muere
#tienen la posibilidad de 0.896 de sobrevivir si perteneces a la clase 2

odd_clase3 = p_class3_vive/p_class3_muere
#tienen la posibilidad de 0.319 de sobrevivir si perteneces a la clase 3

#f. Calcula los odds ratio escogiendo como base la clase 3. Interpreta los resultados.

odd_clase1/odd_clase3 
#la posibilidad de sobrevivir si eres de la clase 1 es 5.314 veces a la probabilidad de sobrevivir en la clase 3
odd_clase2/odd_clase3
#la posibilidad de sobrevivir si eres de la clase 2 es 2.803 veces a la probabilidad de sobrevivir en la clase 3


#g. Haz una regresión logística con la variable ​Pclass. ​Coinciden los valores con los odds ratio estudiados anteriormente? Interpreta el incremento de odds ratio.

dt.Pclass = dt.Pclass.apply(lambda dt: str(dt))

# la funcion get_dummies nos transforma los strings en dummies
dt=pd.get_dummies(dt)

# cojemos la base de pclass_3
dt=dt.drop('Pclass_3', axis = 1)

clf = LogisticRegression(random_state=0, solver='lbfgs')


clf.get_params()
X = dt.drop('Survived', axis=1)
Y = dt['Survived']
clf=clf.fit(X, Y)

# coeficientes del modelo
clf.coef_
clf.intercept_


from math import e
from math import log

# odds de 3 clase
e**(clf.intercept_)
odd_clase3


e**(clf.coef_)*e**(clf.intercept_)
e**(clf.coef_+clf.intercept_)
odd_clase1 
odd_clase2


e**(clf.coef_)
odd_clase1/odd_clase3 
odd_clase2/odd_clase3



#3. Ahora analizaremos más generalmente el dataset:
#a. Cuantos NA hay en el dataset y en que columnas?
train.info()
np.sum(train.isnull())
train.keys()

summary = train.describe()
---------------------------
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
----------------------------

#b. Que nos dice la variable ​SibSp​ y ​Parch​? Cómo se distribuyen estas variables?

train.info()


train.SibSp # of siblings / spouses aboard the Titanic
train.Parch # of parents / children aboard the Titanic

#parientes 


#c. Estudia la función countplot del packete seaborn. Haz un countplots utilizando las columnas ​Pclass ​y ​Sex​.

f = sns.countplot (x='Pclass', data=train)
g = sns.countplot (x='Sex', data=train)

sns.countplot(x='Pclass', hue='Sex', data=train)


#d. Haz un histograma de la variable ​Age

x1=train.Age

plt.hist(x1,facecolor='purple')
plt.xlabel('Age')
plt.ylabel('Number of Passangers')
plt.title('Histograma de edad')
plt.show()


#e. Qué columnas se podrían descartar “en principio” de un modelo solo con observar que significan?





#f. La columna Cabin tiene muchos missings, con que podría tener relación esta columna? Crea una columna para decir si esta variable está informada. Haz un ‘group by’ con esta columna junto a otras variables para encontrar alguna posible relación.



train.describe(include = "all")


train.info()
train.isnull().sum() #687 nulos tiene Cabin

train['Cabin'].count()

train['Ticket'].head()
train['Fare'].head() #tarifa
train['Fare'].min() 



#g. Mira las relaciones que puede tener ​Embarked ​con ​Survived.

train[['Embarked','Survived']].count()
train[['Embarked','Survived']].isnull().sum()
missing_embarked = train['Embarked'].isnull().sum()

train.groupby('Survived').mean()


#4. Ahora vamos a ajustar modelos logístico a partir de las columnas : 'Survived','Pclass','Sex','Age','SibSp','Parch','Fare'.

#a. En las filas donde ​Age ​ sea NA introduce la media total


dt = train[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']]
dt.isnull().sum()

dt['Age'].mean()

dt['Age'] = dt['Age'].fillna(dt['Age'].mean()) 

dt.info()

train['Sex']


#b. Transforma la columna ​Pclass, Sex​ en strings.

dt.Pclass=dt.Pclass.apply(lambda x: str(x))
dt.info()


#c. Convierte ​Pclass, Sex ​ en dummies. Quita las columnas que escojas como variables base.
dt=pd.get_dummies(dt)

#d. Ajusta un modelo Logístico con todas las variables

# creamos un solver
clf = LogisticRegression(random_state=0, solver='lbfgs')

# parametros del modelo
clf.get_params()

# filtramos las clases base que escojemos (la Pclass_3 y el genero mujer)
X=dt.drop('Survived', axis=1)
X=X.drop('Pclass_3', axis=1)
X=X.drop('Sex_female', axis=1)
Y = dt['Survived']

#e. Que Accuracy?

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

metrics.accuracy_score(Y_pred, Y)








