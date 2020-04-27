import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# FUNCIONES

def OrganizadorDfDia(archivo):
    df_dia = pd.read_csv(archivo, names = ['Fecha', 'P_alto', 'P_bajo', 'P_apert', 'P_cierre'], usecols=list(range(1,6)))
    FechaDividida = df_dia['Fecha'].str.split('  ', expand=True)
    df_dia['Fecha'] = FechaDividida[0].str.lstrip()
    df_dia['Hora'] = FechaDividida[1]
    df_dia['Ponderado'] = (df_dia['P_alto']+df_dia['P_bajo']+df_dia['P_apert']+df_dia['P_cierre'])/4 # ECUACION CAMBIANTE
    df_pivot = df_dia.pivot(index='Fecha', columns='Hora', values='Ponderado')
    return df_pivot

# LINEA PRINCIPAL
DirAct = os.getcwd()
os.chdir('DatosHistoricos/AAPL')
ls = sorted(os.listdir())
df = pd.DataFrame()

for file in ls:
    df_org = OrganizadorDfDia(file)
    df = df.append(df_org)

import math
df.sort_index(axis=1, inplace=True) # Organiza Las columnas por orden de horas
# for i in df.index:
#     if math.isnan(df.loc[i,'00:00:00']) == True or math.isnan(df.loc[i,'23:59:00']) == True:
#         df.drop(i, inplace=True)

df.dropna(axis=1, how='any', inplace=True)
print(df)

x = df[['09:00:00','09:05:00','09:10:00','09:15:00','09:20:00','09:25:00','09:30:00','09:35:00']] # ,'09:10:00','09:15:00','09:20:00','09:25:00','09:30:00','09:35:00'
y = df[['09:40:00']]
# df_corr = df.corr('pearson') # Me correlaciona todas las variables con el metodo que yo escoja
# sns.heatmap(data=df_corr, cmap='coolwarm')
# plt.show()


# input = [('polinomio',PolynomialFeatures()), ('regresion', Ridge())]
# parametros = [{'polinomio__degree':list(range(1,20)), 'regresion__alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}] # 0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000
# pipe = Pipeline(steps=input)
# grid = GridSearchCV(pipe, parametros, cv=4, n_jobs=-1, return_train_score=True)
# grid.fit(x,y)
# scores = grid.cv_results_
# print(scores)
# r2 = None
# for grado,param,mean_train,mean_test in zip(scores['param_polinomio__degree'],
#                                                  scores['param_regresion__alpha'],
#                                                  scores['mean_train_score'],
#                                                  scores['mean_test_score']): # ,norm,         scores['param_regresion__normalize'],
#     print(f'Grado polinomio: {grado}, alfa: {param}, media r2 entrenamiento: {mean_train}, '
#           f'media r2 pruebas: {mean_test}')  # , normalizado, {norm}
#     if r2 is None or mean_test >= r2:
#         r2 = mean_test
#         gradoR = grado
#         paramR = param
#         mean_trainR = mean_train
#         mean_testR = mean_test

# print(grid.best_estimator_)

# print(f'Grado polinomio: {gradoR}, alfa: {paramR}, media r2 entrenamiento: {mean_trainR}, '
#       f'media r2 pruebas: {mean_testR}') # , normalizado, {norm}


#
input = [('polinomio',PolynomialFeatures(degree=1)), ('regresion', Ridge(alpha=0.1))]
# parametros = [{'polinomio__degree':list(range(1,20)), 'regresion__alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}] # 0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000
pipe = Pipeline(steps=input)
os.chdir(DirAct)
os.chdir('DatosHistoricos/AAPL_TEST')
ls = sorted(os.listdir())
for file in ls:
    df_org = OrganizadorDfDia(file)

df_org.sort_index(axis=1, inplace=True) # Organiza Las columnas por orden de horas
# for i in df.index:
#     if math.isnan(df.loc[i,'00:00:00']) == True or math.isnan(df.loc[i,'23:59:00']) == True:
#         df.drop(i, inplace=True)
df_org.dropna(axis=1, how='any', inplace=True)
x = df[['09:00:00','09:05:00','09:10:00','09:15:00','09:20:00','09:25:00','09:30:00','09:35:00']] # ,'09:10:00','09:15:00','09:20:00','09:25:00','09:30:00','09:35:00'
# y = df[['09:40:00']]

from matplotlib.ticker import FormatStrFormatter
fig, ax = plt.subplots()
for col in df.columns:
    pipe.fit(x, df[col])
    # scores = grid.cv_results_
    # print(scores)
    Predict = pipe.predict(df_org[['09:00:00','09:05:00','09:10:00','09:15:00','09:20:00','09:25:00','09:30:00','09:35:00']])

    ax.plot(col,Predict,'ro', label='Datos Inteligencia Artificial')
    ax.plot(col,df_org[col],'bo', label='Datos Reales')
plt.xticks(rotation=90)
plt.legend()
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='red', lw=4)]

ax.legend(custom_lines, ['Datos Reales', 'Datos Inteligencia Artificial'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
plt.show()