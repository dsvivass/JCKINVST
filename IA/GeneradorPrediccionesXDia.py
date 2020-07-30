import os
# import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
# import seaborn as sns
import matplotlib.pyplot as plt
import math
import time
import datetime
import joblib
import re
import shutil
import warnings


warnings.filterwarnings("ignore") # IGNORA LOS WARNINGS!!



# FUNCIONES
tini = time.time()

def OrganizadorDfDia(archivo):


    df_dia = pd.read_csv(archivo, names = ['Fecha', 'P_alto', 'P_bajo', 'P_apert', 'P_cierre'], usecols=list(range(1,6)))
    FechaDividida = df_dia['Fecha'].str.split('  ', expand=True)
    df_dia['Fecha'] = FechaDividida[0].str.lstrip()
    df_dia['Hora'] = FechaDividida[1]
    df_dia['Ponderado'] = (df_dia['P_alto']+df_dia['P_bajo']+df_dia['P_apert']+df_dia['P_cierre'])/4 # ECUACION CAMBIANTE
    # df_dia['Ponderado'] = (df_dia['P_apert'] + df_dia['P_cierre']) / 2  # ECUACION CAMBIANTE
    df_pivot = df_dia.pivot(index='Fecha', columns='Hora', values='Ponderado')
    return df_pivot


def OrganizadorDfGeneral(dataFrame):
    '''Organiza por horas, limpia el DataFrame de las filas y columnas con valores NaN'''

    dataFrame.sort_index(axis=1, inplace=True)  # Organiza Las columnas por orden de horas
    # for i in dataFrame.index:
    #     if math.isnan(dataFrame.loc[i, '00:00:00']) == True or math.isnan(dataFrame.loc[i, '23:59:00']) == True:
    #         dataFrame.drop(i, inplace=True)
    # dataFrame.to_excel('/Users/dvs/Desktop/df.xlsx')
    return dataFrame.dropna(axis=1, how='any')


def MejorAjuste(gradoPol, alpha, cv, x, y):
    '''Ingresar los valores en lista, menos cv (int),\n Retorna grado Polinomio y coeficiente alfa que mejor se ajusta'''


    input = [('polinomio',PolynomialFeatures()), ('regresion', Ridge(solver='lsqr'))]
    parametros = [{'polinomio__degree':gradoPol, 'regresion__alpha':alpha}] # 0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000
    pipe = Pipeline(steps=input)
    grid = GridSearchCV(pipe, parametros, cv=cv, n_jobs=-1, return_train_score=True, verbose=10)
    grid.fit(x,y)
    scores = grid.cv_results_
    r2 = None
    for grado,param,mean_train,mean_test in zip(scores['param_polinomio__degree'], scores['param_regresion__alpha'],
                                                scores['mean_train_score'], scores['mean_test_score']): # , norm, scores['param_regresion__normalize'],
        if r2 is None or mean_test >= r2:
            r2 = mean_test
            gradoR = grado
            paramR = param
            mean_trainR = mean_train
            mean_testR = mean_test
    # print(grid.best_estimator_)
    print(f'[MEJOR AJUSTE CALCULADO: Grado polinomio: {gradoR}, alfa: {paramR}, media r2 entrenamiento: {mean_trainR}, '
          f'media r2 pruebas: {mean_testR}]') # , normalizado, {norm}
    return gradoR, paramR


def CadenaHoras(year=1995, month=3, day=28, HInicial=(8,30,0), HFinal=(8,30,0), paso_minutos = 5):
    '''Devuelve cadena de horas'''


    HActual = datetime.datetime(year=year, month=month, day=day, hour=HInicial[0], minute=HInicial[1], second=HInicial[2])
    HFinal = datetime.datetime(year=year, month=month, day=day, hour=HFinal[0], minute=HFinal[1], second=HFinal[2])
    delta = datetime.timedelta(minutes=paso_minutos)
    listaHoras = [str(HActual)[11:]]
    while HActual < HFinal:
        HActual = HActual + delta
        listaHoras.append(str(HActual)[11:])

    return listaHoras


def Graficar(DatosPredict, DatosReales):


    ax.plot(horas, DatosPredict, 'r-', label='Datos Inteligencia Artificial')
    ax.plot(horas, DatosReales, 'b-', label='Datos Reales')
    plt.xticks(rotation=90)
    plt.legend()
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='red', lw=4)]
    ax.legend(custom_lines, ['Datos Reales', 'Datos Inteligencia Artificial'])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    os.chdir(DirAct)
    plt.savefig('fig835.png')
    plt.show()


# LINEA PRINCIPAL
t0 = time.time()
DirAct = os.getcwd()
DirDatos = 'DatosHistoricos/AAPL'
os.chdir(DirDatos)
ls = sorted(os.listdir())

lista_1_ano = ls[-52:-51]
print(lista_1_ano)

lista_restante = [item for item in ls[:-52] if item not in lista_1_ano] # Restante de ls - lista_1_ano
print(lista_restante)

df = pd.DataFrame() # Creacion dataframe que almacena todos los datos
for file in lista_restante: # DataFrame Con los datos bases ~9 anos, para no crear en cada iteracion la misma tabla
    df_org = OrganizadorDfDia(file)
    df = df.append(df_org)

df = OrganizadorDfGeneral(df)

for num, f in enumerate(lista_1_ano):

    lista_gradual = lista_1_ano[:num+1]
    lista_total = ls[:-len(lista_1_ano)+num+1]
    ult_archivo = re.search(r'([0-9])+', lista_gradual[-1])

    df_org = OrganizadorDfDia(f)
    df = df.append(df_org) # Append archivo nuevo al df  con los datos base
    df = OrganizadorDfGeneral(df)

    print(df)

    os.chdir(DirAct)
    nuevo_nombre_file = 'Predicciones/XDia/AAPL_{}'.format(ult_archivo.group())
    try:
        os.mkdir(nuevo_nombre_file)
        os.chdir(nuevo_nombre_file)
    except Exception as e:
        print('[EXCEPCION: {}]'.format(e))
        print('[RESETEANDO CARPETA: {}]'.format(nuevo_nombre_file))
        shutil.rmtree(nuevo_nombre_file)
        os.mkdir(nuevo_nombre_file)
        os.chdir(nuevo_nombre_file)
        pass


    for f in os.listdir():
        os.remove(f)



    i, DatosReales, DatosPredict = 1, [], []
    horas = CadenaHoras(HInicial=(8,20,0), HFinal=(15,0,0))

    print('[PREDICCION {} de {}]'.format(num+1, len(lista_1_ano)))
    print(len(lista_1_ano)-num, f)
    for numero, hora in enumerate(horas):

        nombre_archivo = 'P{}_{}'.format(hora.replace(':',''), ult_archivo.group())
        if horas.index(hora) >= horas.index('08:30:00'):
            y = df[horas[horas.index('08:30:00'):]]
        else:
            y = df[horas[numero:]]
        # MejorGrado, MejorAlpha = MejorAjuste(gradoPol=list(range(1,3)),
        #                                      alpha=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1000], cv=20, x=x, y=y)

        input = [('polinomio',PolynomialFeatures(degree=2)), ('regresion', Ridge(alpha=1))]
        pipe = Pipeline(steps=input)
        Hfinal = tuple(map(int, hora.split(':')))
        x = df[CadenaHoras(HInicial=(4, 0, 0), HFinal=Hfinal, paso_minutos=5)]
        pipe.fit(x, y)
        joblib.dump(pipe, nombre_archivo+'.sav')
        joblib.dump(x.columns, nombre_archivo+'_columnas.sav')

    time.sleep(5)

    os.chdir(DirAct + '/' + DirDatos) # Se devuelve al directorio de datos
print('[TIEMPO TOTAL = {}]'.format(time.time() - tini))
