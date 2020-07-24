import os
import numpy as np
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
import statistics
import sys

# FUNCIONES
tini = time.time()

def OrganizadorDfDia(archivo):

    df_dia = pd.read_csv(archivo, names = ['Fecha', 'P_cierre'], usecols=[1,5])
    FechaDividida = df_dia['Fecha'].str.split('  ', expand=True)
    df_dia['Fecha'] = FechaDividida[0].str.lstrip()
    df_dia['Hora'] = FechaDividida[1]
    # df_dia['Ponderado'] = (df_dia['P_alto']+df_dia['P_bajo']+df_dia['P_apert']+df_dia['P_cierre'])/4 # ECUACION CAMBIANTE
    # df_dia['Ponderado'] = (df_dia['P_apert'] + df_dia['P_cierre']) / 2  # ECUACION CAMBIANTE
    df_pivot = df_dia.pivot(index='Fecha', columns='Hora', values='P_cierre')
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


def CadenaHoras(year=1995, month=3, day=28, HInicial=(8,30,0), HFinal=(8,35,0), paso_segundos = 1):
    '''Devuelve cadena de horas'''


    HActual = datetime.datetime(year=year, month=month, day=day, hour=HInicial[0], minute=HInicial[1], second=HInicial[2])
    HFinal = datetime.datetime(year=year, month=month, day=day, hour=HFinal[0], minute=HFinal[1], second=HFinal[2])
    delta = datetime.timedelta(seconds=paso_segundos)
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
df = pd.DataFrame() # Creacion dataframe que almacena todos los datos
DirAct = os.getcwd()
os.chdir('DatosHistoricos/AAPL_MINUTOS_2')
ls = sorted(os.listdir())

for file in ls:
    df_org = OrganizadorDfDia(file)
    df = df.append(df_org)

df = OrganizadorDfGeneral(df)
# df.to_excel('/Users/dvs/Desktop/df.xslx')


# x = df[CadenaHoras(HInicial=(8,30,0), HFinal=(8,31,0), paso_segundos=1)]
# y = df[CadenaHoras(HInicial=(9,45,0), HFinal=(9,45,0))]
#
# MejorGrado, MejorAlpha = MejorAjuste(gradoPol=list(range(1,20)),
#                                          alpha=[1], cv=4, x=x, y=y) # 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1000

os.chdir(DirAct)
os.chdir('DatosHistoricos/AAPL_TEST_MINUTOS')
ls = sorted(os.listdir())

for file in ls:
    df_org = OrganizadorDfDia(file)

# df_org = OrganizadorDfGeneral(df_org)

from matplotlib.ticker import FormatStrFormatter
fig, ax = plt.subplots()
i, DatosReales, DatosPredict = 1, [], []
horas = CadenaHoras(HInicial=(8,31,0), HFinal=(8,39,59))

# t0 = time.time()
#
# for hora in horas:
#     tini2 = time.time()
#     y = df[[hora]]
#     # MejorGrado, MejorAlpha = MejorAjuste(gradoPol=list(range(1,3)),
#     #                                      alpha=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1000], cv=20, x=x, y=y)
#
#     input = [('polinomio',PolynomialFeatures(degree=2)), ('regresion', Ridge(alpha=1))]
#     pipe = Pipeline(steps=input)
#     pipe.fit(x, y)
#     Predict = pipe.predict(df_org[x.columns])
#     DatosPredict.append(Predict[0][0])
#     # try:
#     #     DatosPredict.append(Predict[0][0])
#     # except:
#     #     DatosPredict.append(Predict.max)
#     try:
#         DatosReales.append(df_org.loc[df_org.index[0], hora])
#     except:
#         DatosReales.append(df_org.loc[df_org.index[0]].max())
#     # print(f'[MEJOR AJUSTE: {i} de {len(horas)}], Tiempo iteracion {time.time() - tini2}: ', )
#     i=i+1
#
# print(time.time()-t0)
# Graficar(DatosPredict, DatosReales)

t0 = time.time()
i=1
print('INICIO ANALISIS')
lista_max, lista_min = [], []
i_min = 10 # Numero minimo de segundos antes de analizar

inversion = False
p_inv, p_ven, tipo = None, None, None
for hora in CadenaHoras(HInicial=(8,31,0), HFinal=(8,39,59), paso_segundos=1):
    Hfinal = tuple(map(int,hora.split(':')))
    x = df[CadenaHoras(HInicial=(8,6,40), HFinal=Hfinal, paso_segundos=1)]
    y = df[horas]
    # print(y)
    input = [('polinomio',PolynomialFeatures(degree=1)), ('regresion', Ridge(alpha=1))]
    pipe = Pipeline(steps=input)
    pipe.fit(x, y)
    Predict = pipe.predict(df_org[x.columns])
    # print(Predict, Predict.max())
    # print(np.where(Predict[0] == Predict[0][598:].max())[0][0])
    # print(i)

    # print('max:', np.where(Predict[0] == Predict.max())[0][0])
    # print('min: ', np.where(Predict[0] == Predict.min())[0][0])
    # print('actual: ', i-1)

    # if np.where(Predict[0] == Predict[0][i-1:].max())[0][0] > i-1 and i>i_min:
    #     lista_max.append(np.where(Predict[0] == Predict[0][i-1:].max())[0][0])

    if np.where(Predict[0] == Predict[0].max())[0][0] > i-1 : # and i>i_min
        lista_max.append(np.where(Predict[0] == Predict[0].max())[0][0])

    # if np.where(Predict[0] == Predict[0][i-1:].min())[0][0] > i-1 and i>i_min:
    #     lista_min.append(np.where(Predict[0] == Predict[0][i-1:].min())[0][0])

    if np.where(Predict[0] == Predict[0].min())[0][0] > i-1 : # and i>i_min
        lista_min.append(np.where(Predict[0] == Predict[0].min())[0][0])

    # print('lista_max: ', lista_max)
    # print('lista_min: ', lista_min)
    # print(lista_max)
    # print(Predict)
    # print(list(range(len(Predict[0]))))

    # # if i>i_min and (i-1 > statistics.mean(lista_max) or i-1 > statistics.mean(lista_min)):
    #     ax1 = plt.plot(list(range(len(Predict[0]))), Predict[0], color='blue')
    #     plt.plot(list(range(len(Predict[0]))), df_org.values[0], color='green')
    #     plt.axvline(i, color='r')
    #     plt.axvline(statistics.mean(lista_max), color='k')
    #     plt.axvline(statistics.mean(lista_min), color='y')
    #     plt.title(str(i))
    #     plt.show()
    #     # exit()

    # if i > i_min and (i - 1 > statistics.mean(lista_max) or i - 1 > statistics.mean(lista_min)):
    # if i > i_min and  i - 1 > statistics.mean(lista_min):

    if i>0:
    # print(statistics.stdev(lista_max))
    #     print('i: es', i)
        # print(len(Predict[0]))
        # print(len(df_org.values[0][-600:]))


        # ax1 = plt.plot(list(range(len(Predict[0]))), Predict[0], color='blue')
        # plt.plot(list(range(len(Predict[0]))), df_org.values[0][-600:], color='green')
        # plt.axvline(i, color='r')
        # try:
        #     plt.axvline(statistics.mean(lista_max), color='k')
        #     plt.axvline(statistics.mean(lista_min), color='y')
        #     plt.axvspan(statistics.mean(lista_max)-statistics.stdev(lista_max), statistics.mean(lista_max)+statistics.stdev(lista_max), alpha=0.5, color='k')
        #     plt.axvspan(statistics.mean(lista_min)-statistics.stdev(lista_min), statistics.mean(lista_min)+statistics.stdev(lista_min), alpha=0.5, color='y')
        # except :
        #     continue


        try:
            # print(lista_max)
            # print(lista_min)
            # plt.axvline(statistics.mean(lista_max), color='k')
            # plt.axvline(statistics.mean(lista_min), color='y')
            # plt.axvspan(statistics.mean(lista_max)-statistics.stdev(lista_max), statistics.mean(lista_max)+statistics.stdev(lista_max), alpha=0.5, color='k')
            # plt.axvspan(statistics.mean(lista_min)-statistics.stdev(lista_min), statistics.mean(lista_min)+statistics.stdev(lista_min), alpha=0.5, color='y')
            #
            if i >= statistics.mean(lista_min)-statistics.stdev(lista_min) and i <= statistics.mean(lista_min)+statistics.stdev(lista_min) and inversion == False:
                inversion = True
                p_inv = i
                tipo = 'C'
                print('INVERSION CALL')
            elif i >= statistics.mean(lista_max)-statistics.stdev(lista_max) and i <= statistics.mean(lista_max)+statistics.stdev(lista_max) and inversion == True and tipo is not 'P':
                p_ven = i
                ax1 = plt.plot(list(range(len(Predict[0]))), Predict[0], color='blue')
                plt.plot(list(range(len(Predict[0]))), df_org.values[0][-600:], color='green')
                plt.axvline(i, color='r')
                plt.axvline(statistics.mean(lista_max), color='k')
                plt.axvline(statistics.mean(lista_min), color='y')
                plt.axvspan(statistics.mean(lista_max) - statistics.stdev(lista_max),
                            statistics.mean(lista_max) + statistics.stdev(lista_max), alpha=0.5, color='k')
                plt.axvspan(statistics.mean(lista_min) - statistics.stdev(lista_min),
                            statistics.mean(lista_min) + statistics.stdev(lista_min), alpha=0.5, color='y')

                plt.axhspan(Predict[0][p_inv], Predict[0][p_ven], color = 'g')
                plt.title(str(i))
                plt.show()
                print('p_inv: ', p_inv, ', p_ven: ', p_ven)
                time.sleep(1000)

            if i >= statistics.mean(lista_max)-statistics.stdev(lista_max) and i <= statistics.mean(lista_max)+statistics.stdev(lista_max) and inversion == False:
                inversion = True
                p_inv = i
                tipo = 'P'
                print('INVERSION PUT')

            elif i >= statistics.mean(lista_min)-statistics.stdev(lista_min) and i <= statistics.mean(lista_min)+statistics.stdev(lista_min) and inversion == True and tipo is not 'C':
                p_ven = i
                ax1 = plt.plot(list(range(len(Predict[0]))), Predict[0], color='blue')
                plt.plot(list(range(len(Predict[0]))), df_org.values[0][-600:], color='green')
                plt.axvline(i, color='r')
                plt.axvline(statistics.mean(lista_max), color='k')
                plt.axvline(statistics.mean(lista_min), color='y')
                plt.axvspan(statistics.mean(lista_max) - statistics.stdev(lista_max),
                            statistics.mean(lista_max) + statistics.stdev(lista_max), alpha=0.5, color='k')
                plt.axvspan(statistics.mean(lista_min) - statistics.stdev(lista_min),
                            statistics.mean(lista_min) + statistics.stdev(lista_min), alpha=0.5, color='y')

                plt.axhspan(Predict[0][p_inv], Predict[0][p_ven], color = 'g')
                plt.title(str(i))
                plt.show()
                print('p_inv: ', p_inv, ', p_ven: ', p_ven)
                time.sleep(1000)
                exit()

        except:
            continue




        # plt.title(str(i))
        # plt.show()
        # time.sleep(2)


    i+=1

print('FINAL ANALISIS')
print(time.time()-t0)

print(f'[TIEMPO TOTAL: {time.time() - tini}]')
