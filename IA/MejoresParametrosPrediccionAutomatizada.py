import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import math
import time
import datetime
import mplfinance as fplt
import statistics
import re
import joblib

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# FUNCIONES
tini = time.time()

def OrganizadorDfDia(archivo, horas=False):

    df_dia = pd.read_csv(archivo, names = ['Fecha', 'P_alto', 'P_bajo', 'P_apert', 'P_cierre', 'Volumen'], usecols=list(range(1,7)))
    FechaDividida = df_dia['Fecha'].str.split('  ', expand=True)
    df_dia['Fecha'] = FechaDividida[0].str.lstrip()
    df_dia['Hora'] = FechaDividida[1]
    df_dia['Ponderado'] = (df_dia['P_alto']+df_dia['P_bajo']+df_dia['P_apert']+df_dia['P_cierre'])/4 # ECUACION CAMBIANTE
    # df_dia['Ponderado'] = df_dia['P_cierre']
    if horas is not False: df_dia = df_dia[df_dia['Hora'].isin(horas)]
    df_pivot = df_dia.pivot(index='Fecha', columns='Hora', values='Ponderado')
    return df_pivot

def OrganizadorDfDia_horizontal(Data, df_org_base, df_dia, fecha, hora, ponderado):
    df_dia['Fecha'] = [fecha]
    df_dia['Hora'] = [hora]
    df_dia['Ponderado'] = [ponderado]  # Ya son todos los puntos de la vela, calculando el promedio
    Data = Data.append(df_dia)
    df_pivot = Data.pivot(index='Fecha', columns='Hora', values='Ponderado')
    df_org = pd.concat([df_org_base, df_pivot], axis=1)
    return Data, df_org


def OrganizadorDfGeneral(dataFrame):
    '''Organiza por horas, limpia el DataFrame de las filas y columnas con valores NaN'''

    # print('[ESTADO DATAFRAME: CREANDO...]')
    dataFrame.sort_index(axis=1, inplace=True)  # Organiza Las columnas por orden de horas
    # print('[ESTADO DATAFRAME: CREADO!]')
    return dataFrame.dropna(axis=1, how='any')


def MejorAjuste(gradoPol, alpha, cv, x, y):
    '''Ingresar los valores en lista, menos cv (int),\n Retorna grado Polinomio y coeficiente alfa que mejor se ajusta'''

    print('[ESTADO MEJOR AJUSTE: ANALIZANDO...]')
    input = [('polinomio',PolynomialFeatures()), ('regresion', Ridge(solver='lsqr'))]
    parametros = [{'polinomio__degree':gradoPol, 'regresion__alpha':alpha}] # 0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000
    pipe = Pipeline(steps=input)
    grid = GridSearchCV(pipe, parametros, cv=cv, n_jobs=-1, return_train_score=True)
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
    print(f'[ESTADO MEJOR AJUSTE: Grado polinomio: {gradoR}, alfa: {paramR}, media r2 entrenamiento: {mean_trainR}, '
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

    print('[ESTADO GRAFICO: GRAFICANDO...]')
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
    plt.grid()
    # plt.savefig('fig1015.png')

def pd_graf(horas = False):

    # for file in ls:
    df_dia = pd.read_csv(archivo_test, names=['Fecha', 'P_alto', 'P_bajo', 'P_apert', 'P_cierre', 'Volumen'],
                         usecols=list(range(1, 7)))
    FechaDividida = df_dia['Fecha'].str.split('  ', expand=True)
    df_dia['Fecha'] = FechaDividida[0].str.lstrip()
    df_dia['Hora'] = FechaDividida[1]
    if horas is not False: df_dia = df_dia[df_dia['Hora'].isin(horas)]
    return df_dia[['Hora', 'P_alto', 'P_bajo', 'P_apert', 'P_cierre', 'Volumen']] # .set_index(df_dia['Hora'])

# LINEA PRINCIPAL
DirBase = os.getcwd()
DirPred = 'Predicciones/XDia'
os.chdir(DirPred)
ls_pred = sorted(os.listdir())
os.chdir(DirBase)
DirDatos = 'DatosHistoricos/AAPL'
os.chdir(DirDatos)
ls = sorted(os.listdir())

#Chequeos

l_horas_base = CadenaHoras(HInicial=(4,0,0), HFinal=(8,25,0), paso_minutos=5)
l_horas_acum = CadenaHoras(HInicial=(8,30,0), HFinal=(15,0,0), paso_minutos=5)

ganancias_tot = []
for predictor,archivo_predecir in zip(ls_pred, ls[-len(ls_pred)+1:]):
    print('predictor ', predictor, 'archivo_predecir ', archivo_predecir)
    i, inversion = 0, False
    n_inversion = 1
    n_inversion_max = 4 # Inversion maxima
    decision = 'n'
    ganancias = 0
    hora_inicial = (8, 30, 0)
    hora_final = (9, 0, 0)
    df_org_base = OrganizadorDfDia(archivo_predecir, l_horas_base)
    fecha = re.search(r'([0-9]+)', archivo_predecir).group(1)
    # print(df_org_base)
    f = open(archivo_predecir)
    Data, df_5_min = pd.DataFrame(), pd.DataFrame()
    lista_min, lista_max = [], []
    os.chdir(DirBase + '/' + DirPred + '/' + predictor)
    ls_pred_dia = sorted(os.listdir())
    # print(ls_pred_dia)


    for linea in f: # Iteracion de cada barra
        # print(linea)
        hora = re.search(r'([0-9]+)  ([0-9/:]+)', linea).group(2)
        ls_pred_match = [i for i in ls_pred_dia if i.find(hora.replace(':','')) is not -1]
        if hora in l_horas_acum and (hora_final[0] < 11 and hora_final[1] < 59 and hora_final[2] < 59) and n_inversion <= n_inversion_max :
            linea_vector = linea.replace(' ', '').rstrip().split(',')
            Ponderado = (float(linea_vector[2])+float(linea_vector[3])+float(linea_vector[4])+float(linea_vector[5]))/4
            Data, df_org = OrganizadorDfDia_horizontal(Data, df_org_base, df_5_min, fecha, hora, Ponderado)

            nombre_archivo_actual = df_org.columns[-1].replace(':', '')
            # ls = sorted(os.listdir(os.getcwd()))

            # print(predictor)
            # print(archivo_predecir)
            # # print(ls_pred)
            # print(ls_pred_match)
            # print(fecha)
            # print(nombre_archivo_actual)

            DatosReales = []
            for hora in l_horas_acum:
                try:
                    DatosReales.append(df_org.loc[df_org.index[0], hora])
                except:
                    DatosReales.append(None)
            # print('-----------------------------------------')
            # print(DatosReales)

            # _________________________________________________________
            pipe = joblib.load(ls_pred_match[0])
            Predict = pipe.predict(df_org[joblib.load(ls_pred_match[1])])

            # Rango en el cual quiero analizar minimos y maximos
            horas_min_max = CadenaHoras(HInicial=hora_inicial, HFinal=hora_final)
            min_ind = l_horas_acum.index(horas_min_max[1])
            try:
                max_ind = l_horas_acum.index(horas_min_max[-1])
            except:
                max_ind = l_horas_acum.index('14:55:00')

            # print('-----------------------------------------')
            #
            # print('predict', Predict[0][:max_ind], len(Predict[0][:max_ind]))
            # # print('maximo', Predict[0][min_ind-1:max_ind].max())
            # print('Donde esta maximo', np.where(Predict[0][:max_ind] == Predict[0][min_ind:max_ind].max())[0][0])
            # # print('minimo', Predict[0][min_ind-1:max_ind].min())
            # print('Donde esta minimo', np.where(Predict[0][:max_ind] == Predict[0][min_ind:max_ind].min())[0][0])
            # print('min_ind', min_ind)
            # print('max_ind', max_ind)
            # print('Condicion min', np.where(Predict[0][:max_ind] == Predict[0][min_ind:max_ind].min())[0][0] >= i)
            # print('Condicion max', np.where(Predict[0][:max_ind] == Predict[0][min_ind:max_ind].max())[0][0] >= i)
            #
            # print('horas', l_horas_acum[:max_ind], len(l_horas_acum[:max_ind]))
            if np.where(Predict[0][:max_ind] == Predict[0][min_ind:max_ind].max())[0][0] >= i:  # and i>i_min
                lista_max.append(np.where(Predict[0][:max_ind] == Predict[0][min_ind:max_ind].max())[0][0])

            if np.where(Predict[0][:max_ind] == Predict[0][min_ind:max_ind].min())[0][0] >= i:  # and i>i_min
                lista_min.append(np.where(Predict[0][:max_ind] == Predict[0][min_ind:max_ind].min())[0][0])

            print('i: ', i)
            print('lista-min: ', lista_min)
            print('lista-max: ', lista_max)

            # print('-----------------------------------------')

            # __________________________________________________________________
            # plt.plot(l_horas_acum, Predict[0], 'b-', label='Datos Inteligencia Artificial')
            # plt.axvline(i, color='r')
            #
            # plt.plot(l_horas_acum, DatosReales, 'g-', label='Datos Reales')
            # plt.axvline('08:30:00', color='r')
            # plt.xticks(rotation=90)
            #
            # try:
            #     plt.axvline(statistics.mean(lista_max), color='k')
            # except:
            #     pass
            #
            # try:
            #     plt.axvline(statistics.mean(lista_min), color='y')
            # except :
            #     pass
            #
            #
            # try:
            #     plt.axvspan(statistics.mean(lista_max) - statistics.stdev(lista_max),
            #                 statistics.mean(lista_max) + statistics.stdev(lista_max), alpha=0.5, color='k')
            # except:
            #     pass
            #
            # try:
            #     plt.axvspan(statistics.mean(lista_min) - statistics.stdev(lista_min),
            #                 statistics.mean(lista_min) + statistics.stdev(lista_min), alpha=0.5, color='y')
            # except:
            #     pass
            #
            # plt.show()
            # time.sleep(2)

            #_____________________________________________________________________

            try:
                if tipo == 'C':
                    print('Ganancia parcial: ', Predict[0][i] - Predict[0][p_inv])
            except:
                pass

            try:
                if tipo == 'P':
                    print('Ganancia parcial: ', Predict[0][p_inv] - Predict[0][i])
            except:
                pass


            if inversion == True:
                # decision = input('Desea sacar el dinero (s/n): ')
                decision = 'n'


            try:

                if len(lista_min) == 1 and inversion == False:
                    if i >= statistics.mean(lista_min) and i <= statistics.mean(
                            lista_min) and inversion == False:
                        inversion = True
                        p_inv = i
                        tipo = 'C'
                        print('INVERSION CALL')

                elif (len(lista_max) == 1 or len(lista_min) == 1) and (i >= statistics.mean(
                        lista_max) or decision == 's') and inversion == True and tipo == 'C' :
                    p_ven = i
                    # plt.plot(list(range(len(Predict[0]))), Predict[0], color='blue')
                    # # plt.plot(list(range(len(Predict[0]))), df_org.values[0][-600:], color='green')
                    # plt.axvline(i, color='r')
                    # plt.axvline(statistics.mean(lista_max), color='k')
                    # plt.axvline(statistics.mean(lista_min), color='y')
                    # plt.axvspan(statistics.mean(lista_max),
                    #             statistics.mean(lista_max), alpha=0.5, color='k')
                    # plt.axvspan(statistics.mean(lista_min),
                    #             statistics.mean(lista_min), alpha=0.5, color='y')
                    #
                    # plt.axhspan(Predict[0][p_inv], Predict[0][p_ven], color='g')
                    # plt.title(str(i))
                    # plt.show()
                    # print('p_inv: ', p_inv, ', p_ven: ', p_ven)
                    # print('p_inv: ', l_horas_acum[p_inv], ', p_ven: ', l_horas_acum[p_ven])
                    # print('Dinero Pred: ', Predict[0][p_ven] - Predict[0][p_inv])
                    # print('Dinero Real: ', DatosReales[p_ven] - DatosReales[p_inv])
                    # print('Datos Reales: ', DatosReales)
                    ganancias = ganancias + DatosReales[p_ven] - DatosReales[p_inv]
                    tipo = None
                    hora_inicial = tuple(map(int, l_horas_acum[p_ven].split(':')))
                    hora_final = tuple(
                        map(int, CadenaHoras(HInicial=hora_inicial, HFinal=(23, 0, 0), paso_minutos=15)[1].split(':')))
                    # print(hora_inicial, hora_final)
                    lista_min, lista_max = [], []
                    inversion = False
                    n_inversion += 1
                    # time.sleep(10)

                elif i >= statistics.mean(lista_min) - statistics.stdev(lista_min) and i <= statistics.mean(
                        lista_min) + statistics.stdev(lista_min) and inversion == False:
                    inversion = True
                    p_inv = i
                    tipo = 'C'
                    print('INVERSION CALL')
                elif ((i >= statistics.mean(lista_max) - statistics.stdev(lista_max) and i <= statistics.mean(
                        lista_max) + statistics.stdev(lista_max)) or decision == 's') and inversion == True and tipo == 'C':
                    p_ven = i
                    # plt.plot(list(range(len(Predict[0]))), Predict[0], color='blue')
                    # # plt.plot(list(range(len(Predict[0]))), df_org.values[0][-600:], color='green')
                    # plt.axvline(i, color='r')
                    # plt.axvline(statistics.mean(lista_max), color='k')
                    # plt.axvline(statistics.mean(lista_min), color='y')
                    # plt.axvspan(statistics.mean(lista_max) - statistics.stdev(lista_max),
                    #             statistics.mean(lista_max) + statistics.stdev(lista_max), alpha=0.5, color='k')
                    # plt.axvspan(statistics.mean(lista_min) - statistics.stdev(lista_min),
                    #             statistics.mean(lista_min) + statistics.stdev(lista_min), alpha=0.5, color='y')
                    #
                    # plt.axhspan(Predict[0][p_inv], Predict[0][p_ven], color='g')
                    # plt.title(str(i))
                    # plt.show()
                    # print('p_inv: ', p_inv, ', p_ven: ', p_ven)
                    # print('p_inv: ', l_horas_acum[p_inv], ', p_ven: ', l_horas_acum[p_ven])
                    # print('Dinero Pred: ', Predict[0][p_ven] - Predict[0][p_inv])
                    # print('Dinero Real: ', DatosReales[p_ven] - DatosReales[p_inv])
                    # print('Datos Reales: ', DatosReales)
                    ganancias = ganancias + DatosReales[p_ven] - DatosReales[p_inv]
                    # print('Ganancias: ', ganancias)
                    tipo = None
                    hora_inicial = tuple(map(int, l_horas_acum[p_ven].split(':')))
                    hora_final = tuple(
                        map(int, CadenaHoras(HInicial=hora_inicial, HFinal=(23, 0, 0), paso_minutos=15)[1].split(':')))
                    # print(hora_inicial, hora_final)
                    lista_min, lista_max = [], []
                    inversion = False
                    n_inversion += 1
                    # time.sleep(10)

            except Exception as e:
                print('[EL PROBLEMA ES: call]', e)
                pass

            try:
                if len(lista_max) == 1 and inversion == False:
                    if i >= statistics.mean(lista_max) and i <= statistics.mean(lista_max) and inversion == False:
                        inversion = True
                        p_inv = i
                        tipo = 'P'
                        print('INVERSION PUT')
                        print('AAAAAAAAA1')

                elif (len(lista_max) == 1 or len(lista_min) == 1) and (i >= statistics.mean(
                        lista_min) or decision == 's') and inversion == True and tipo == 'P':
                    # print('Adentro')
                    p_ven = i
                    # print('len predict', len(Predict[0]))
                    # print('p_ven', p_ven)
                    # plt.plot(list(range(len(Predict[0]))), Predict[0], color='blue')
                    # # plt.plot(list(range(len(Predict[0]))), df_org.values[0][-600:], color='green')
                    # plt.axvline(i, color='r')
                    # plt.axvline(statistics.mean(lista_max), color='k')
                    # plt.axvline(statistics.mean(lista_min), color='y')
                    # plt.axvspan(statistics.mean(lista_max),
                    #             statistics.mean(lista_max), alpha=0.5, color='k')
                    # plt.axvspan(statistics.mean(lista_min),
                    #             statistics.mean(lista_min), alpha=0.5, color='y')
                    #
                    # plt.axhspan(Predict[0][p_inv], Predict[0][p_ven], color='g')
                    # plt.title(str(i))
                    # plt.show()
                    # print('p_inv: ', p_inv, ', p_ven: ', p_ven)
                    # print('p_inv: ', l_horas_acum[p_inv], ', p_ven: ', l_horas_acum[p_ven])
                    # print('Dinero Pred: ', Predict[0][p_inv] - Predict[0][p_ven])
                    # print('Dinero Real: ', DatosReales[p_inv] - DatosReales[p_ven])
                    # print('Datos Reales: ', DatosReales)
                    ganancias = ganancias + DatosReales[p_inv] - DatosReales[p_ven]
                    # print('Ganancias: ', ganancias)
                    tipo = None
                    hora_inicial = tuple(map(int, l_horas_acum[p_ven].split(':')))
                    hora_final = tuple(
                        map(int, CadenaHoras(HInicial=hora_inicial, HFinal=(23, 0, 0), paso_minutos=15)[1].split(':')))
                    # print(hora_inicial, hora_final)
                    lista_min, lista_max = [], []
                    inversion = False
                    n_inversion += 1
                    # time.sleep(10)
                    print('AAAAAAAAA2')

                elif i >= statistics.mean(lista_max) - statistics.stdev(lista_max) and i <= statistics.mean(
                        lista_max) + statistics.stdev(lista_max) and inversion == False:
                    inversion = True
                    p_inv = i
                    tipo = 'P'
                    print('INVERSION PUT')
                    print('i es: ', i)
                    print(statistics.mean(lista_max) - statistics.stdev(lista_max), statistics.mean(lista_max) + statistics.stdev(lista_max))
                    print('AAAAAAAAA3')


                elif ((i >= statistics.mean(lista_min) - statistics.stdev(lista_min)) or decision == 's') and inversion == True and tipo == 'P':
                    print('Adentro')
                    p_ven = i
                    print('len predict', len(Predict[0]))
                    print('p_ven', p_ven)
                    # plt.plot(list(range(len(Predict[0]))), Predict[0], color='blue')
                    # # plt.plot(list(range(len(Predict[0]))), df_org.values[0][-600:], color='green')
                    # plt.axvline(i, color='r')
                    # plt.axvline(statistics.mean(lista_max), color='k')
                    # plt.axvline(statistics.mean(lista_min), color='y')
                    # plt.axvspan(statistics.mean(lista_max) - statistics.stdev(lista_max),
                    #             statistics.mean(lista_max) + statistics.stdev(lista_max), alpha=0.5, color='k')
                    # plt.axvspan(statistics.mean(lista_min) - statistics.stdev(lista_min),
                    #             statistics.mean(lista_min) + statistics.stdev(lista_min), alpha=0.5, color='y')
                    #
                    # plt.axhspan(Predict[0][p_inv], Predict[0][p_ven], color='g')
                    # plt.title(str(i))
                    # plt.show()
                    # print('p_inv: ', p_inv, ', p_ven: ', p_ven)
                    # print('p_inv: ', l_horas_acum[p_inv], ', p_ven: ', l_horas_acum[p_ven])
                    # print('p_inv: ', l_horas_acum[p_inv], ', p_ven: ', l_horas_acum[p_ven])
                    # print('Dinero Pred: ', Predict[0][p_inv] - Predict[0][p_ven])
                    # print('Dinero Real: ', DatosReales[p_inv] - DatosReales[p_ven])
                    # print('Datos Reales: ', DatosReales)
                    ganancias = ganancias + DatosReales[p_inv] - DatosReales[p_ven]
                    # print('Ganancias: ', ganancias)
                    tipo = None
                    hora_inicial = tuple(map(int, l_horas_acum[p_ven].split(':')))
                    hora_final = tuple(map(int, CadenaHoras(HInicial=hora_inicial, HFinal=(23,0,0), paso_minutos=15)[1].split(':')))
                    # print(hora_inicial, hora_final)
                    lista_min, lista_max = [], []
                    inversion = False
                    n_inversion += 1
                    # time.sleep(10)
                    print('AAAAAAAAA4')

            except Exception as e:
                print('[EL PROBLEMA ES: ]', e)
                pass

            # time.sleep(2)

            i += 1

    ganancias_tot.append(ganancias)

    f.close()  # Cierro archivo
    os.chdir(DirBase + '/' + DirDatos)

print(ganancias_tot)
g = np.cumsum(ganancias_tot)
plt.plot(range(len(g)), g)
plt.title(n_inversion_max)
plt.show()