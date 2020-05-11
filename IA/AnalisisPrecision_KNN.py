import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import math
import time
import datetime
import mplfinance as fplt
from sklearn.neighbors import KNeighborsClassifier


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
    # df_dia['Ponderado'] = (df_dia['P_alto']+df_dia['P_bajo']+df_dia['P_apert']+df_dia['P_cierre'])/4 # ECUACION CAMBIANTE
    # df_dia['Ponderado'] = df_dia['P_cierre']
    # df_dia.loc[df_dia['P_cierre'] >= df_dia['P_apert'], 'Condicion'] = 'Sube'
    # df_dia.loc[df_dia['P_cierre'] < df_dia['P_apert'], 'Condicion'] = 'Baja'
    if horas is not False:
        df_dia = df_dia[df_dia['Hora'].isin(horas)]
    df_pivot = df_dia.pivot(index='Fecha', columns='Hora', values=['P_alto', 'P_bajo', 'P_apert', 'P_cierre', 'Volumen'])
    if horas is False:
        df_pivot.loc[df_pivot.iloc[[0]][('P_cierre', '08:30:00')] >= df_pivot.iloc[[0]][('P_apert', '08:30:00')], 'Condicion'] = 'Sube'
        df_pivot.loc[df_pivot.iloc[[0]][('P_cierre', '08:30:00')] < df_pivot.iloc[[0]][('P_apert', '08:30:00')], 'Condicion'] = 'Baja'
    return df_pivot


def OrganizadorDfGeneral(dataFrame):
    '''Organiza por horas, limpia el DataFrame de las filas y columnas con valores NaN'''

    # print('[ESTADO DATAFRAME: CREANDO...]')
    dataFrame.sort_index(axis=1, inplace=True)  # Organiza Las columnas por orden de horas
    # print('[ESTADO DATAFRAME: CREADO!]')
    return dataFrame.dropna(axis=1, how='any')


def MejorAjuste(vecino, cv, x, y):
    '''Ingresar los valores en lista, menos cv (int),\n Retorna grado Polinomio y coeficiente alfa que mejor se ajusta'''

    print('[ESTADO MEJOR AJUSTE: ANALIZANDO...]')
    input = [('escala', StandardScaler()), ('vecinos', KNeighborsClassifier())]
    parametros = [{'vecinos__n_neighbors':vecino}] # 0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000
    pipe = Pipeline(steps=input)
    grid = GridSearchCV(pipe, parametros, cv=cv, n_jobs=-1, return_train_score=True, scoring='accuracy')
    grid.fit(x,y)
    scores = grid.cv_results_
    print(scores)
    r2 = None
    for n_vecinos, mean_train, mean_test in zip(scores['param_vecinos__n_neighbors'],
                                                scores['mean_train_score'], scores[
                                                    'mean_test_score']):  # , norm, scores['param_regresion__normalize'],
        if r2 is None or mean_test >= r2:
            r2 = mean_test
            n_vecinosR = n_vecinos
            mean_trainR = mean_train
            mean_testR = mean_test
    # print(grid.best_estimator_)
    print(f'[ESTADO MEJOR AJUSTE: Numero de vecinos: {n_vecinosR}, media r2 entrenamiento: {mean_trainR}, '
          f'media r2 pruebas: {mean_testR}]')  # , normalizado, {norm}
    return n_vecinosR


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
DirAct = os.getcwd()
os.chdir('DatosHistoricos/AAPL6')
ls = sorted(os.listdir())
print(ls)

l_horas = CadenaHoras(HInicial=(4,0,0), HFinal=(8,25,0), paso_minutos=5)
contT, contF, acum= 0, 0, []
for i,archivo_test in enumerate(ls):

    print(f'[ANALIZANDO ARCHIVO {i} de {len(ls)}]')
    df, x = pd.DataFrame(), pd.DataFrame() # Creacion dataframe que almacena todos los datos
    for file in ls:
        if file != archivo_test:
            # print('El archivo de entrenamiento es: ', file)
            df_org = OrganizadorDfDia(file)
            df = df.append(df_org)
            df_org = OrganizadorDfDia(file, l_horas)
            x = x.append(df_org)
        # else:
        #     print('El archivo de prueba es: ', archivo_test)

    df = OrganizadorDfGeneral(df)
    # cor = df.corr('pearson')
    #
    # sns.heatmap(cor)
    # plt.show()

    # pd.set_option('display.max_columns', None)
    # df2 = pd.DataFrame()
    # for i in l_horas:
    #     # df2 = df.iloc[:, df.columns.get_level_values(1) == i]
    #     df2 = pd.concat([df2,df.iloc[:, df.columns.get_level_values(1) == i]], axis=1)


    # x = df.iloc[:, df.columns.get_level_values(1)== ['04:00:00', '04:05:00']]
    # print(x)
    # y = df[CadenaHoras(HInicial=(9,45,0), HFinal=(9,45,0), paso_minutos=10)]
    # MejorGrado, MejorAlpha = MejorAjuste(gradoPol=list(range(1,5)),
    #                                          alpha=[1], cv=20, x=x, y=y) # 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1000

    # os.chdir(DirAct)
    # os.chdir('DatosHistoricos/AAPL_TEST')
    # ls = sorted(os.listdir())

    # for file in ls:
    df_org = OrganizadorDfDia(archivo_test, l_horas)
    # print('df_org: ', df_org)

    # df_org = OrganizadorDfGeneral(df_org)

    from matplotlib.ticker import FormatStrFormatter
    df_pred = pd.DataFrame()

    horas = CadenaHoras(HInicial=(8,30,0), HFinal=(8,30,0), paso_minutos=5)
    # fig, ax = plt.subplots()
    for vecino in [61]:

        i, DatosReales, DatosPredict = 1, [], []
        for hora in horas:
            tini2 = time.time()
            y = df[['Condicion']].values.ravel()
            # print(f'[ENCONTRANDO MEJOR GRADO Y ALFA: {i} de {len(horas)}')
            # MejorVecino = MejorAjuste(vecino=list(range(1,180)), cv=20, x=x, y=y)

            input = [('escala', StandardScaler()), ('vecinos', KNeighborsClassifier(n_neighbors=vecino))]
            pipe = Pipeline(steps=input)
            pipe.fit(x, y)
            Predict = pipe.predict(df_org[x.columns])

            DatosPredict.append(Predict[0][0])
            df_pred = df_pred.append(pd.DataFrame(Predict), ignore_index=True)
            i=i+1

        df_actual = OrganizadorDfDia(archivo_test)
        # print(df_actual['Condicion'])
        # with pd.option_context('display.max_rows', None, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print(df_actual)
        print(df_pred[0].values == df_actual['Condicion'].values )

        if df_pred[0].values == df_actual['Condicion'].values:
            acum.append(1)
            contT += 1
        else:
            contF += 1
            acum.append(-1)


ac = np.cumsum(acum)
print(acum, ac)
print(f'Acertados: {contT}, Errados: {contF}')
print(f'[ACERTO UN {contT/(contT+contF)}]')
print(f'[TIEMPO TOTAL DE EJECUCION: {time.time() - tini}: ')
plt.plot(range(len(ac)), ac, 'g-o')
plt.xticks(np.arange(0,len(ac)-1, len(ac)/100), rotation=90)
plt.show()

    # plt.show()

