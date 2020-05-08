import os
# import numpy as np
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
    if horas is not False: df_dia = df_dia[df_dia['Hora'].isin(horas)]
    df_pivot = df_dia.pivot(index='Fecha', columns='Hora', values=['P_alto', 'P_bajo', 'P_apert', 'P_cierre', 'Volumen'])
    return df_pivot


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
DirAct = os.getcwd()
os.chdir('DatosHistoricos/AAPL6')
ls = sorted(os.listdir())
print(ls)

l_horas = CadenaHoras(HInicial=(4,0,0), HFinal=(8,30,0), paso_minutos=5)
contT, contF, = 0, 0
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
    for grado, alfa in zip([1],[0.00001]):

        i, DatosReales, DatosPredict = 1, [], []
        for hora in horas:
            tini2 = time.time()
            y = df.iloc[:,df.columns.get_level_values(1)==hora]
            # print(f'[ENCONTRANDO MEJOR GRADO Y ALFA: {i} de {len(horas)}')
            # MejorGrado, MejorAlpha = MejorAjuste(gradoPol=list(range(1,5)),
            #                                      alpha=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1000], cv=4, x=x, y=y)

            input = [('escala', StandardScaler()), ('polinomio',PolynomialFeatures(degree=grado)), ('regresion', Ridge(alpha=alfa))]
            pipe = Pipeline(steps=input)
            pipe.fit(x, y)
            Predict = pipe.predict(df_org[x.columns])

            DatosPredict.append(Predict[0][0])
            df_pred = df_pred.append(pd.DataFrame(Predict), ignore_index=True)
            i=i+1


        df_pred['Hora'] = horas
        df_pred.rename(columns={0: 'P_alto', 1: 'P_bajo', 2: 'P_apert', 3: 'P_cierre', 4: 'Volumen'}, inplace=True)
        df_pred = df_pred[['Hora', 'P_alto', 'P_bajo', 'P_apert', 'P_cierre', 'Volumen']]
        # print('df_pred', df_pred)
        # print(df_pred.columns, df_pred.iloc[0, 3], df_pred.iloc[0, 1], df_pred.iloc[0, 2], df_pred.iloc[0, 4])
        # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
        # dat.set_index(dat['Hora'], inplace=True)

        reformatted_data1 = dict()
        reformatted_data1['Date'] = []
        reformatted_data1['Open'] = []
        reformatted_data1['High'] = []
        reformatted_data1['Low'] = []
        reformatted_data1['Close'] = []
        reformatted_data1['Volume'] = []

        # print(dat, dat.index, dat.columns)
        # print('asas', dat.iloc[0,1])

        # datetime_str = '09/19/18 13:55:26'
        #
        # datetime_object = datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
        # ['Hora', 'P_alto', 'P_bajo', 'P_apert', 'P_cierre', 'Volumen']]

        for i in range(len(df_pred)):
            # reformatted_data1['Date'].append(datetime.datetime.fromtimestamp(i))
            reformatted_data1['Date'].append(datetime.datetime.strptime(df_pred.iloc[i, 0], '%H:%M:%S'))
            reformatted_data1['Open'].append(df_pred.iloc[i, 3])
            reformatted_data1['High'].append(df_pred.iloc[i, 1])
            reformatted_data1['Low'].append(df_pred.iloc[i, 2])
            reformatted_data1['Close'].append(df_pred.iloc[i, 4])
            reformatted_data1['Volume'].append(df_pred.iloc[i, 5])
            # reformatted_data['Volume'].append(dict['vol'])
        # print("reformatted data:", reformatted_data1)
        pdata1 = pd.DataFrame.from_dict(reformatted_data1)
        pdata1.set_index('Date', inplace=True)
        pdata1.loc[pdata1['Close'] >= pdata1['Open'], 'CondicionPred'] = 'Sube'
        pdata1.loc[pdata1['Close'] < pdata1['Open'], 'CondicionPred'] = 'Baja'
        # print('Predicho, pdata1: ', pdata1)

        dat = pd_graf(horas)
        # print('real: ', dat)
        reformatted_data = dict()
        reformatted_data['Date'] = []
        reformatted_data['Open'] = []
        reformatted_data['High'] = []
        reformatted_data['Low'] = []
        reformatted_data['Close'] = []
        reformatted_data1['Volume'] = []

        # print(dat, dat.index, dat.columns)
        # print('asas', dat.iloc[0,1])

        for i in range(len(dat)):
            reformatted_data['Date'].append(datetime.datetime.strptime(dat.iloc[i,0], '%H:%M:%S'))
            reformatted_data['Open'].append(dat.iloc[i, 3])
            reformatted_data['High'].append(dat.iloc[i, 1])
            reformatted_data['Low'].append(dat.iloc[i, 2])
            reformatted_data['Close'].append(dat.iloc[i, 4])
            reformatted_data1['Volume'].append(dat.iloc[i, 5])
            # reformatted_data['Volume'].append(dict['vol'])
        # print("reformatted data:", reformatted_data)
        pdata = pd.DataFrame.from_dict(reformatted_data)
        pdata.set_index('Date', inplace=True)

        pdata.loc[pdata['Close'] >= pdata['Open'], 'CondicionReal'] = 'Sube'
        pdata.loc[pdata['Close'] < pdata['Open'], 'CondicionReal'] = 'Baja'
        # print('Real, pdata1: ', pdata)



        # fplt.plot(
        #     pdata,
        #     type='candle',
        #     style='charles',
        #     title='REAL',
        #     ylabel='Price'
        # )
        #
        # fplt.plot(
        #     pdata1,
        #     type='candle',
        #     style='charles',
        #     title='PRED',
        #     ylabel='Price'
        # )

        cond = pd.DataFrame()
        cond['Comparacion'] = pdata['CondicionReal'] == pdata1['CondicionPred']
        Contador = cond['Comparacion'].value_counts()
        print(Contador.index[0], Contador.iloc[0])

        if Contador.index[0] == True: contT += 1
        else: contF += 1

print(f'Acertados: {contT}, Errados: {contF}')
print(f'[TIEMPO TOTAL DE EJECUCION: {time.time() - tini}: ')

    # plt.show()

