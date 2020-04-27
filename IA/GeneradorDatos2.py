from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import os
import pandas as pd

ult_bar_date = None
nombre_archivo = None

# CREACION DE DIRECTORIO PARA ESCRITURA DE ARCHIVOS .TXT
DirActual = os.getcwd() # DirActual
os.chdir('DatosHistoricos') # Cambio de directorio
os.chdir('AAPL') # Cambio subdirectorio
EstadoConexion = False
# os.mkdir(NombreDataFrame[:-4]) # Creacion de directorio
# os.chdir(NombreDataFrame[:-4]) # Cambio a subdirectorio
# df = pd.read_csv('NASDAQ_AAPL.csv', names=)
# print(df.head)
with open('NASDAQ_AAPL.csv', 'r') as leer:
    lectura = leer.read().split('\n')
    for n,linea in enumerate(lectura):
        print(n,linea)
        if n == 0:
            pass
        else:
            lista = linea.split(',')
            if ult_bar_date != lista[1][:8] or ult_bar_date is None:
                ult_bar_date = lista[1][:8]
                nombre_archivo = f'DH{ult_bar_date}.csv'
            # self.DataFrameHistoric(reqId, bar.date, bar.high, bar.low, bar.open, bar.close) # Guarda los datos en un Dataframe y en un .txt
            with open(nombre_archivo, 'a+') as archivo:
                archivo.write(f'{lista[0]}, {lista[1][:8]}  {lista[1][8:10]}:{lista[1][10:]}:00, {lista[2]}, {lista[3]}, {lista[4]}, {lista[5]},\n')


            # def DataFrameHistoric(self, reqId, date, high, low, open, close):  # Guarda en el DataFrame los datos historicos
            #     global nombre_archivo
            #     with open(nombre_archivo, 'a+') as archivo:
            #         archivo.write(f'{reqId},{date},{high},{low},{open},{close}\n')

    



