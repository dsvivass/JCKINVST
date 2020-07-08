#!/usr/bin/env python3

# import os
# import subprocess
# import sys
#
# print(os.environ) # Muestra todas las variables de entorno
# mi_env = os.environ.copy()
#
# mi_env['PATH'] = os.pathsep.join(['/Users/dvs/Documents/INGENIERIA_CIVIL/PYTHON/IB/twsapi_macunix/IBJts/source/pythonclient/ibapi', mi_env['PATH']])
# print(mi_env['PATH'])
# # sys.exit(mi_env['PATH'])
# # subprocess.run(args=['export', 'PATH={}'.format(mi_env['PATH'])]) # , env=mi_env
# # subprocess.run(args=['echo', 'hola'])

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from threading import Timer
import sys
import os

# raise ValueError('Error Intencional')

ult_bar_date = None
nombre_archivo = None
nombre_archivo2 = None
i = 0

# CREACION DE DIRECTORIO PARA ESCRITURA DE ARCHIVOS .TXT
DirActual = os.getcwd() # DirActual
os.chdir('DatosHistoricos') # Cambio de directorio
# os.chdir('IA/DatosHistoricos') # Cambio de directorio
carpeta = 'AAPL'
os.chdir(carpeta) # Cambio subdirectorio
EstadoConexion = False
# os.mkdir(NombreDataFrame[:-4]) # Creacion de directorio
# os.chdir(NombreDataFrame[:-4]) # Cambio a subdirectorio

class TestApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId, errorCode, errorString):
        global EstadoConexion
        EstadoConexion = True # Variable de prueba para que el codigo entre solo
        print('Error: ', reqId, ' ', errorCode, ' ', errorString)

    # -------------------------------------------------------- #
    # RECOLECCION DE DATOS HISTORICOS Y ACTUALIZACION DE DATOS #
    # -------------------------------------------------------- #

    def historicalData(self, reqId, bar):
        global ult_bar_date, nombre_archivo, nombre_archivo2, i
        print('contractDetails: ', reqId, ' ', 'Fecha: ', bar.date, ' ', 'Punto alto: ', bar.high, ' ', 'Punto bajo: ',
              bar.low, ' ', 'Punto apertura: ', bar.open, ' ', 'Punto cierre: ', bar.close, 'Volumen: ', bar.volume)
        if ult_bar_date != bar.date[:8] or ult_bar_date is None:
            ult_bar_date = bar.date[:8]
            nombre_archivo = f'DH{ult_bar_date}.csv'
            nombre_archivo2 = f'DH{ult_bar_date}K.csv'
            i = 0
        # self.DataFrameHistoric(reqId, bar.date, bar.high, bar.low, bar.open, bar.close) # Guarda los datos en un Dataframe y en un .txt
        if bar.volume == -1:
            with open(nombre_archivo, 'a+') as archivo:
                archivo.write(f'{reqId}, {bar.date}, {bar.high}, {bar.low}, {bar.open}, {bar.close}, {bar.volume}\n')
        else:
            with open(nombre_archivo, 'r') as archivo:
                lines = archivo.readlines()
                with open(nombre_archivo2, 'a+') as escribir:
                    # for line in archivo:
                    lines[i] = lines[i].rstrip()[:-2] + str(bar.volume) + '\n'
                    # print(line, file=escribir)
                    escribir.write(lines[i])
            i+=1

    def DataFrameHistoric(self, reqId, date, high, low, open, close): # Guarda en el DataFrame los datos historicos
        global nombre_archivo
        with open(nombre_archivo, 'a+') as archivo:
            archivo.write(f'{reqId},{date},{high},{low},{open},{close}\n')

def main():
    app = TestApp()
    app.connect('127.0.0.1', 7497, 0)
    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "STK"
    contract.currency = "USD"
    contract.exchange = "SMART"
    # contract.symbol = "EUR"
    # contract.secType = "CASH"
    # contract.exchange = 'IDEALPRO'
    # contract.currency = "USD"

    # '20130701 23:59:59 GMT'

    app.reqHistoricalData(1, contract, '', '2 D', '5 mins', 'MIDPOINT', 0, 1, True, [])  # MIDPOINT
    app.reqHistoricalData(1, contract, '', '2 D', '5 mins', 'TRADES', 0, 1, True, [])
    # Timer(5, sys.exit(101)).start()
    app.run()

    if EstadoConexion is False: # Mira si entra al atributo error
        main()

if __name__ == '__main__':
    main()
