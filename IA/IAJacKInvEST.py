# Primer programa piloto para realizar inversiones en LARGO
import pandas as pd
import numpy as np
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from threading import Timer
import concurrent.futures
import ColocarOrdenAvanzada
import time
from ibapi.order import *
import subprocess
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from datetime import datetime
import os

Data = pd.DataFrame(columns=['contractDetails: ', 'Fecha: ', 'Punto alto: ', 'Punto bajo: ', 'Punto apertura: ', 'Punto cierre: '])
FechaActual = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
NombreDataFrame = f'DataFrame{FechaActual}.txt' # Nombre de Datos de barras

class TestApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId, errorCode, errorString):
        print('Error: ', reqId, ' ', errorCode, ' ', errorString)

    # -------------------------------------------------------- #
    # RECOLECCION DE DATOS HISTORICOS Y ACTUALIZACION DE DATOS #
    # -------------------------------------------------------- #

    def historicalData(self, reqId, bar):
        global Data
        global compra
        print('contractDetails: ', reqId, ' ', 'Fecha: ', bar.date, ' ', 'Punto alto: ', bar.high, ' ', 'Punto bajo: ',
              bar.low, ' ', 'Punto apertura: ', bar.open, ' ', 'Punto cierre: ', bar.close, 'Volumen: ', bar.volume)
        Dat = [reqId, bar.date, bar.high, bar.low, bar.open, bar.close] #Lista de datos del dia
        self.DataFrameHistoric(Dat) # Guarda los datos en un Dataframe y en un .txt

    def DataFrameHistoric(self, Dat): # Guarda en el DataFrame los datos historicos
        global Data, NombreDataFrame
        Data = Data.append({i:j for i,j in zip(list(Data.columns.values), Dat)}, ignore_index = True) #Concatena la lista de datos
                                                            # en el DataFrame
        with open(NombreDataFrame, 'a+') as archivo:
            archivo.write(f'{Dat[0]},{Dat[1]},{Dat[2]},{Dat[3]},{Dat[4]},{Dat[5]}\n')

def main():
    app = TestApp()
    app.connect('127.0.0.1', 7497, 0)
    contract = Contract()
    # contract.symbol = "AMD"
    # contract.secType = "STK"
    # contract.currency = "USD"
    # contract.exchange = "SMART"
    contract.symbol = "EUR"
    contract.secType = "CASH"
    contract.exchange = 'IDEALPRO'
    contract.currency = "USD"

    app.reqHistoricalData(1, contract, '', '1 M', '1 min', 'MIDPOINT', 0, 1, True, [])
    app.run()

if __name__ == '__main__':
    main()

