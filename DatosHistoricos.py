import pandas as pd
import numpy as np
from ibapi.client import EClient
from ibapi.wrapper import EWrapper, BarData
from builtins import int
from ibapi.contract import Contract
from threading import Timer
import concurrent.futures
import ColocarOrdenAvanzada
import time
from ibapi.order import *

import psutil
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('TkAgg')

# %matplotlib notebook
# plt.rcParams['animation.html']='jshtml'

Data = pd.DataFrame(columns=['contractDetails: ', 'Fecha: ', 'Punto alto: ', 'Punto bajo: ', 'Punto apertura: ', 'Punto cierre: '])
ult_bar_date = None
ult_Dat = list()
checker = None
grafica_encendida = 0
i=0
x, y = [-2,-1], [1,1]
class TestApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId, errorCode, errorString):
        print('Error: ', reqId, ' ', errorCode, ' ', errorString)

    def historicalData(self, reqId, bar):
        global Data
        global compra
        global grafica_encendida
        print('contractDetails: ', reqId, ' ', 'Fecha: ', bar.date, ' ', 'Punto alto: ', bar.high, ' ', 'Punto bajo: ',
              bar.low, ' ', 'Punto apertura: ', bar.open, ' ', 'Punto cierre: ', bar.close)
        Dat = [reqId, bar.date, bar.high, bar.low, bar.open, bar.close] #Lista de datos del dia
        Data = Data.append({i:j for i,j in zip(list(Data.columns.values),Dat)}, ignore_index = True) #Concatena la lista de datos
                                                            # en el DataFrame

    def historicalDataUpdate(self, reqId, bar):
        global Data
        global ult_bar_date
        global ult_Dat
        global checker
        global ax
        global fig
        global grafica_encendida
        global i, x, y
        print('AcontractDetails: ', reqId, ' ', 'Fecha: ', bar.date, ' ', 'Punto alto: ', bar.high, ' ', 'Punto bajo: ',
              bar.low, ' ', 'Punto apertura: ', bar.open, ' ', 'Punto cierre: ', bar.close)
        if bar.date != ult_bar_date and ult_bar_date is not None:
            Data = Data.append({i:j for i,j in zip(list(Data.columns.values),ult_Dat)}, ignore_index = True) #Concatena la lista de datos
            if Data.iloc[-1]['Fecha: '] == Data.iloc[-2]['Fecha: ']:
                Data = Data.drop(index = Data.index[len(Data) - 2])
        ult_bar_date = bar.date
        ult_Dat = [reqId, bar.date, bar.high, bar.low, bar.open, bar.close]
        x.append(i)
        y.append(bar.close)
        # plt.plot(x,y)
        plt.draw()
        # fig.canvas.draw()
        i+=1

def stop():
    global x,y
        # self.done = True
        # self.disconnect()
        # global Data
        # print(Data)
    plt.plot(x,y)
    plt.show()

Timer(3, stop).start()

def main():
    global app
    app = TestApp()
    app.connect('127.0.0.1', 7497, 0)

    contract = Contract()
    contract.symbol = "EUR"
    contract.secType = "CASH"
    contract.exchange = 'IDEALPRO'
    contract.currency = "USD"
    app.reqHistoricalData(1, contract, '', '1 D', '1 min', 'MIDPOINT', 0, 1, True, [])
    app.run()

if __name__ == '__main__':
    main()
    
    
