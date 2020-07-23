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


#VARIABLES GLOBALES
EstadoConexion = False
PrecioMkt = 0
Data = pd.DataFrame(columns=['contractDetails: ', 'Fecha: ', 'Punto alto: ', 'Punto bajo: ', 'Punto apertura: ', 'Punto cierre: '])
# Columnas
ColOrderStatus= ['ID de orden: ', 'Fecha: ', 'parentId: ', 'permId: ', 'clientId: ', 'Estado: ', 'Ejecutado: ', 'Pendiente: ',
                 'PrecioPromLLenado: ', 'UltimpoPrecioLLenado: '] # Columnas de df de OrderStatus
DataOrderStatus = pd.DataFrame(columns=ColOrderStatus)
DataOrderStatus.loc['0'] = 0
ult_bar_date = None
ult_Dat = list()
checker = None # Si hay una orden abierta es 1 sino es None
FechaActual = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
NombreDataFrame = f'DataFrame{FechaActual}.txt' # Nombre de Datos de barras
NombreDataFrameOrderStatus = 'OrdStat' + NombreDataFrame # Nombre de Datos de Estado de orden
NombreDataFrameOpenOrder = 'OpenOrd' + NombreDataFrame # Nombre de Datos de Orden Abierta
NombreDataFrameExecDetails = 'ExecDetails' + NombreDataFrame # Nombre de Datos de Detalles de Ejecucion

# CREACION DE DIRECTORIO PARA ESCRITURA DE ARCHIVOS .TXT
DirActual = os.getcwd() # DirActual
os.chdir('DatosInversionesHistoricas') # Cambio de directorio
os.mkdir(NombreDataFrame[:-4]) # Creacion de directorio
os.chdir(NombreDataFrame[:-4]) # Cambio a subdirectorio

class TestApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId, errorCode, errorString):
        global EstadoConexion
        EstadoConexion = True  # Variable de prueba para que el codigo entre solo
        # print('Error: ', reqId, ' ', errorCode, ' ', errorString)

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


    def historicalDataUpdate(self, reqId, bar):
        global Data, ult_bar_date, ult_Dat, checker, DataOrderStatus, PrecioMkt

        print('AcontractDetails: ', reqId, ' ', 'Fecha: ', self.FechaHoraActual(), ' ', 'Punto alto: ', bar.high, ' ', 'Punto bajo: ',
              bar.low, ' ', 'Punto apertura: ', bar.open, ' ', 'Punto cierre: ', bar.close, 'Volumen: ', bar.volume) # bar.date
        if bar.date != ult_bar_date and ult_bar_date is not None:
            self.DataFrameHistoricUpdate(ult_Dat) # Concatena los datos en el Dataframe creado previamente y en el .txt
            if Data.iloc[-1]['Fecha: '] == Data.iloc[-2]['Fecha: ']:
                Data = Data.drop(index = Data.index[len(Data) - 2]) # Borra fila en caso de haber un dato con fecha repetida

        ult_bar_date = bar.date
        ult_Dat = [reqId, bar.date, bar.high, bar.low, bar.open, bar.close]
        print('[CHECKER ES: {}]'.format(checker))
        # Condicion para llamar a metodo de Colocacion de ordenes
        if Data['Punto cierre: '].iloc[-1] > 1.075 and checker is None :
            PrecioMkt = Data['Punto cierre: '].iloc[-1]
            checker = 1
            # print('------DENTROOOOOO------')
            self.ColocarOrden(PrecioMkt) # Coloca una orden a precio MKT

        elif checker == 1 and (DataOrderStatus['Estado: '].iloc[-1] == 'Filled' or DataOrderStatus['Estado: '].iloc[-1] == 'Cancelled'):
            checker = None

        elif checker == 1:
            self.ActualizarOrden(PrecioMkt)

        plt.plot([0,1],[0,1], 'b-')
        plt.show()


            # contract = Contract()
            # contract.symbol = "EUR"
            # contract.secType = "CASH"
            # contract.exchange = 'IDEALPRO'
            # contract.currency = "USD"
            #
            # bracket = self.BracketOrder(self.nextValidId, "BUY", 100, 0, round(PrecioMkt*1.2,3), round(PrecioMkt*0.9,3))
            # for o in bracket:
            #     self.placeOrder(o.orderId, contract, o)
            #     self.nextValidId
            # subprocess.call(['afplay', 'nt2.m4a'])

        # elif Data['Punto cierre: '].iloc[-1] > 1.06:
        #     PrecioMkt = Data['Punto cierre: '].iloc[-1]
        #     print(PrecioMkt)
        #     contract = Contract()
        #     contract.symbol = "EUR"
        #     contract.secType = "CASH"
        #     contract.exchange = 'IDEALPRO'
        #     contract.currency = "USD"
        #
        #     bracket = self.BracketOrder(self.nextValidId, "BUY", 100, 0, round(PrecioMkt*1.02,3), round(PrecioMkt*0.98,3))
        #     for o in bracket:
        #         self.placeOrder(o.orderId, contract, o)
        #         self.nextValidId


    # -------------------------------------------------------- #
    #             ESTADO DE ORDENES EN EL MERCADO              #
    # -------------------------------------------------------- #

    def nextValidId(self, orderId):
        self.nextValidId = orderId
        # self.empezar()

    def orderStatus(self, orderId , status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId,
                    whyHeld, mktCapPrice):
        fecha = self.FechaHoraActual()
        # print('ID de orden: ', orderId, 'Fecha: ', fecha, 'parentId', parentId, 'permId', permId, 'clientId', clientId, 'Estado: ', status, 'Ejecutado: ',
        #       filled, 'Pendiente: ', remaining, 'PrecioPromEjecutado: ', avgFillPrice, 'UltimpoPrecioLLenado: ', lastFillPrice)
        DatosOrderStatus = [orderId, fecha, parentId,  permId, clientId, status, filled, remaining, avgFillPrice,  lastFillPrice]
        self.DataFrameOrderStatus(DatosOrderStatus) # Guarda los datos en un Dataframe y en un .txt

    def openOrder(self, orderId, contract, order, orderState):
        fecha = self.FechaHoraActual()
        # print('Orden abierta: ', orderId, 'Fecha: ', fecha, 'Contrato: ', contract.symbol, contract.secType, contract.currency, 'Estado de orden: ', orderState.status)
        DatosOpenOrder = [orderId, fecha, contract.symbol, contract.secType, contract.currency, orderState.status]
        self.DataFrameOpenOrder(DatosOpenOrder) # Guarda los datos en un .txt

    def execDetails(self, reqId, contract, execution):
        # print('ID: ', execution.orderId, 'Fecha: ', execution.time, 'Contrato: ', contract.symbol, contract.secType, contract.currency, 'Numero de intercambios: ', execution.shares,
        #       'Precio', execution.price)
        DatosExecDetails = [execution.orderId, execution.time, contract.symbol, contract.secType, contract.currency, execution.shares, execution.price]
        self.DataFrameExecDetails(DatosExecDetails) # Guarda los datos en un .txt

    # -------------------------------------------------------- #
    #            DETALLES DE OPCIONES EN EL MERCADO            #
    # -------------------------------------------------------- #

    # def securityDefinitionOptionParameter(self, reqId, exchange, underlyingConId, tradingClass, multiplier, expirations, strikes):
    #     super().securityDefinitionOptionParameter(reqId, exchange, underlyingConId, tradingClass, multiplier, expirations, strikes)
    #     print("SecurityDefinitionOptionParameter.",
    #     "ReqId:", reqId, "Exchange:", exchange, "Underlying conId:", underlyingConId, "TradingClass:", tradingClass, "Multiplier:", multiplier,
    #     "Expirations:", expirations, "Strikes:", str(strikes))
    #
    # def tickOptionComputation(self, tickerId, field, impliedVolatility, delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
    #     print('TICKOPCIONES',tickerId, field, impliedVolatility, delta, optPrice, pvDividend, gamma, vega, theta, undPrice)
    #
    # def tickPrice(self, tickerId, field, price, attribs):
    #     print('PRUEBAAA', 'A', tickerId, 'B', field,'C', price,'D', attribs)

    # -------------------------------------------------------- #
    #           COLOCACION DE ORDENES EN EL MERCADO            #
    # -------------------------------------------------------- #

    def ColocarOrden(self, PrecioMkt): # NOO ESTA EN USO
        # Tipo de contrato
        contract = Contract()
        # contract.symbol = "EUR"
        # contract.secType = "CASH"
        # contract.exchange = 'IDEALPRO'
        # contract.currency = "USD"

        contract.symbol = "AAPL"
        contract.secType = "OPT"
        contract.lastTradeDateOrContractMonth = '20200724'
        contract.exchange = 'SMART'
        contract.currency = "USD"
        contract.multiplier = '100'
        contract.strike = '387.5'
        contract.right = 'C'

        # contract.symbol = "AMD"
        # contract.secType = "STK"
        # contract.currency = "USD"
        # contract.exchange = "SMART"

        bracket = self.BracketOrder(self.nextValidId, "BUY", 1, 0, round(PrecioMkt*1.2,3), round(PrecioMkt*0.9,3))
        for o in bracket:
            # print('COLOCAR OOORDEN', o.orderId)
            self.placeOrder(o.orderId, contract, o)
            self.nextValidId

    def ActualizarOrden(self, PrecioMkt):
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CASH"
        contract.exchange = 'IDEALPRO'
        contract.currency = "USD"

        bracket = self.BracketOrder(self.nextValidId, "BUY", 10, 0, round(PrecioMkt * 1.2, 3),
                                    round(PrecioMkt * 0.9, 3))
        # print(bracket)
        for o in bracket:
            # print('ACTUALIZAR OOORDEN', o.orderId)
            self.placeOrder(o.orderId, contract, o)
            self.nextValidId

    def BracketOrder(self, parentOrderId, action, quantity, limitPrice, takeProfitLimitPrice, stopLossPrice):
        global checker
        '''BracketOrder(parentOrderId, action, quantity, limitPrice, takeProfitLimitPrice, stopLossPrice)'''

        #This will be our main or "parent" order
        parent = Order()
        parent.action = 'BUY'
        parent.orderType = "MKT"
        parent.totalQuantity = quantity
        # parent = Order()
        parent.orderId = parentOrderId
        # parent.action = action
        # parent.orderType = "LMT"
        # parent.totalQuantity = quantity
        # parent.lmtPrice = limitPrice
        #The parent and children orders will need this attribute set to False to prevent accidental executions.
        #The LAST CHILD will have it set to True,
        parent.transmit = False

        takeProfit = Order()
        takeProfit.orderId = parent.orderId + 1
        takeProfit.action = "SELL" if action == "BUY" else "BUY"
        takeProfit.orderType = "LMT"
        takeProfit.totalQuantity = quantity
        takeProfit.lmtPrice = takeProfitLimitPrice
        takeProfit.parentId = parentOrderId
        if checker is None:
            takeProfit.transmit = False
        else:
            takeProfit.transmit = True

        stopLoss = Order()
        stopLoss.orderId = parent.orderId + 2
        stopLoss.action = "SELL" if action == "BUY" else "BUY"
        stopLoss.orderType = "STP"
        #Stop trigger price
        stopLoss.auxPrice = stopLossPrice
        stopLoss.totalQuantity = quantity
        stopLoss.parentId = parentOrderId
        #In this case, the low side order will be the last child being sent. Therefore, it needs to set this attribute to True
        #to activate all its predecessors
        stopLoss.transmit = True

        bracketOrder = [parent, takeProfit, stopLoss]
        return bracketOrder


    # def dibujar_splines(self):
    #     global Data
    #     f = interp1d(list(Data.index), Data['Punto cierre: '], kind='cubic')
    #     plt.plot(list(Data.index), f(list(Data.index)), 'b-')
    #     plt.show()

    def stop(self):
        self.done = True
        self.disconnect()
        global Data
        global DataOrderStatus
        # Data = pd.DataFrame(Info).drop([0]).rename(columns =
        # {0 : 'contractDetails: ', 1 : 'Fecha: ', 2 : 'Punto alto: ', 3 : 'Punto bajo: ', 4 : 'Punto apertura: ', 5 : 'Punto cierre: '})
        # print(Data, DataOrderStatus, DataOrderStatus['ID de orden: '])
        f = interp1d(list(Data.index), Data['Punto cierre: '], kind='cubic')
        plt.plot(np.linspace(0, len(Data)-1, num=200), f(np.linspace(0, len(Data)-1, num=200)), 'b-')
        plt.show()

    # -------------------------------------------------------- #
    #          GUARDAR DATOS EN ARCHIVO DE TEXTO .TXT          #
    # -------------------------------------------------------- #

    def DataFrameHistoric(self, Dat): # Guarda en el DataFrame los datos historicos
        global Data, NombreDataFrame
        Data = Data.append({i:j for i,j in zip(list(Data.columns.values), Dat)}, ignore_index = True) #Concatena la lista de datos
                                                            # en el DataFrame
        with open(NombreDataFrame, 'a+') as archivo:
            archivo.write(f'{Dat[0]},{Dat[1]},{Dat[2]},{Dat[3]},{Dat[4]},{Dat[5]}\n')

    def DataFrameHistoricUpdate(self, ult_Dat): # Guarda en el DataFrame los datos actualizados (EN VIVO)
        global Data, NombreDataFrame
        Data = Data.append({i:j for i,j in zip(list(Data.columns.values), ult_Dat)}, ignore_index=True)  # Concatena la lista de datos

        with open(NombreDataFrame, 'a') as archivoUpd:
            archivoUpd.write(f'A{ult_Dat[0]},{ult_Dat[1]},{ult_Dat[2]},{ult_Dat[3]},{ult_Dat[4]},{ult_Dat[5]}\n')

    def DataFrameOrderStatus(self, DatosOrderStatus): # Guarda en el Dataframe los datos de los estados de orden
        global ColOrderStatus, DataOrderStatus, NombreDataFrameOrderStatus
        DataOrderStatus = DataOrderStatus.append({i:j for i,j in zip(ColOrderStatus, DatosOrderStatus)}, ignore_index=True) # Concatena la lista de datos
        # print('----------------------------------------------------------------------------------------------')
        # print(DataOrderStatus['Estado: '])

        with open(NombreDataFrameOrderStatus, 'a+') as archivo:
            archivo.write(f'{DatosOrderStatus[0]},{DatosOrderStatus[1]},{DatosOrderStatus[2]},{DatosOrderStatus[3]},'
                          f'{DatosOrderStatus[4]},{DatosOrderStatus[5]},{DatosOrderStatus[6]},{DatosOrderStatus[7]},'
                          f'{DatosOrderStatus[8]},{DatosOrderStatus[9]}\n')

    def DataFrameOpenOrder(self, DatosOpenOrder): # Guarda en el archivo .txt los datos de ordenes abiertas
        with open(NombreDataFrameOpenOrder, 'a+') as archivo:
            archivo.write(f'{DatosOpenOrder[0]},{DatosOpenOrder[1]},{DatosOpenOrder[2]},{DatosOpenOrder[3]},'
                          f'{DatosOpenOrder[4]},{DatosOpenOrder[5]}\n')

    def DataFrameExecDetails(self, DatosExecDetails):  # Guarda en el archivo .txt los datos de ordenes abiertas
        with open(NombreDataFrameExecDetails, 'a+') as archivo:
            archivo.write(f'{DatosExecDetails[0]},{DatosExecDetails[1]},{DatosExecDetails[2]},{DatosExecDetails[3]},'
                          f'{DatosExecDetails[4]},{DatosExecDetails[5]},{DatosExecDetails[6]}\n')

    # -------------------------------------------------------- #
    #                HORA Y FECHA (TIEMPO REAL)                #
    # -------------------------------------------------------- #

    def FechaHoraActual(self):
        return datetime.now().strftime('%d%m%Y  %H:%M:%S')

# def compra():
#     ColocarOrdenAvanzada.main()

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

    # contract.symbol = "AAPL"
    # contract.secType = "OPT"
    # contract.lastTradeDateOrContractMonth = '20200724'
    # contract.exchange = 'SMART'
    # contract.currency = "USD"
    # contract.multiplier = '100'
    # contract.strike = '387.5'
    # contract.right = 'C'

    # app.reqHistoricalData(1, contract, '', '2000 S', '1 secs', 'MIDPOINT', 0, 1, True, [])
    # time.sleep(10)

    contract1 = Contract()
    contract1.symbol = "AAPL"
    contract1.secType = "STK"
    contract1.currency = "USD"
    contract1.exchange = "SMART"
    app.reqHistoricalData(1, contract1, '', '1 D', '5 min', 'MIDPOINT', 0, 1, True, [])
    # app.reqContractDetails(7, contract.OptionForQuery())
    # app.reqSecDefOptParams(2, "AMD", "", "STK", 8314)
    # app.reqMktData(57, contract, "233", False, False, [])
    # Timer(20, app.stop).start()
    app.run()

    # if EstadoConexion is False: # Mira si entra al atributo error
    #     time.sleep(3)
    #     print('[INTENTANDO CONECTAR DE NUEVO]')
    #     main()


    # if EstadoConexion is False: # Mira si entra al atributo error
    #     main()

if __name__ == '__main__':
    # with concurrent.futures.ThreadPoolExecutor() as ejecutor:
    #     p1 = ejecutor.submit(main)
    main()
    # compra()
        # p2 = ejecutor.submit(compra)

