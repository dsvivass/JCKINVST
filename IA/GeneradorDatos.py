from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import os

ult_bar_date = None
nombre_archivo = None
nombre_archivo2 = None
i = 0

# CREACION DE DIRECTORIO PARA ESCRITURA DE ARCHIVOS .TXT
DirActual = os.getcwd() # DirActual
os.chdir('DatosHistoricos') # Cambio de directorio
carpeta = 'AAPL6'
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

    app.reqHistoricalData(1, contract, '', '2 D', '5 mins', 'MIDPOINT', 0, 1, True, [])  # MIDPOINT
    app.reqHistoricalData(1, contract, '', '2 D', '5 mins', 'TRADES', 0, 1, True, [])
    app.run()

    if EstadoConexion is False: # Mira si entra al atributo error
        main()

if __name__ == '__main__':
    main()
