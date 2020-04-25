from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *
from threading import Timer

class TestApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId, errorCode, errorString):
        print('Error: ', reqId, ' ', errorCode, ' ', errorString)

    def nextValidId(self, orderId):
        self.nextValidId = orderId
        self.empezar()

    def orderStatus(self, orderId , status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId,
                    whyHeld, mktCapPrice):
        print('ID de orden: ', orderId, 'Estado: ', status, 'Lleno: ', filled, 'Pendiente: ', remaining, 'PrecioPromLLenado: ',
        avgFillPrice, 'UltimpoPrecioLLenado: ', lastFillPrice)

    def openOrder(self, orderId, contract, order, orderState):
        print('Orden abierta: ', orderId, 'Contrato: ', contract.symbol, contract.secType, contract.currency, 'Estado de orden: ', orderState.status)

    def execDetails(self, reqId, contract, execution):
        print('ID: ', reqId, 'Contrato: ', contract.symbol, contract.secType, contract.currency, 'Ejecucion: ', execution.time)

    def empezar(self):
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CASH"
        contract.exchange = 'IDEALPRO'
        contract.currency = "USD"

        order = Order()
        order.action = 'BUY'
        order.orderType = "MKT"
        order.totalQuantity = 500

        self.placeOrder(self.nextOrderId , contract, order)

    def stop(self):
        self.done = True
        self.disconnect()

def main():
    app = TestApp()
    app.nextOrderId = 9
    app.connect('127.0.0.1', 7497, 0)

    # app.reqExecutions(1, )
    Timer(15, app.stop).start()
    app.run()

if __name__ == '__main__':
    main()