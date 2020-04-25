from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *
import OrderSamples
import threading
import multiprocessing
import random
import concurrent.futures

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

    def BracketOrder(self, parentOrderId, action, quantity, limitPrice, takeProfitLimitPrice, stopLossPrice):
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
        takeProfit.transmit = False

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
    
    def empezar(self):
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CASH"
        contract.exchange = 'IDEALPRO'
        contract.currency = "USD"

        # order = Order()
        # order.action = 'BUY'
        # order.orderType = "MKT"
        # order.totalQuantity = 500
        # app.nextOrderId = 1

        bracket = self.BracketOrder(self.nextValidId, "BUY", 100, 1.1, 1.1, 1.05)
        for o in bracket:
            self.placeOrder(o.orderId, contract, o)
            self.nextValidId

    def stop(self):
        self.done = True
        self.disconnect()

def main():
    # global app
    app = TestApp()
    # cod = input('Codigo: ')
    # app.nextOrderId = c
    app.connect('127.0.0.1', 7497, 0)
    threading.Timer(3, app.stop).start()
    app.run()

if __name__ == '__main__':
    # with concurrent.futures.ProcessPoolExecutor() as ejecutor:
    #     l = list(range(10))
    #     ejecutor.map(main, l)
    main()

