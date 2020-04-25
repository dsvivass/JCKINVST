from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class TestApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId, errorCode, errorString):
        print('Error: ', reqId, ' ', errorCode, ' ', errorString)

    def tickPrice(self, reqId , tickType, price, attrib):
        print('ID: ', reqId, ' ', 'Tipo de Tick: ', tickType, ' ', 'Precio: ', price, ' ', 'Atributo: ', attrib)

def main():
    app = TestApp()
    app.connect('127.0.0.1', 7497, 0)

    contract = Contract()
    contract.symbol = "EUR"
    contract.secType = "CASH"
    contract.exchange = 'IDEALPRO'
    contract.currency = "USD"

    app.reqMktData(1, contract, "233,236", False, False, [])
    app.run()

if __name__ == '__main__':
    main()
