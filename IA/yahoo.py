import pandas as pd
import yfinance as yf
import yahoofinancials
import os

DirActual = os.getcwd() # DirActual
os.chdir('DatosHistoricos') # Cambio de directorio
carpeta = 'AAPL3'
os.chdir(carpeta)

def datosYahoo(simbolo):
    tickerdata = yf.Ticker(simbolo)
    tickerinfo = tickerdata.info
    tickerDF = tickerdata.history(interval='5m', prepost=True)
    print(tickerDF)
    # tickerDF.to_csv('csvtest.csv')


datosYahoo('AAPL')

# yahoo_financials = yahoofinancials.YahooFinancials('AAPL')
#
# data = yahoo_financials.get_historical_price_data(start_date='2000-01-01',
#                                                   end_date='2019-12-31',
#                                                   time_interval='daily')
#
# tsla_df = pd.DataFrame(data['AAPL']['prices'])
# tsla_df = tsla_df.drop('date', axis=1).set_index('formatted_date')
# print(tsla_df.head())