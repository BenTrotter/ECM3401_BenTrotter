# -----------------------------------------------------------------------------
# File is used by evolution.py. It contains the functions used to obtain the
# values for the technical indicators from the custom csv file that is saved
# in the current directory. To find the details for the particular technical
# indicators please either view prelod.py, the project report or use 
# Investopedia's website to find definitons.
#
# Created for student individual project at the University of Exeter
# ECM3401
# Last edited: 10/05/2021
# -----------------------------------------------------------------------------

import pandas as pd
from globals import file

def ma(date,window):
    """
    Finds the value for the moving average on a particular date.

    @param date string representing the date to find the moving average
    @param window the window for the moving average. e.g. 7 day moving average
    @return a float representing the moving average
    """
    panda = pd.read_csv(file)
    mas = panda["SMA "+str(window)+".0"]
    dates = panda["Date"]
    dict1 = dict(zip(dates,mas))
    ma = dict1[date]
    return ma

def rsi(date,window):
    """
    Finds the RSI on a particular date.

    @param date string representing the date to find the RSI
    @param window the window for the RSI
    @return a float representing the RSI
    """
    panda = pd.read_csv(file)
    rsi = panda["RSI "+str(window)+".0"]
    dates = panda["Date"]
    dict1 = dict(zip(dates,rsi))
    RSI = dict1[date]
    return RSI

def ema(date,window):
    """
    Finds the EMA on a particular date.

    @param date string representing the date to find the EMA
    @param window the window for the EMA
    @return a float representing the EMA
    """
    panda = pd.read_csv(file)
    emas = panda["EMA "+str(window)+".0"]
    dates = panda["Date"]
    dict1 = dict(zip(dates,emas))
    ema = dict1[date]
    return ema

def macd(date,s,f,sig):
    """
    Finds the MACD for  particular date.

    @param date string representing the date to find the MACD
    @param s the slow average window
    @param f the fast average window
    @param sig the signal window
    @return a float representing the MACD
    """
    panda = pd.read_csv(file)
    macd = panda["MACD "+str(s)+str(f)]
    sig = panda["MACD Signal "+str(sig)+str(s)+str(f)]
    dates = panda["Date"]
    dict1 = dict(zip(dates,macd))
    dict2 = dict(zip(dates,sig))
    macd1 = dict1[date]
    sig1 = dict2[date]
    ans = macd1 - sig1
    return ans

def so(date,window):
    """
    Finds the SO for a particular date.

    @param date string representing the date to find the SO
    @param window the window for the SO
    @return a float representing the SO
    """
    panda = pd.read_csv(file)
    SO = panda["SO "+str(window)]
    sig = panda["%D 3-"+str(window)]
    dates = panda["Date"]
    dict1 = dict(zip(dates,SO))
    dict2 = dict(zip(dates,sig))
    so = dict1[date]
    sig1 = dict2[date]
    ans = so - sig1
    return ans

def if_then_else(input, output1, output2): 
    """
    A boolean conditon to be used by the genetic programming operation in
    evolution.py. It is used with DEAP library.

    @param input boolean to check
    @param output1 return output if input is true
    @param output2 return output if input is false.
    @return
    """
    return output1 if input else output2