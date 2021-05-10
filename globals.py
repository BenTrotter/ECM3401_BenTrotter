# -----------------------------------------------------------------------------
# This file contains the global variables which are used to carry out tests.
# These should be changed according to the experiment/test that is being
# undertaken. It ensures that the variables remain constant throughout all the
# files.
#
# Created for student individual project at the University of Exeter
# ECM3401
# Last edited: 10/05/2021
# -----------------------------------------------------------------------------

global file
file = "MSFT.csv"

global ticker
ticker = "MSFT"

global resolution
resolution = '1d'

# ---------- Objective Options ---------- #
# 1 : Profit and PC
# 2 : Profit, PC and Risk Exposure
# 3 : Profit, PC, Risk Exposure and number of trades
# 4 : PC and Risk Exposure
# 5 : Profit and Sharpe Ratio
# 6 : Profit, PC and Sharpe Ratio
# 7 : Profit and Risk Exposure
# 8 : Profit and no. trades
# 9 : Profit, Risk Exposure and no. trades
# 10: Profit, Sharpe ratio, Risk Exposure and no.trades
# 11: PC, Sharpe ratio, Risk Exposure and no. trades
global objectivesOption
objectivesOption = 1

global trainingStart
global trainingEnd
global unseenStart
global unseenEnd
global validateStart
global validateEnd

# Window start and end dates for each data window split.
trainingStart = "2018-01-01"
trainingEnd = "2019-01-01"
unseenStart = "2019-01-01"
unseenEnd = "2020-01-01"
validateStart = "2020-01-01"
validateEnd = "2021-01-01"

# This is the parameter for the performance consistency
global k
k = 24
# The performance consistency parameter on the unseen window
global unseenk
unseenk = 24
# The risk free rate used when calcualting the Sharpe ratio
global riskFreeRate
riskFreeRate = 0.05

global scores
scores = []

# Evolution parameters
global ngen
ngen = 5
global mu
mu = 8
global cxpb
cxpb = 0.4
global mutpb
mutpb = 0.5

# Technicl indicator parameter sets
global maWindows
maWindows = [5, 10, 20, 30, 50, 100, 200]
global emaWindows
emaWindows = [5, 12, 26, 30, 50, 100, 200]
global rsiWindows 
rsiWindows = [7, 14, 28]
global macdS
macdS = [26, 35]
global macdF
macdF = [5, 12]
global macdSig
macdSig = [5, 9]
global soWindows
soWindows = [7,14,28]
