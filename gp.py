# -----------------------------------------------------------------------------
# File is used to define the genetic programming procedure that is used by
# DEAP library. It also defines the evolutionary process that the DEAP library 
# will use: NSGA-II. It defines classes to define the type restrictions as the form
# of genetic programming used is strongly-typed. The operator nodes (primitives)
# and terminal nodes are defined. These define the scope in which trees can
# be evolved. This genetic programming structure is used in evolution.py during
# the evolutionary process. The fitness function is then defined. 'simulation'
# is the fitness function. It hs some helper functions to aid its opertaion.
# The DEAP toolbox parameters are then defined to inform the rules and methods
# for the evolutionary process that is used in evolution.py.
#
# Created for student individual project at the University of Exeter
# ECM3401
# Last edited: 10/05/2021
# -----------------------------------------------------------------------------

# External libaries
import operator
import numpy
from datetime import datetime, timedelta
from deap import algorithms, base, creator, tools, gp

# Other python files
from indicators import *
from globals import *

# Below are the classes to enforce types in strongly-typed genetic programming
class pd_float(object):
    pass

class pd_bool(object):
    pass
class position_bool(object):
    pass

class pd_int(object):
    pass

class ema_int(object):
    pass

class ma_int(object):
    pass

class rsi_int(object):
    pass
class rsi_bounds(object):
    pass
class rsi_ans(object):
    pass
class rsi_lt(object):
    pass
class rsi_gt(object):
    pass

class macd_sig(object):
    pass
class macd_f(object):
    pass
class macd_s(object):
    pass
class macd_ans(object):
    pass
class macd_bounds(object):
    pass

class so_int(object):
    pass
class so_ans(object):
    pass
class so_bounds(object):
    pass


# ---------------------- Genetic Programming Representation ----------------- #

# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", [str,position_bool], pd_bool)
# boolean operators
pset.addPrimitive(if_then_else, [position_bool, rsi_gt, rsi_lt], pd_bool)
pset.addPrimitive(operator.and_, [pd_bool, pd_bool], pd_bool)
pset.addPrimitive(operator.or_, [pd_bool, pd_bool], pd_bool)
pset.addPrimitive(operator.not_, [pd_bool], pd_bool)
pset.addPrimitive(operator.lt, [pd_float, pd_float], pd_bool)
pset.addPrimitive(operator.gt, [pd_float, pd_float], pd_bool)
pset.addPrimitive(operator.lt, [rsi_ans,rsi_bounds], rsi_lt)
pset.addPrimitive(operator.gt, [rsi_ans, rsi_bounds], rsi_gt)
pset.addPrimitive(operator.lt, [macd_bounds, macd_ans], pd_bool)
pset.addPrimitive(operator.gt, [macd_bounds, macd_ans], pd_bool)
pset.addPrimitive(operator.lt, [so_bounds, so_ans], pd_bool)
pset.addPrimitive(operator.gt, [so_bounds, so_ans], pd_bool)
pset.addPrimitive(ma, [str, ma_int], pd_float)
pset.addPrimitive(ema, [str, ema_int], pd_float)
pset.addPrimitive(rsi, [str, rsi_int], rsi_ans)
pset.addPrimitive(macd, [str, macd_s, macd_f, macd_sig], macd_ans)
pset.addPrimitive(so, [str, so_int], so_ans)

for i in emaWindows:
    pset.addTerminal(i,ema_int)

for i in maWindows:
    pset.addTerminal(i,ma_int)

for i in rsiWindows:
    pset.addTerminal(i,rsi_int)
for i in range(0,100):
    pset.addTerminal(i,rsi_bounds)

for i in macdS:
    pset.addTerminal(i,macd_s)
for i in macdF:
    pset.addTerminal(i,macd_f)
for i in macdSig:
    pset.addTerminal(i,macd_sig)
pset.addTerminal(0,macd_bounds)

for i in soWindows:
    pset.addTerminal(i,so_int)
pset.addTerminal(0,so_bounds)

pset.renameArguments(ARG0='Date')
pset.renameArguments(ARG1='Position')


# --------------------------------------------------------------------------- #


def splitTrainingPeriod(startD, endD, k):
    """
    simulation helper function. Splits the overall trading window by an integer
    k. The function then returns the interval of days that the window has been
    split into. This function is used to calculate the performance consistency.

    @param startD start date for the data window
    @param endD end date for the data window
    @param k Performance consistency parameter.
    @return the integer representing the number of days between PC checks.
    """
    days = endD - startD
    interval = days/k
    return interval

def getPCUpdateDates(startDay,endDay,interval):
    """
    simulation helper function. Returns a list containing all of the dates that
    the PC count must make a check and update its value. Used in the simulation
    function so that it knows on which date to take a PC measurement.

    @param startDay start date for the data window
    @param endDay end date for the data window
    @param interval the number of days after which PC is checked.
    @return a list containing all of the dates that the PC count must make a
    check and update its value.  
    """
    performanceConsistencyDates = []
    checkDay = startDay

    while(checkDay < endDay):
        checkDay += interval
        performanceConsistencyDates.append(checkDay)

    return performanceConsistencyDates

def getPriceDataDict(csv,startDate,endDate):
    """
    simulation helper function. Returns a dict of date and price as keys and
    values. It trims the dict to only contain the relevant data between certain
    dates. Obtained from the custom csv file that has been created from the 
    load function.

    @param csv The csv file in which to obtain the price data.
    @param startDay start date for the data window
    @param endDay end date for the data window
    @return a dictionary containing date and closing price as keys and values.
    """
    panda = pd.read_csv(csv)
    panda["Date"] = pd.to_datetime(panda['Date']) # date column to datetime
    # retrieve dates from series between start and end date
    mask = (panda['Date'] > startDate) & (panda['Date'] <= endDate)
    panda = panda.loc[mask]
    prices = panda["Close"]
    panda = panda.astype(str) # makes everything in the panda to a string
    dates = panda['Date']
    combined = dict(zip(dates, round(prices,2)))

    return combined

def simulation(individual):
    """
    This is the fitness function. It is used to be run for each individual
    in the population and to return a number of fitness values, depending on
    the objective option that is being used. This function is added to the
    DEAP toobox as the evalutaion function. The fitness values are obtained
    after running the trading strategy through a data window. The peformance
    metrics are then recorded and returned after the trading simulation has
    finished.

    @param individual a Genetic programming representation of an individual
    from the population. The individual encodes a trading strategy.
    @return a series of values depending on the obbjective 
    """
    rule = toolbox.compile(expr=individual)

    startDate = trainingStart # Start date of the overall trading window.
    endDate = trainingEnd # End date of the overall trading window.
    shares = 0 # The number of shares that the trader currently owns.
    position = False # Whether the trader is currently invested in the stock.
    startingBalance = 1000 # The starting amount that the trader will trade with.
    balance = startingBalance # The current balance or equity of the trader.
    numTrades = 0 # The number of trades that have been executed.
    iter = 0
    findStart = False # Bool to help find the first tradable start date.
    # Converting date strings to  datetime objects
    startDay = datetime.strptime(startDate,'%Y-%m-%d')
    endDay = datetime.strptime(endDate,'%Y-%m-%d')
    pcSplit = k # The number of intervals to split the trading window into for PC
    # Get the interval in days for PC
    interval = splitTrainingPeriod(startDay, endDay, pcSplit)
    # Get the dates that the PC count needs to be checked and updated.
    performanceConsistencyDates = getPCUpdateDates(startDay, endDay, interval)
    # Gets a dict of date and price as keys and values.
    priceData = getPriceDataDict(file,startDate,endDate)
    riskExposure = 0 # The num days that a trader is in the market.
    pcCount = 0 # The count for the performancy consistency value.
    pcIter = 0 # The index for the performanceConsistencyDatess
    answer = 1 # Initialising answer to prevent zero division error.

    dailyReturn = []
    oldP = False
    numTDays = len(priceData)

    for date, price in priceData.items():
        # Starts the sim from the start date
        if datetime.strptime(date,'%Y-%m-%d') < startDay and not findStart:
            continue
        # calculating the b&h strategy at start date and the first PC interval.
        elif not findStart:
            startD = date
            findStart = True
            bhShares = startingBalance / priceData[date]
            oldPrice = price
            oldBalance = startingBalance

        if iter == 0:
            oldDate = date
            iter += 1
            continue

        # PC update if the correct date has been reached
        if datetime.strptime(date, '%Y-%m-%d') >= performanceConsistencyDates[pcIter]:
            percentIncPriceTest = ((price-oldPrice)/oldPrice)*100
            percentIncBalanceStrategy =  ((balance-oldBalance)/oldBalance)*100
            if percentIncPriceTest <= percentIncBalanceStrategy:
                pcCount += 1
            oldPrice = price
            oldBalance = balance
            pcIter+=1

        action = rule(date,position)

        if action and position == False:
            buy = True
            sell = False
        elif not action and position == False:
            sell = False
            buy = False
        elif action and position == True:
            sell = False
            buy = False
        elif not action and position == True:
            sell = True
            buy = False

        if buy:
            numTrades += 1
            position = True
            shares = balance/price
            oldAmount = balance
            balance = round(price*shares,2)

        elif sell and shares > 0:
            position = False
            balance = round(shares*price, 2)
            profit = balance - oldAmount
            shares = 0

        elif shares != 0:
            balance = round(price*shares,2)

        if position == True:
            riskExposure += 1

        # Ends the simulation at the end date
        if datetime.strptime(date,'%Y-%m-%d') >= endDay:
            bhBalance = bhShares*priceData[oldDate]
            break

        oldDate = date

        if oldP != False:
            if position:
                dailyReturn.append(((price-oldP)/oldP)*100)

        oldP = price

    # Final peformance consitency check on final iterval
    if(pcIter != pcSplit):
        percentIncPriceTest = ((price-oldPrice)/oldPrice)*100
        percentIncBalanceStrategy =  ((balance-oldBalance)/oldBalance)*100
        if percentIncPriceTest <= percentIncBalanceStrategy:
            pcCount += 1
        pcIter+=1

    # Check to ensure every interval as been included in the PC count
    if(pcIter != pcSplit):
        print("\nASSERT ERROR: Not all pc intervals have been calculated.\n")

    # Percentage return
    answer = ((balance - startingBalance)/startingBalance)*100

    # Sharpe Ratio
    if len(dailyReturn) == 0:
        sharpeRatio = 0
    else:
        aveDailyReturn = sum(dailyReturn)/len(dailyReturn)
        stdDailyRateOfReturn = numpy.std(dailyReturn)
        if stdDailyRateOfReturn == 0:
            sharpeRatio = 0
        else:
            sharpeRatio = round((aveDailyReturn-(riskFreeRate/numTDays))/stdDailyRateOfReturn,2)

    if numTrades == 0:
        numTrades = 100
        riskExposure = 100

    if objectivesOption == 1:
        return round(answer,2), pcCount,
    elif objectivesOption == 2:
        return round(answer,2), pcCount, riskExposure,
    elif objectivesOption == 3:
        return round(answer,2), pcCount, riskExposure, numTrades,
    elif objectivesOption == 4:
        return pcCount, riskExposure,
    elif objectivesOption == 5:
        return round(answer,2), sharpeRatio,
    elif objectivesOption == 6:
        return round(answer,2), pcCount, sharpeRatio,
    elif objectivesOption == 7:
        return round(answer,2), riskExposure,
    elif objectivesOption == 8:
        return round(answer,2), numTrades,
    elif objectivesOption == 9:
        return round(answer,2), riskExposure, numTrades,
    elif objectivesOption == 10:
        return round(answer,2), sharpeRatio, riskExposure, numTrades,
    elif objectivesOption == 11:
        return pcCount, sharpeRatio,riskExposure, numTrades,

# Setting toolbox weights depending on objective option
if objectivesOption == 1:
    creator.create("Fitness", base.Fitness, weights=(1.0,1.0))
elif objectivesOption == 2:
    creator.create("Fitness", base.Fitness, weights=(1.0,1.0,-1.0))
elif objectivesOption == 3:
    creator.create("Fitness", base.Fitness, weights=(1.0,1.0,-1.0,-1.0))
elif objectivesOption == 4:
    creator.create("Fitness", base.Fitness, weights=(1.0,-1.0))
elif objectivesOption == 5:
    creator.create("Fitness", base.Fitness, weights=(1.0,1.0))
elif objectivesOption == 6:
    creator.create("Fitness", base.Fitness, weights=(1.0,1.0,1.0))
elif objectivesOption == 7:
    creator.create("Fitness", base.Fitness, weights=(1.0,-1.0))
elif objectivesOption == 8:
    creator.create("Fitness", base.Fitness, weights=(1.0,-1.0))
elif objectivesOption == 9:
    creator.create("Fitness", base.Fitness, weights=(1.0,-1.0,-1.0))
elif objectivesOption == 10:
    creator.create("Fitness", base.Fitness, weights=(1.0,1.0,-1.0,-1.0))
elif objectivesOption == 11:
    creator.create("Fitness", base.Fitness, weights=(1.0,1.0,-1.0,-1.0))

# Registering and initialising the DEAP toolbox
creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

terminal_types = [so_bounds,so_int,macd_bounds,macd_sig,macd_f,macd_s, ema_int ,ma_int, rsi_int, rsi_bounds, str, position_bool]
toolbox = base.Toolbox()

toolbox.register("expr", gp.generate_safe, pset=pset, min_=1, max_=5, terminal_types=terminal_types)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", simulation)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.generate_safe, min_=1, max_=6, terminal_types=terminal_types)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)