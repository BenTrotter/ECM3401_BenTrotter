# -----------------------------------------------------------------------------
# This file is the main file that is run to carry out the evoltuion using
# NSGA-II with srongly-typed genetic programming. The global variables are
# obtained from globals.py. These can be altered depending on the tests that
# are being run. The DEAP toolbox and genetic programming definitions are
# obtained from gp.py. Training, testing and validation runs are all 
# deployed to obtain the results. The results are saved in a pkl file. Graphs
# from the results are also plotted and saved in the current directory.
# The evolutionary process has been optimised using parallel processing from
# the Scoop library. This can be switched off if required. After all the 
# data windows have been run, the results are collected in a single directory
# named 'results'.
#
# Created for student individual project at the University of Exeter
# ECM3401
# Last edited: 10/05/2021
# -----------------------------------------------------------------------------

# External libraries
import random
import operator
import csv
import itertools
import os
import numpy
from deap import algorithms, base, creator, tools, gp
from deap.benchmarks.tools import diversity, convergence, hypervolume
from datetime import datetime, timedelta
import pandas as pd
import pickle
import urllib3
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from scoop import futures
import multiprocessing
from multiprocessing import freeze_support, Pool
from itertools import repeat
import shutil

# Other python files
from preload import load
from graph import *
from indicators import *
from gp import *
from globals import *


def training(s,e,parallel=True,save=True):
    """
    Carries out the training data window split. Performs the evolutionary process
    using NSGA-II and Genetic Programming. Further details of how this operation is 
    carried out is found in the project report.

    @param s the training window start date
    @param e the training window end date
    @param parallel a boolean, true if using parallel computing, false if not
    @param save a boolean, true if to save results, false if not.
    @return paretofront the individuals that are located on the Pareto front
    @return logbook the logbook recording the progress of the evolution
    """
    random.seed()
    trainBH = findBH(file,s,e)

    NGEN = ngen
    MU = mu
    CXPB = cxpb
    MUTPB = mutpb
    print("\n * ----------------- Evolution Info ----------------- *")
    if objectivesOption == 1:
        print("Using two objectives: Profit and PC")
    elif objectivesOption == 2:
        print("Using three objectives: Profit, PC and Risk Exposure")
    elif objectivesOption == 3:
        print("Using four objectives: Profit, PC, Risk Exposure and Number of trades")
    elif objectivesOption == 4:
        print("Using two objectives: PC and Risk Exposure")
    elif objectivesOption == 5:
        print("Using two objectives: Profit and Sharpe Ratio")
    elif objectivesOption == 6:
        print("Using three objectives: Profit, PC and Sharpe Ratio")
    elif objectivesOption == 7:
        print("Using two objectives: Profit and Risk Exposure")
    elif objectivesOption == 8:
        print("Using two objectives: Profit and Number of trades")
    elif objectivesOption == 9:
        print("Using three objectives: Profit, Risk Exposure and Number of trades")
    elif objectivesOption == 10:
        print("Using four objectives: Profit, Sharpe Ratio, Risk Exposure and Number of trades")
    elif objectivesOption == 11:
        print("Using four objectives: PC, Sharpe Ratio, Risk Exposure and Number of Trades")
    print("Retrieving data from ", file)
    print("Number of generations: ",NGEN)
    print("Population size: ",MU)
    print("CXPB: ",CXPB)
    print("MUTPB: ",MUTPB)
    print("PC k value is: ",k)
    print("Training PC split is: ",splitTrainingPeriod(datetime.strptime(s,'%Y-%m-%d'), datetime.strptime(e,'%Y-%m-%d'), k),"\n")
    print("Training B&H is ",trainBH,"%")
    print("Training on data from ",s," to ",e,"\n")

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "avg", "max"

    # Multiprocessing
    if parallel:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
        toolbox.register("map", futures.map)

    pop = toolbox.population(n=MU)

    paretofront = tools.ParetoFront()
    all = []
    hypers = {}

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Perform both mutation and crossover variations on all pop(probability dependant)
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)
        paretofront.update(pop)
        for ind in pop:
            if ind not in all:
                all.append(ind)

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

        if objectivesOption == 1:
            hypers[gen] = hypervolume(pop, [0.0, 0.0])
        elif objectivesOption == 2:
            hypers[gen] = hypervolume(pop, [0.0, 0.0, 500])
        elif objectivesOption == 3:
            hypers[gen] = hypervolume(pop, [0.0, 0.0, 500, 500])
        elif objectivesOption == 4:
            hypers[gen] = hypervolume(pop, [0.0, 500])
        elif objectivesOption == 5:
            hypers[gen] = hypervolume(pop, [0.0, 0.0])
        elif objectivesOption == 6:
            hypers[gen] = hypervolume(pop, [0.0, 0.0, 0.0])
        elif objectivesOption == 7:
            hypers[gen] = hypervolume(pop, [0.0, 500])
        elif objectivesOption == 8:
            hypers[gen] = hypervolume(pop, [0.0, 500])
        elif objectivesOption == 9:
            hypers[gen] = hypervolume(pop, [0.0, 500, 500])
        elif objectivesOption == 10:
            hypers[gen] = hypervolume(pop, [0.0, 0.0, 500, 500])
        elif objectivesOption == 11:
            hypers[gen] = hypervolume(pop, [0.0, 0.0, 500, 500])

        print(logbook.stream)

    if save:
        cp = dict(population=pop, generation=gen, pareto=paretofront,
            logbook=logbook, all=all, rndstate=random.getstate(),
            k=k,mu=MU,cxpb=CXPB,mutpb=MUTPB,hypers=hypers)

        with open("SavedOutput.pkl", "wb") as cp_file:
            pickle.dump(cp, cp_file)

    if parallel:
        pool.close()

    # ---------------- Graphing procdeures depending on objective option ------------- #
    if objectivesOption == 1:
        twoObjectivePareto(all, paretofront, objectivesOption, "Fitness of all individuals")
    elif objectivesOption == 2:
        allValues = []
        for i in all:
            allValues.append(i.fitness.values)
        scatter(all,paretofront,objectivesOption)
        threeObjectivePareto(allValues,dominates,objectivesOption)
    elif objectivesOption == 3:
        scatter(all,paretofront,objectivesOption)
    elif objectivesOption == 4:
        twoObjectivePareto(all, paretofront,objectivesOption, "Fitness of all individuals")
    elif objectivesOption == 5:
        twoObjectivePareto(all, paretofront, objectivesOption, "Fitness of all individuals")
    elif objectivesOption == 6:
        allValues = []
        for i in all:
            allValues.append(i.fitness.values)
        scatter(all,paretofront,objectivesOption)
        threeObjectivePareto(allValues,dominates,objectivesOption)
    elif objectivesOption == 7:
        twoObjectivePareto(all, paretofront, objectivesOption, "Fitness of all individuals")
    elif objectivesOption == 8:
        twoObjectivePareto(all, paretofront, objectivesOption, "Fitness of all individuals")
    elif objectivesOption == 9:
        allValues = []
        for i in all:
            allValues.append(i.fitness.values)
        scatter(all,paretofront,objectivesOption)
        threeObjectivePareto(allValues,dominates,objectivesOption)
    elif objectivesOption == 10:
        scatter(all,paretofront,objectivesOption)
    elif objectivesOption == 11:
        scatter(all,paretofront,objectivesOption)

    plotHypervolume(hypers)

    return paretofront, logbook


def findBH(name,tStart,tEnd):
    """
    Returns the percentage profit return using a buy and hold strategy.

    @param name the name of the csv file
    @param tStart the data window start date
    @param tEnd the data window end date
    @return bh the buy and hold return
    """
    # Gets a dict of date and price as keys and values.
    priceData = getPriceDataDict(name,tStart,tEnd)
    dataList = list(priceData.values())
    p1 = dataList[0]
    p2 = dataList[-1]

    bh = round(((p2-p1)/p1)*100,2)

    return bh

def unseenSimulation(i,unseenBH,data):
    """
    Takes an individual and simulates a trading window and returns the indivudals scores.
    This function can either be used for the testing or vvalidation data windows (defined
    in the project report). These windows have slightly different requirements so the
    procedure of this function changes depending on which data window has been chosen.
    This has been done in order to prevent repeated code for both testing and validation
    data windows.

    @param i the individual from the Pareto front which is being run on the data window
    @param unseenBH the B&H value for the data window.
    @param data a string either 'testing' or 'validation'. Informs which data window is
    being run.
    @return a list containing the info and fitness metrics of the individual obtained from
    the data window.
    """
    if data == "testing":
        if i.fitness.values[0] in scores:
            return
        scores.append(i.fitness.values[0])

    rule = toolbox.compile(expr=i)

    if data == "testing":
        startDate = unseenStart # Start date of the overall trading window.
        endDate = unseenEnd # End date of the overall trading window.
    elif data == "validation":
        startDate = validateStart # Start date of the overall trading window.
        endDate = validateEnd # End date of the overall trading window.

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
    pcSplit = unseenk# The number of intervals to split the trading window into for PC
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
    numTrDays = len(priceData)

    for date, price in priceData.items():
        # to start the sim from the start date
        if datetime.strptime(date,'%Y-%m-%d') < startDay and not findStart:
            continue
        # calculating the b&h strategy at start date and the first PC interval.
        elif not findStart:
            startD = date
            findStart = True
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
            break

        oldDate = date

        if oldP != False:
            if position:
                dailyReturn.append((((price-oldP)/oldP)*100))

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

    answer = ((balance - startingBalance)/startingBalance)*100

    if len(dailyReturn) == 0:
        sharpe = 0
    else:
        aveDailyReturn = sum(dailyReturn)/len(dailyReturn)
        numTDays = len(dailyReturn)
        stdDailyRateOfReturn = numpy.std(dailyReturn)
        if stdDailyRateOfReturn == 0:
            sharpe = 0
        else:
            sharpe = round((aveDailyReturn-(riskFreeRate/numTDays))/stdDailyRateOfReturn,2)

    if numTrades == 0:
        numTrades = 100
        riskExposure = 100

    above = round(((round(answer,2)-round(unseenBH,2))/round(unseenBH,2))*100,2)

    if data == "testing":
        return [str(i),pcCount,round(answer,2),i.fitness.values[0],sharpe,above]
    elif data == "validation":
        return [str(i),pcCount,round(answer,2),0,sharpe,above]


def unseen(paretofront, tStart, tEnd,test_K, fileName, data):
    """
    Performs the analysis of Pareto front on both testing and validation data windows.
    Uses parallel processing and prints the results to the terminal. The procedure 
    changes depending on whether the testing or validation window is being used in
    order to prevent repeated code.

    @param paretofront the individuals on the Pareto front
    @param tStart the window start date
    @param tEnd the window end date
    @param test_k the performance consistency k value
    @param filename the name of the csv file containg price data for the window.
    @param data a string either 'testing' or 'validation'. Informs which data window.
    @return pcDict the dictionary containing the performance of the individuals
    @return interval the PC intervl value in days.
    """
    BandH = True
    bh = findBH(fileName,tStart,tEnd)
    if data == "testing":
        print("Unseen B&H is ", bh,"%")
        print("Testing on unseen data from ",tStart," to ", tEnd,'\n')
    elif data == "validation":
        print("Validation B&H is ", bh,"%")
        print("Testing on validation data from ",tStart," to ", tEnd,'\n')

    pcDict = {}
    pcSplit = unseenk
    startDay = datetime.strptime(tStart,'%Y-%m-%d')
    endDay = datetime.strptime(tEnd,'%Y-%m-%d')
    interval = splitTrainingPeriod(startDay, endDay, pcSplit)

    if data == "testing":
        print("Number on pareto front is ",len(paretofront))
        print("Using ",multiprocessing.cpu_count()," processors. ")
    elif data == "validation":
        print("First result is best profit on pareto, second is best PC.\n")

    if data == "testing":
        with Pool() as p:
            mapped = p.starmap(unseenSimulation, zip(paretofront, repeat(bh), repeat('testing')))
    elif data == "validation":
        a = unseenSimulation(paretofront[0], bh, "validation")
        b = unseenSimulation(paretofront[1], bh, "validation")
        mapped = [a,b]

    for i in mapped:
        if i is not None:
            pcDict[i[0]] = i[1:]

    cp = dict(pareto=pcDict)
    with open("FinalOutput.pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file)

    return pcDict, interval


def processPareto(paretoDict, interval):
    """
    Loops throught a Pareto front and process/ranks each individual according to profit
    and PC (The selcting factors as defined in the report). Prints the results to the
    terminal.

    @param paretoDict the dictionary containing all the individuals on the Pareto front.
    @param interval the number of days for PC checks to occur.
    @return bestOnPareto a list containing the dictionary keys for the best individuals
    on the Pareto front.
    """
    print("Interval length for unseen PC is: ",interval,"\n")

    # Ordering the dictionary by pc value
    sorted_d = dict( sorted(paretoDict.items(), key=operator.itemgetter(1),reverse=True))

    beating = []
    maxProfit = -100
    maxPC = 0
    count = 0
    for key,v in sorted_d.items():
        if count == 0:
            pcKey = key
            count = 1
        print("Strategy:")
        print(key)
        print("Achieved a pc score of ",v[0],"/",unseenk, " on unseen data.")
        print("Training score: ",v[2])
        print("Unseen score: ",v[1]," -> This is an change of ",v[4],"% from the B&H.")
        print("Sharpe ratio: ",v[3],'\n')
        if v[1] > maxProfit:
            maxProfit = v[1]
            percent = v[4]
            pcMax = v[0]
            profKey = key

        if v[4] > 0:
            beating.append(v)

    print('\nMaximum profit is ', maxProfit," -> This is an change of ",percent,"% from the B&H.")
    print('PC value for this was ',pcMax,'/', unseenk)
    print('\nNumber that outperformed BH is ',len(beating),'/',len(sorted_d),'\n')

    bestOnPareto = [profKey, pcKey]

    return bestOnPareto


if __name__ == "__main__":

    # Load training data
    load(ticker,trainingStart,trainingEnd,resolution)
    freeze_support()
    random.seed()
    # Evolution on training data
    pareto, stats = training(trainingStart,trainingEnd)

    # Load testing data
    load(ticker,unseenStart,unseenEnd,resolution)
    # Run pareto front solutions on testing data
    answer, interval = unseen(pareto, unseenStart, unseenEnd, unseenk,file,"testing")
    # Retrieve testing results and process/print results
    bestPareto = processPareto(answer,interval)

    # Load Validation data
    load(ticker,validateStart,validateEnd,resolution)
    # Run pareto front solutions on validation data
    vAnswer, vInterval = unseen(bestPareto, validateStart, validateEnd, unseenk,file,"validation")
    bestPareto = processPareto(vAnswer,vInterval)

    # Remove old results folder
    try:
        cwd = os.getcwd()
        shutil.rmtree(cwd+"/results")
    except FileNotFoundError:
        print("No file name results to remove. Continuing...")
    os.mkdir("results")
    # Move final output files into a separate folder
    files = ['FinalOutput.pkl', 'SavedOutput.pkl', 'hv.png','map.png','3d.png','scatter.png']
    for f in files:
        try:
            shutil.move(f, 'results')
        except:
            continue
