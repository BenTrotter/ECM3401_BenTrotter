# -----------------------------------------------------------------------------
# File is used to generate the graphs used to analyse the performance of the
# evolutionary process (Genetic Programming with NSGA-II). The graphs are saved
# to the current directory which is where they can be accessed. Each graph is
# customised depending on the objective option that is used. The objective
# options have been given selected numbers to allow the graph functions to
# identify what graph to draw and with what titles.
#
# Created for student individual project at the University of Exeter
# ECM3401
# Last edited: 10/05/2021
# -----------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy


def twoObjectivePareto(pop,paretofront, option, title=""):
    """
    Plots the Pareto front using two objectives. The plot shows all scores for 
    all solutions aswell. Saves the plot to the current directory.

    @param pop the list containing all of the individuals
    @param paretofront list containing the individuals on pareto front
    @param option the objective option number used.
    @param title the name for the plot (optional)
    """
    x=[]
    y=[]
    for p in paretofront:
        fitness = p.fitness.values
        x.append(fitness[0])
        y.append(-fitness[1])
    xp=[]
    yp=[]
    for p in pop:
        fitness = p.fitness.values
        xp.append(fitness[0])
        yp.append(-fitness[1])
    fig,ax=plt.subplots(figsize=(5,5))
    ax.plot(xp,yp,".",label="All Solutions")
    ax.plot(x,y,linewidth=2, marker='o',markersize=4,label="Pareto Front")
    fitpareto=list(zip(x,y))
    fitpop=list(zip(xp,yp))

    # Changing the axis titles depending on the option number
    ax.set_title(title)
    if option == 1:
        plt.xlabel('% increase in profit')
        plt.ylabel('Performance consitency')
    if option == 4:
        plt.xlabel('Performance consistency')
        plt.ylabel('Risk Exposure')
    if option == 5:
        plt.xlabel('% increase in profit')
        plt.ylabel('Sharpe Ratio')
    if option == 7:
        plt.xlabel('% increase in profit')
        plt.ylabel('Risk Exposure')
    if option == 8:
        plt.xlabel('% increase in profit')
        plt.ylabel('Number of trades')
    plt.legend()
    # Saving the plot to the current directory
    plt.savefig('map.png')


def threeObjectivePareto(inputPoints, dominates, option):
    """
    Generates a three dimensional plot of all solutions using three objecitves
    with the Pareto front highlighted in red. Saves the plot to the current
    directory.

    @param inputPoints all solutions
    @param dominates the pareto points
    @param option the objective option used to help label the axis correctly.
    """
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if option == 2:
        ax.set_xlabel('% increase in profit', rotation=150)
        ax.set_ylabel('Performance consitency')
        ax.set_zlabel('Risk Exposure', rotation=60)
    elif option == 6:
        ax.set_xlabel('% increase in profit', rotation=150)
        ax.set_ylabel('Performance consitency')
        ax.set_zlabel('Sharpe Ratio', rotation=60)
    elif option == 9:
        ax.set_xlabel('% increase in profit', rotation=150)
        ax.set_ylabel('Risk Exposure')
        ax.set_zlabel('No. Trades', rotation=60)
    dp = numpy.array(list(dominatedPoints))
    pp = numpy.array(list(paretoPoints))
    print(pp.shape,dp.shape)
    ax.scatter(dp[:,0],dp[:,1],dp[:,2])
    ax.scatter(pp[:,0],pp[:,1],pp[:,2],color='red')

    import matplotlib.tri as mtri
    triang = mtri.Triangulation(pp[:,0],pp[:,1])
    ax.plot_trisurf(triang,pp[:,2],color='red')
    plt.savefig('3d.png')


def dominates(row, candidateRow):
    """
    Dominates function to be used in threeObjectivePareto. Used to see if one point
    dominates another. Returns a boolean true if dominates, false if not.

    @param row the current row
    @param candidateRow the test row
    @return a boolean, true if dominates, false if not.
    """
    return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row)


def plotHypervolume(hypers, title="Hypervolume Plot"):
    """
    Plots the hypervolume against the generation number using Matplotlib. Saves
    the plot to the current directory.

    @param hypers a dictionary containing the generation number as the key and
    the hypervolume as the value.
    @param title the title for the plot (optional)
    """
    x=[]
    y=[]
    for gen,hv in hypers.items():
        x.append(gen)
        y.append(hv)
    plt.rcParams.update({'font.size': 22})
    fig,ax=plt.subplots(figsize=(13,13))
    fitpareto=list(zip(x,y))
    ax.set_title(title)
    plt.plot(x, y, color='red', marker='o')
    plt.xlabel('Generation Number')
    plt.ylabel('Hypervolume')
    plt.savefig('hv.png')


def scatter(all,pareto,option):
    """
    Generates a three or four objective scatter graphs comparing each objective
    pairing idividually on one plot. Saves the plot to the current directory.

    @param all a list of all solutions tested during the evolution.
    @param pareto the pareto front solutions.
    @param option the objective option used to help lable the plot.
    """
    allValues = []
    for i in all:
        allValues.append(i.fitness.values)

    allPareto = []
    for x in pareto:
        allPareto.append(x.fitness.values)

    d = {}
    target = []
    for ind in allValues:
        if ind in allPareto:
            target.append(1)
        else:
            target.append(0)
    
    d['data'] = allValues
    d['target'] = target

    if option == 2:
        d['target_names'] = ['Profit','PC','Risk Exposure']
    elif option == 3:
        d['target_names'] = ['Profit','PC','Risk Exposure','No. Trades']
    elif option == 6:
        d['target_names'] = ['Profit','PC','Sharpe Ratio']
    elif option == 9:
        d['target_names'] = ['Profit','Risk Exposure','No. Trades']
    elif option == 10:
        d['target_names'] = ['Profit','Sharpe Ratio','Risk Exposure','No. Trades']
    elif option == 11:
        d['target_names'] = ['PC','Sharpe Ratio','Risk Exposure','No. Trades']

    X = d['data']
    y = d['target']
    df = pd.DataFrame(X, columns = d['target_names'])

    pd.plotting.scatter_matrix(df, c=y, figsize = [8,8],
                        s=80, marker = 'D');
    df['y'] = y
    sns.set(style="ticks", color_codes=True)

    sns.pairplot(df,hue='y')

    plt.rcParams["figure.subplot.right"] = 0.8

    handles = [plt.plot([],[], ls="", marker=".", \
                    markersize=numpy.sqrt(200))[0] for i in range(2)]
    labels=["All solutions", "Pareto Front"]
    plt.legend(handles, labels)
 
    plt.savefig(r"scatter.png")