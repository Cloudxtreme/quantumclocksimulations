from __future__ import print_function
from sys import stdout
import pandas as pd
import numpy as np
import os
from qutip import *
import matplotlib.pyplot as plt
from copy import deepcopy as dc

class SimulationsController:
    """ SimulationsController class
        ___________________________

        Is used to perform many consecutive simulations (usually with different dimensions)
        and to analyze the results.


        Functions
        _________
       
        clear() : clears the simulation controller (all previously added simulations are deleted)

        info(sim) : returns information about the SimulationsController, this information is usually
                    manually added to it by instantiating it with a description, sim can be specified
                    to get additional information about a specific simulation

        readResults(filename) : loads the simulation results from a .csv file

        plotAveragesVsDimensions(errorbar, stdOfMean, xmin, xmax, ymin, ymax, title) : Plots the standard
                    plot of the number of alternate ticks vs. the dimension of the system. All arguments
                    are optional. Errorbar = True shows the standard deviation as error bars, stdOfMean = True
                    changes this to the standard deviation of the Mean.
                    title specifies the title of the plot.

        plotHistogram(sim, title) : Plots a histogram of alternate ticks for a specific simulation.

        add(simulation) : adds a simulation to the SimulationController

        remove(index) : removes simulation index

        performSimulations() : performs all simulations that have been added to it

        getNSimulations() : returns the number of simulations added to the controller

"""

    # private functions
    # ________________

    def __init__(self, filePath = None, description = None):
        # private attributes
        # __________________
        self.__simulations = []
        self.__nSimulations = 0
        self.__filePath = filePath
        # if no filepath is given, set it to results + current time
        # on the desktop
        if self.__filePath is None:
            self.__filePath = os.path.join(os.path.expanduser('~'), 'simulation_results')
            if not os.path.exists(self.__filePath):
                os.makedirs(self.__filePath)
        else:
            if not os.path.exists(filePath):
                print ('Created path ' + filePath + '.')
                os.makedirs(filePath)
                self.__filePath = filePath
        from datetime import datetime
        now = datetime.now()
        nowString = "%d_%d_%d_%d_%d_%d" % (now.day, now.month, now.year,
                now.hour, now.minute, now.second)
        filename = 'results_' + nowString + '.csv'

        self.__filename = filename

        self.__description = description
        if description is None:
            self.__description = 'Controller created %d.%d.%d - %d:%d:%d.' % (now.day, now.month, now.year,
                    now.hour, now.minute, now.second)

        self.__results = None

    def __save(self, sim, alternateTicks):
        pathToFile = os.path.join(self.__filePath, self.__filename)
        avgAlternateTicks = np.mean(alternateTicks)
        stdAlternateTicks = np.std(alternateTicks)
        if not os.path.isfile(pathToFile):
            data = pd.DataFrame(columns = [
                'n_average',
                'dimension',
                'summary',
                'list_alternate_ticks',
                'average_alternate_ticks'
                'std_alternate_ticks',
                'label',
                'description'
                ])
        else:
            data = pd.read_csv(pathToFile)

        ind = len(data)

        # encode the states and hamiltonians as a string
        summary = sim.summary()

        appendix = 1
        while sim.label in data['label']:
            newLabel = sim.label + str(appendix)
            print ('Simulation with label ' + sim.label + ' already exists. Changed label to ' + \
                    newlabel + '.')
            sim.label = newLabel

        newData = pd.DataFrame({
            'n_average':[sim.nAverage],
            'dimension':[sim.getDimension()],
            'summary':[summary],
            'list_alternate_ticks':[alternateTicks],
            'average_alternate_ticks':[avgAlternateTicks],
            'std_alternate_ticks':[stdAlternateTicks],
            'label':[sim.label],
            'description':[self.__description]
            }, index = [ind])
        data = pd.concat([data, newData])
        data.to_csv(pathToFile, index = False, index_label = False)
        print('Saved simulation to ' + pathToFile  + '.', end = '\n')
        stdout.flush()

    # public functions
    # _______________

    def clear(self, filePath = None, description = None):
        self.__init__(filePath = self.__filePath, description = description)

    def info(self, sim = None):
        if sim is None:
            return (self.__description)
        simSummary = '' 
        if type(sim) is int:
            try:
                simSummary = self.__results['summary'][sim-1]
            except:
                print ('Simulation number does not exist.')
                print
                return
        elif type(sim) is str:
            simSummary = self.__resulte['summary'][self.__results['label'] == simLabel]
        else:
            print ('Unsupported type for simulation. Please provide the number or the label of the simulation.')
            print
            return
        return self._description + '\n\n' + simSummary

    def readResults(self, filename = None):
        if filename is None:
            # if no filename is specified, return the newest simulation
            # (1) get all files
            listOfFiles = [f for f in os.listdir(self.__filePath) if os.path.isfile(os.path.join(self.__filePath,f))]
            # (2) filter for csv files
            listOfFiles = [f for f in listOfFiles if f.endswith('.csv')]
            if len(listOfFiles) == 0:
                print ('Can\'t open latest simulation results. No result files in specified file path.')
                return
            # (3) get the newest one
            listOfFiles.sort(key = lambda x: os.path.getmtime(
                os.path.join(self.__filePath, x)))
            filename = os.path.join(self.__filePath,listOfFiles[-1])
        else:
            if type(filename) is not str:
                print ('Can\'t read simulation results. Filename must be string.')
                return
            if not os.path.isfile(filename):
                print ('Can\'t read simulation results. File does not exist.')
                return
            else:
                if not filename.endswith('.csv'):
                    print ('Cant\'t read simulation results. Specified file name has to be of type .csv')
                    return
        self.__results = pd.read_csv(filename)
        # some of the older simulations don't have  description yet, so there needs to be a workaround
        # for loading the description
        try:
            self.__description = self.__results['description'][0]
        except:
            return

    def plotAveragesVsDimensions(self, errorbar = False, stdOfMean = False, xmin = None, xmax = None, ymin = None, ymax = None, title = None):
        if self.__results is None:
            print ('No results loaded yet. Read the latest simulation results with SimulationsController.readResults().')
            return
        plt.figure()
        autoTitle = 'Alternate Ticks'
        nAverage = self.__results['n_average'][0]
        if not all(navg == self.__results['n_average'][0] for navg in self.__results['n_average']):
            print ('Warning: Alternate ticks have been averaged over different number of iterations ' + \
                    'for different dimensions...')
        else:
            autoTitle += ' averaged over %d iterations.' % nAverage 
        if errorbar:
            if stdOfMean:
                plt.errorbar(self.__results['dimension'], self.__results['average_alternate_ticks'],
                    yerr = self.__results['std_alternate_ticks']/np.sqrt(nAverage), ls = '-', c = 'b')
                autoTitle += ' (error bar from std of mean)'
            else:
                plt.errorbar(self.__results['dimension'], self.__results['average_alternate_ticks'],
                    yerr = self.__results['std_alternate_ticks'], ls = '-', c = 'b')
                autoTitle += ' (error bar from std)'
        else:
            plt.plot(self.__results['dimension'], self.__results['average_alternate_ticks'],
                    ls = '-', c = 'b')
        plt.xlabel('Dimension')
        plt.ylabel('# Alternate Ticks')
        if xmin is not None:
            plt.xlim(xmin = xmin)
        if xmax is not None:
            plt.xlim(xmax = xmax)
        if ymin is not None:
            plt.ylim(ymin = ymin)
        if ymax is not None:
            plt.ylim(ymax = ymax)
        if title is None:
            plt.title(autoTitle)
        else:
            plt.title(title)
        plt.show()

    def plotHistogram(self, sim, title = None):
        if self.__results is None:
            print ('No results loaded yet. Read the latest simulation results with SimulationController.readResults().')
            return
        # read the list of alternate ticks
        alternateTicksAsString = None
        autoTitle = 'Histogram of # alternate ticks for Simulation '
        if type(sim) is int:
            try:
                alternateTicksAsString = self.__results['list_alternate_ticks'][sim-1]
                autoTitle += str(sim) + ' with dimension ' + str(self.__results['dimension'][sim-1]) + '.'
            except:
                print ('Simulation number does not exist.')
                return
        elif type(sim) is str:
            alternateTicksAsString = self.__results['list_alternate_ticks'][self.__results['label'] == sim]
            autoTitle += simLabel + ' with dimension ' + str(self.__results['dimension'][self.__results['label'] == sim]) + '.'
        else:
            print ('Unsupported type for simulation. Please provide the number or the label of the simulation.')
            return
        alternateTicks = map(int, alternateTicksAsString[1:-1].split(','))
        histo, bins = np.histogram(alternateTicks, bins = range(max(alternateTicks) + 2))
        plt.bar(bins[:-1], histo, width = 1, color = 'b')
        plt.xlabel('# Alternate Ticks')
        plt.ylabel('Appearances')
        if title is None:
            plt.title(autoTitle)
        else:
            plt.title(title)
        plt.show()

        plt.figure()

    def add(self, simulation):
        self.__simulations.append(simulation)
        self.__nSimulations += 1

    def remove(self, index = -1):
        if self.__nSimulations == 0:
            print ('No simulations added. Nothing to remove.')
            return
        self.__simulations.pop(index % self.__nSimulations)
        self.__nSimulation -= 1

    def performSimulations(self):
        from sys import stdout

        for index in range(self.__nSimulations):
            sim = self.__simulations[index]
            # check if it is a valid simulation

            sim.initialize()
            if not sim.isReady():
                print('Simulation %d/%d is not valid. Skipping...' % (index+1, self.__nSimulations))
                stdout.flush()
                continue

            alternateTicks = []
            currentAverage = 0.
            for lap in range(sim.nAverage):
                # bool variable to indicate if a simulation needs to be stopped
                stopSimulation = False
                # tickArray will look like 0101010... and indicates which clocks have ticked
                tickArray = [] 
                tickString = ''
                while True:
                    tickingClocks = sim.run()
                    # if several clocks have ticked at exactly the same time (very unlikely)
                    # the simulation should stop (not alternate)
                    if len(tickingClocks) > 1 or len(tickingClocks) == 0:
                        stopSimulation = True
                    else:
                        # now, we know that only one clock has ticked...
                        tickingClock = tickingClocks[0]
                        # check if it is in alternate order...
                        # note that the code allows for two modes:
                        # 1) strictly alternate: clocks have to tick in a predefined order
                        # 2) less strictly alternate: all clocks need to tick before the first one ticks a second time
                        if sim.mode == 'normal':
                            relevant = len(tickArray) % sim.getNClocks()
                            if relevant == 0:
                                tickArray.append(tickingClock)
                                tickString += str(tickingClock)
                            else:
                                if not tickingClock in tickArray[-relevant:]:
                                    tickArray.append(tickingClock)
                                    tickString += str(tickingClock)
                                else:
                                    stopSimulation = True
                        elif sim.mode == 'strict' and sim.order is None:
                            if not tickingClock in tickArray:
                                tickArray.append(tickingClock)
                                tickString += str(tickingClock)
                            else:
                                stopSimulation = True
                            if len(tickArray) == sim.getNClocks():
                                sim.order = dc(tickArray) 
                        else: # this means mode == 'strict'
                            if len(tickArray) == 0:
                                requiredClock = sim.order[0]
                            else:
                                requiredClock = sim.order[(sim.order.index(tickArray[-1]) + 1) % sim.getNClocks()]
                            if requiredClock == tickingClock:
                                tickArray.append(tickingClock)
                                tickString += str(tickingClock)
                            else:
                                stopSimulation = True
                    nAlternateTicks = len(tickArray)
                    tempCurrentAverage = currentAverage * lap / float(lap + 1) + nAlternateTicks / float(lap + 1)

                    # create console output
                    simOutput = 'Sim ' + ' '*(6-(len(str(index+1)))) + str(index+1) + ' / ' + str(self.__nSimulations) + ' '*(6-len(str(self.__nSimulations))) + '  ||  '
                    lapOutput = 'Lap ' + ' '*(6-(len(str(lap+1)))) + str(lap+1) + ' / ' + str(sim.nAverage) + ' '*(6-len(str(sim.nAverage))) + '  ||  '
                    currentExperimentOutput = '['

                    maxStringLength = 117
                    appendix = ''
                    if stopSimulation:
                        maxStringLength -= 9 
                        appendix = '-Stopped.'
                    if nAlternateTicks > maxStringLength:
                        currentExperimentOutput += '...' + tickString[-maxStringLength:] + appendix
                    else:
                        currentExperimentOutput += tickString + appendix
                    currentExperimentOutput += '-' * (maxStringLength + 4 - len(currentExperimentOutput)) + ']' + '  ||  '
                    avgOutput = 'Avg: ' + str(round(tempCurrentAverage,3))

                    end = '\r'
                    if stopSimulation and lap == sim.nAverage-1:
                        end = '\n'

                    print(simOutput + lapOutput + currentExperimentOutput + avgOutput, end = end)
                    stdout.flush()

                    # check if simulation needs to be stopped
                    if stopSimulation:
                        currentAverage = tempCurrentAverage
                        alternateTicks.append(nAlternateTicks)
                        sim.initialize()
                        break

            self.__save(sim, alternateTicks)

    def getNSimulations(self):
        return self.__nSimulations
