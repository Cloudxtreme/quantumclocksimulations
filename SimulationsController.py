from __future__ import print_function
from sys import stdout
import pandas as pd
import numpy as np
import os
from qutip import *
import matplotlib.pyplot as plt

class SimulationsController:
    """ Highest level class of the simulation. Objects of this
    class control the simulations, i.e. it starts and terminates them
    and saves the results. It also controls the terminal output during
    the simulation. """
    def __init__(self, filePath = None, description = None):
        self.simulations = []
        self.nSimulations = 0
        self.filePath = filePath

        # if no filepath is given, set it to results + current time
        # on the desktop
        if self.filePath is None:
            self.filePath = os.path.join(os.path.expanduser('~'), 'simulation_results')
            if not os.path.exists(self.filePath):
                os.makedirs(self.filePath)
        else:
            if not os.path.exists(filePath):
                print ('Created path ' + filePath + '.')
                os.makedirs(filePath)
                self.filePath = filePath

        from datetime import datetime
        now = datetime.now()
        nowString = "%d_%d_%d_%d_%d_%d" % (now.day, now.month, now.year,
                now.hour, now.minute, now.second)
        filename = 'results_' + nowString + '.csv'

        self.filename = filename

        self.description = description
        if description is None:
            self.description = 'Controller created %d.%d.%d - %d:%d:%d.' % (now.day, now.month, now.year,
                    now.hour, now.minute, now.second)

        self.results = None

    def clear(self, filePath = None, description = None):
        self.__init__(filePath = self.filePath, description = description)

    def info(self):
        print (self.description)

    def readResults(self, filename = None):
        if filename is None:
            # if no filename is specified, return the newest simulation
            # (1) get all files
            listOfFiles = [f for f in os.listdir(self.filePath) if os.path.isfile(os.path.join(self.filePath,f))]
            # (2) filter for csv files
            listOfFiles = [f for f in listOfFiles if f.endswith('.csv')]
            if len(listOfFiles) == 0:
                print ('Can\'t open latest simulation results. No result files in specified file path.')
                return
            # (3) get the newest one
            listOfFiles.sort(key = lambda x: os.path.getmtime(
                os.path.join(self.filePath, x)))
            filename = os.path.join(self.filePath,listOfFiles[-1])
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

        self.results = pd.read_csv(filename)
        try:
            self.description = self.results['description'][0]
        except:
            return

    def plotAveragesVsDimensions(self, errorbar = False, stdOfMean = False, xmin = None, xmax = None, ymin = None, ymax = None):
        if self.results is None:
            print ('No results loaded yet. Read the latest simulation results with SimulationsController.readResults().')
            return

        plt.figure()

        title = 'Alternate Ticks'
        nAverage = self.results['n_average'][0]
        if not all(navg == self.results['n_average'][0] for navg in self.results['n_average']):
            print ('Warning: Alternate ticks have been averaged over different number of iterations ' + \
                    'for different dimensions...')
        else:
            title += ' averaged over %d iterations.' % nAverage 

        if errorbar:
            if stdOfMean:
                plt.errorbar(self.results['dimension'], self.results['average_alternate_ticks'],
                    yerr = self.results['std_alternate_ticks']/np.sqrt(nAverage), ls = '-', c = 'b')
                title += ' (error bar from std of mean)'
            else:
                plt.errorbar(self.results['dimension'], self.results['average_alternate_ticks'],
                    yerr = self.results['std_alternate_ticks'], ls = '-', c = 'b')
                title += ' (error bar from std)'

        else:
            plt.plot(self.results['dimension'], self.results['average_alternate_ticks'],
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
        plt.title(title)
        plt.show()

    def plotHistogram(self, sim):
        if self.results is None:
            print ('No results loaded yet. Read the latest simulation results with SimulationController.readResults().')
            return

        # read the list of alternate ticks
        alternateTicksAsString = None
        title = 'Histogram of # alternate ticks for Simulation '
        if type(sim) is int:
            try:
                alternateTicksAsString = self.results['list_alternate_ticks'][sim-1]
                title += str(sim) + ' with dimension ' + str(self.results['dimension'][sim-1]) + '.'
            except:
                print ('Simulation number does not exist.')
                return
        elif type(sim) is str:
            alternateTicksAsString = self.results['list_alternate_ticks'][self.results['label'] == simLabel]
            title += simLabel + ' with dimension ' + str(self.results['dimension'][self.results['label'] == simLabel]) + '.'
        else:
            print ('Unsupported type for simulation. Please provide the number or the label of the simulation.')
            return

        alternateTicks = map(int, alternateTicksAsString[1:-1].split(','))
        histo, bins = np.histogram(alternateTicks, bins = range(max(alternateTicks) + 2))
        plt.bar(bins[:-1], histo, width = 1, color = 'b')
        plt.xlabel('# Alternate Ticks')
        plt.ylabel('Appearances')
        plt.title(title)
        plt.show()

        plt.figure()


    def add(self, simulation):
        self.simulations.append(simulation)
        self.nSimulations += 1

    def performSimulations(self):
        from sys import stdout

        for index in range(self.nSimulations):
            sim = self.simulations[index]

            # check if it is a valid simulation
            if not sim.ready:
                print('Simulation %d/%d is not valid. Skipping...' % (index+1, self.nSimulations))
                stdout.flush()
                continue

            alternateTicks = []
            currentAverage = 0.
            for lap in range(sim.nAverage):
                # bool variable to indicate if a simulation needs to be stopped
                stopSimulation = False

                # tickString will look like 0101010... and indicates which clocks have ticked
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
                            relevant = len(tickString) % sim.nClocks
                            if relevant == 0:
                                tickString += str(tickingClock)
                            else:
                                if not tickingClock in map(int, list(tickString[-relevant:])):
                                    tickString += str(tickingClock)
                                else:
                                    stopSimulation = True

                        elif sim.mode == 'strict' and sim.order is None:
                            if not tickingClock in map(int, list(tickString)):
                                tickString += str(tickingClock)
                            else:
                                stopSimulation = True
                            if len(tickString) == sim.nClocks:
                                sim.order = map(int, list(tickString)) 

                        else: # this means mode == 'strict'
                            if len(tickString) == 0:
                                requiredClock = sim.order[0]
                            else:
                                requiredClock = sim.order[(sim.order.index(int(tickString[-1])) + 1) % sim.nClocks]

                            if requiredClock == tickingClock:
                                tickString += str(tickingClock)
                            else:
                                stopSimulation = True

                    nAlternateTicks = len(tickString)
                    tempCurrentAverage = currentAverage * lap / float(lap + 1) + nAlternateTicks / float(lap + 1)


                    simOutput = 'Sim ' + ' '*(6-(len(str(index+1)))) + str(index+1) + ' / ' + str(self.nSimulations) + ' '*(6-len(str(self.nSimulations))) + '  ||  '
                    lapOutput = 'Lap ' + ' '*(6-(len(str(lap+1)))) + str(lap+1) + ' / ' + str(sim.nAverage) + ' '*(6-len(str(sim.nAverage))) + '  ||  '
                    currentExperimentOutput = '['

                    maxStringLength = 117
                    appendix = ''
                    if stopSimulation:
                        maxStringLength -= 9 
                        appendix = '-Stopped.'
                    if len(tickString) > maxStringLength:
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
                        sim.reset()
                        break

            self.save(sim, alternateTicks)

    def save(self, sim, alternateTicks):
        pathToFile = os.path.join(self.filePath, self.filename)
        avgAlternateTicks = np.mean(alternateTicks)
        stdAlternateTicks = np.std(alternateTicks)
        if not os.path.isfile(pathToFile):
            data = pd.DataFrame(columns = [
                'n_average',
                'dimension',
                'clock_states',
                'reset_states',
                'hamiltonian',
                'tickProjection',
                'notickProjection',
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
        statesString = str([str(Qobj(sim.clockKets[i])) for i in range(sim.nClocks)])
        hamiltonianString = str(Qobj(sim.hamiltonian))
        tickProjectionString = str(Qobj(sim.tickProjector))
        noTickProjectionString = str(Qobj(sim.noTickProjector))

        appendix = 1
        while sim.label in data['label']:
            newLabel = sim.label + str(appendix)
            print ('Simulation with label ' + sim.label + ' already exists. Changed label to ' + \
                    newlabel + '.')
            sim.label = newLabel

        newData = pd.DataFrame({
            'n_average':[sim.nAverage],
            'dimension':[sim.dimension],
            'clock_states':[str(statesString)],
            'reset_states':[sim.resetState],
            'hamiltonian':[hamiltonianString],
            'tickProjection':[tickProjectionString],
            'notickProjection':[noTickProjectionString],
            'list_alternate_ticks':[alternateTicks],
            'average_alternate_ticks':[avgAlternateTicks],
            'std_alternate_ticks':[stdAlternateTicks],
            'label':[sim.label],
            'description':[self.description]
            }, index = [ind])
        data = pd.concat([data, newData])
        data.to_csv(pathToFile, index = False, index_label = False)
        print('Saved simulation to ' + pathToFile  + '.', end = '\n')
        stdout.flush()
