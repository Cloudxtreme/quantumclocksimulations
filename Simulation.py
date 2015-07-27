from qutip import *
import numpy as np
from scipy.linalg import expm2 as expm
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats as stats



class Simulation:
    def __init__(self, tau = 1., nAverage = 100, mode = None, order = None, label = None):
        self.tau = tau

        self.initialClockStates = []
        self.clockKets = []
        self.clockMats = []
        self.clockStates = []
        self.entangledClockKet = None
        self.entangledClockMat = None
        self.entangledClockState = None
        self.initialEntangledClockState = None

        self.resetState = None

        self.hamiltonian = None
        self.unitary = None
        self.unitaryDagger = None
        self.entangledHamiltonian = None
        self.entangledUnitary = None
        self.entangledUnitaryDagger = None

        self.tickProjector = None
        self.noTickProjector = None
        self.tickProjectorDagger = None
        self.noTickProjectorDagger = None

        self.projectorMode = 'equal'
        self.differentTickProjectors = []
        self.differentNoTickProjectors = []
        self.differentTickProjectorsDagger = []
        self.differentNoTickProjectorsDagger = []

        self.entangledTickProjectors = []
        self.entangledNoTickProjectors = []
        self.entangledTickProjectorsDagger = []
        self.entangledNoTickProjectorsDagger = []

        self.nClocks = 0

        self.nAverage = nAverage

        self.mode = mode
        self.order = order
        self.originalOrder = order

        self.dimension = None

        self.label = label

        self.efficient = False
        self.veryEfficient = False
        self.entangled = False

        self.ready = False

    def setNAverage(self, nAverage):
        self.nAverage = nAverage

    def setMode(self, mode):
        self.mode = mode

    def setOrder(self, order):
        self.order = order

    def setLabel(self, label):
        self.label = label

    def entangle(self):
        if self.checkForEntanglement():
            self.entangled = True
            if self.hamiltonian is not None:
                self.getEntangledHamiltonian()
            if self.tickProjector is not None and self.projectorMode == 'equal':
                self.getEntangledProjectors()
            if None in self.differentTickProjectors and self.projectorMode == 'different':
                print 'Warning: Projectors have not been specified for all clocks. ' + \
                        'Will entangle the projectors as soon as they are specified for all clocks.'
            elif None not in self.differentTickProjectors and self.projectorMode == 'different':
                self.getEntangledProjectors()
            if len(self.clockKets) > 0:
                self.getEntangledClockState()

    def checkForEntanglement(self, potentialNewState = None):
        allKets = None
        if potentialNewState is None:
            allKets = self.clockKets
        else:
            allKets = self.clockKets + [potentialNewState]
        if self.nClocks == 0:
            return True
        if not all(np.array_equal(allKets[0], state) for state in allKets):
            print 'Warning: Entanglement is ambiguous. All Clock State vectors should be identical. ' + \
                    'Cowardly refusing to perform action.'
            return False
        return True

    def checkDimensionConsistency(self, obj):
        dim = max(obj.shape)
        if self.dimension is None:
            self.dimension = dim
            return True

        if not (dim == self.dimension):
            print 'Warning: Dimension mismatch. Cowardly refused to perform action.'
            return False

        return True

    def createPeresHamiltonian(self, dim = None, tP = 1.):
        if dim is None:
            if self.dimension is None:
                print 'Warning: The simulation is still dimensionless. If you want to create a predefined ' + \
                        'Hamiltonian, you have to specify a dimension. Cowardly refused to perform action.'
                return
            else:
                dim = self.dimension
        else:
            if not self.checkDimensionConsistency(np.zeros(dim)):
                return
        h = np.zeros((dim, dim), dtype = np.complex_)
        v = np.eye(dim, dtype = np.complex_)
        u = np.zeros((dim, dim), dtype = np.complex_)
        for i in range(dim):
            for j in range(dim):
                u[:,i] += 1./np.sqrt(dim) * np.exp(1.j * 2 * np.pi * i * j / dim) * v[:,j]

        for i in range(dim):
            h += 2 * np.pi / (dim * tP) * i * np.outer(u[:,i], u[:,i].T.conj())

        hamiltonian = 0.5 * (h + h.T.conj())
        self.addHamiltonian(hamiltonian)

    def createNaiveHamiltonian(self, dim = None, corners = False):
        if dim is None:
            if self.dimension is None:
                print 'Warning: The simulation is still dimensionless. If you want to create a predefined ' + \
                        'Hamiltonian, you have to specify a dimension. Cowardly refused to perform action.'
                return
            else:
                dim = self.dimension
        else:
            if not self.checkDimensionConsistency(np.zeros(dim)):
                return
        h = np.diag(np.ones(dim-1, dtype = np.complex_),-1) + \
                np.diag(np.ones(dim-1, dtype = np.complex_),1)
        if corners:
            h[0,dim-1] = 1.
            h[dim-1,0] = 1.
        self.addHamiltonian(h)

    def createUniformSuperpositionState(self, dim = None, left = 0, right = 0):
        if dim is None:
            if self.dimension is None:
                print 'Warning: The simulation is still dimensionless. If you want to create a predefined ' + \
                        'initial state, you have to specify a dimension. Cowardly refused to perform action.'
                return
            else:
                dim = self.dimension
        else:
            if not self.checkDimensionConsistency(np.zeros(dim)):
                return
        state = np.zeros(dim, dtype = np.complex_)
        for pos in range(left, right+1):
            state[pos] = 1.
        state = state / np.linalg.norm(state)
        self.addClockState(state)

    def createGaussianSuperpositionState(self, dim = None, cent = 0, width = 1):
        if dim is None:
            if self.dimension is None:
                print 'Warning: The simulation is still dimensionless. If you want to create a predefined ' + \
                        'initial state, you have to specify a dimension. Cowardly refused to perform action.'
                return
            else:
                dim = self.dimension
        else:
            if not self.checkDimensionConsistency(np.zeros(dim)):
                return
        state = np.zeros(dim, dtype = np.complex_)
        for i in range(width + 1):
            leftPos = cent - i
            rightPos = (cent + i) % dim
            state[leftPos] = stats.norm(0,1).pdf(i * 3. / width)
            state[rightPos] = stats.norm(0,1).pdf(i * 3. / width)

        state = (1. / np.linalg.norm(state)) * state
        self.addClockState(state)

    def createStandardProjectors(self, dim = None, delta = 0.1, d0 = 1, clock = None, loc = None):
        if dim is None:
            if self.dimension is None:
                print 'Warning: The simulation is still dimensionless. If you want to create a predefined ' + \
                        'initial state, you have to specify a dimension. Cowardly refused to perform action.'
                return
            else:
                dim = self.dimension
        else:
            if not self.checkDimensionConsistency(np.zeros(dim)):
                return
        if loc is None:
            loc = dim
        loc = loc % dim
        tick = np.zeros((dim, dim), dtype = np.complex_)
        noTick = np.eye(dim, dtype = np.complex_)
        for i in range(1, d0+1):
            tick[loc-i,loc-i] = np.sqrt(delta)
            noTick[loc-i,loc-i] = np.sqrt(1. - delta)
        self.addProjectors(tick, noTick, clock = clock)

    def addHamiltonian(self, hamiltonian):
        hamiltonianOk, hamiltonian = self.checkHamiltonian(hamiltonian)
        if hamiltonianOk:
            dimensionsOk = self.checkDimensionConsistency(hamiltonian)
            if dimensionsOk:
                self.hamiltonian = hamiltonian
                self.unitary = expm(-1.j * self.tau * hamiltonian)
                self.unitaryDagger = self.unitary.T.conj()
                if self.entangled:
                    self.getEntangledHamiltonian()

    def getEntangledHamiltonian(self):
        # if the Hamiltonian is added before the states,
        # this should just be ignored, because
        # entanglement does not make sense yet...
        if self.nClocks == 0:
            return
        self.entangledHamiltonian = sum([np.kron(np.kron(np.eye(self.dimension ** i),
            self.hamiltonian), np.eye(self.dimension ** (self.nClocks - i - 1))) \
                    for i in range(self.nClocks)])
        self.entangledUnitary = expm(-1.j * self.tau * self.entangledHamiltonian)
        self.entangledUnitaryDagger = self.entangledUnitary.T.conj()
        print 'Entangled Hamiltonian.'

    def addClockState(self, clockState):
        clockStateOk, clockState = self.checkClockState(clockState)
        if clockStateOk:
            dimensionOk = self.checkDimensionConsistency(clockState)
            if dimensionOk:
                # check if the state is already normalized.
                # if not, normalize it:
                norm = np.linalg.norm(clockState)
                if np.abs(norm - 1) >= 10**-1:
                    print 'Warning: You handed an unnormalized state to the simulation. State has been normalized.'
                    clockState = (1. / norm) * clockState
                # check if the clockworks are already entangled,
                # only add the state if it matches the other ones
                if self.entangled:
                    if self.checkForEntanglement(potentialNewState = clockState):
                        self.clockKets.append(clockState)
                        self.nClocks += 1
                        # reentangle
                        self.unentangle()
                        self.entangle()
                else:
                    self.clockKets.append(clockState)
                    self.nClocks += 1
                    while len(self.differentTickProjectors) < self.nClocks:
                        self.differentTickProjectors.append(None)
                        self.differentNoTickProjectors.append(None)
                        self.differentTickProjectorsDagger.append(None)
                        self.differentNoTickProjectorsDagger.append(None)

    def clear(self):
        self.__init__(tau = self.tau, nAverage = self.nAverage,
                mode = self.mode, order = self.order)

    def reset(self):
        if self.entangled:
            self.entangledClockState = self.initialEntangledClockState
        else:
            self.clockStates = [self.initialClockStates[i] for i in range(self.nClocks)]
        self.order = self.originalOrder

    def evolve(self, animationClock = None):
        if animationClock is not None: # evolving for an animation (not a simulation)
            self.clockKets[animationClock] = self.unitary.dot(self.clockKets[animationClock])
            return

        if self.entangled:
            self.entangledClockState = self.entangledUnitary.dot(self.entangledClockState)
            if not self.veryEfficient:
                self.entangledClockState = self.entangledClockState.dot(self.entangledUnitaryDagger)
        else:
            for i in range(self.nClocks):
                self.clockStates[i] = self.unitary.dot(self.clockStates[i])
                if not self.veryEfficient:
                    self.clockStates[i] = self.clockStates[i].dot(self.unitaryDagger)

    def removeClockState(self, ind = -1):
        if self.nClocks == 0:
            print 'Nothing to remove...'
            return
        if ind < self.nClocks:
            self.clockKets.pop(ind)
            self.differentTickProjectors.pop(ind)
            self.differentTickProjectorsDagger.pop(ind)
            self.differentNoTickProjectors.pop(ind)
            self.differentNoTickProjectorsDagger.pop(ind)
            if self.nClocks == 0:
                self.projectorMode = 'equal'
            self.nClocks -= 1
            # if the clocks are entangled: reentangle them
            if self.entangled:
                self.unentangle()
                self.entangle()
            if self.hamiltonian is None and self.tickProjector is None and self.nClocks == 0:
                self.dimension = None

    def measure(self):
        res = []
        for i in range(self.nClocks):
            # calculate probabilities for a tick
            proj = None
            prob = 0.
            if self.entangled:
                if self.veryEfficient:
                    proj = self.entangledTickProjectors[i] * self.entangledClockState * \
                            self.entangledTickProjectorsDagger[i]
                    prob = np.linalg.norm(proj)
                elif self.efficient:
                    proj = np.copy(self.entangledClockState)
                    for j in range(self.dimension ** self.nClocks):
                        proj[j,:] *= self.entangledTickProjectors[i][j]
                        proj[:,j] *= self.entangledTickProjectorsDagger[i][j]
                    prob = proj.trace()
                else:
                    proj = self.entangledTickProjectors[i].dot(
                        self.entangledClockState).dot(self.entangledTickProjectorsDagger)
                    prob = proj.trace()
            else:
                if self.projectorMode == 'equal':
                    if self.veryEfficient:
                        proj = self.tickProjector * self.clockStates[i] * self.tickProjectorDagger
                        prob = np.linalg.norm(proj)
                    elif self.efficient:
                        proj = np.copy(self.clockStates[i])
                        for j in range(self.dimension):
                            proj[j,:] *= self.tickProjector[j]
                            proj[:,j] *= self.tickProjectorDagger[j]
                        prob = proj.trace()
                    else:
                        proj = self.tickProjector.dot(self.clockStates[i]).dot(
                                self.tickProjectorDagger)
                        prob = proj.trace()
                else:
                    if self.veryEfficient:
                        proj = self.differentTickProjectors[i] * self.clockStates[i] * self.differentTickProjectorsDagger[i]
                        prob = np.linalg.norm(proj)
                    elif self.efficient:
                        proj = np.copy(self.clockStates[i])
                        for j in range(self.dimension):
                            proj[j,:] *= self.differentTickProjectors[i][j]
                            proj[:,j] *= self.differentTickProjectorsDagger[i][j]
                        prob = proj.trace()
                    else:
                        proj = self.differentTickProjectors[i].dot(self.clockStates[i]).dot(
                                self.differentTickProjectorsDagger[i])
                        prob = proj.trace()


            rand = np.random.uniform()

            # simulate a measurement by drawing a random number
            # if the random number is smaller than the probability: tick
            if rand <= prob:
                res.append(i)

                # reset the state (if that's an option)
                if self.entangled:
                    if self.resetState is not None:
                        # this really does not make any sense and should never occur
                        proj = self.initialEntangledState
                    else:
                        proj = (1. / prob) * proj
                else:
                    if self.resetState is not None:
                        proj = self.initialClockStates[self.resetState]
                    else:
                        proj = (1. / prob) * proj

            else:
                # unsucessful measurement, project the state to the new state
                if self.entangled:
                    if self.veryEfficient:
                        proj = self.entangledNoTickProjectors[i] * self.entangledClockState * \
                                self.entangledNoTickProjectorsDagger[i]
                        proj = (1. / np.linalg.norm(proj)) * proj
                    elif self.efficient:
                        proj = np.copy(self.entangledClockState)
                        for j in range(self.dimension ** self.nClocks):
                            proj[j,:] *= self.entangledNoTickProjectors[i][j]
                            proj[:,j] *= self.entangledNoTickProjectorsDagger[i][j]
                        proj = (1. / proj.trace()) * proj
                    else:
                        proj = self.entangledNoTickProjectors[i].dot(self.entangledClockState).dot(
                                self.entangledNoTickProjectorsDagger[i])
                        proj = (1. / proj.trace()) * proj
                else:
                    if self.projectorMode == 'equal':
                        if self.veryEfficient:
                            proj = self.noTickProjector * self.clockStates[i] *\
                                    self.noTickProjectorDagger
                            proj = (1. / np.linalg.norm(proj)) * proj
                        elif self.efficient:
                            proj = np.copy(self.clockStates[i])
                            for j in range(self.dimension):
                                proj[j,:] *= self.noTickProjector[j]
                                proj[:,j] *= self.noTickProjectorDagger[j]
                            proj = (1. / proj.trace()) * proj
                        else:
                            proj = self.noTickProjector.dot(self.clockStates[i]).dot(
                                    self.noTickProjectorDagger)
                            proj = (1. / proj.trace()) * proj
                    else:
                        if self.veryEfficient:
                            proj = self.differentNoTickProjectors[i] * self.clockStates[i] *\
                                    self.differentNoTickProjectorsDagger[i]
                            proj = (1. / np.linalg.norm(proj)) * proj
                        elif self.efficient:
                            proj = np.copy(self.clockStates[i])
                            for j in range(self.dimension):
                                proj[j,:] *= self.differentNoTickProjectors[i][j]
                                proj[:,j] *= self.differentNoTickProjectorsDagger[i][j]
                            proj = (1. / proj.trace()) * proj
                        else:
                            proj = self.differentTickProjectors[i].dot(self.clockStates[i]).dot(
                                    self.differentNoTickProjectorsDagger[i])
                            proj = (1. / proj.trace()) * proj

            if self.entangled:
                self.entangledClockState = proj
            else:
                self.clockStates[i] = proj
        return res

    def run(self, maxIterations = 50000):
        for _ in range(maxIterations):
            self.evolve()
            ticks = self.measure()
            if len(ticks) > 0:
                return ticks
        return []

    def summary(self):
        print '+++ Summary of this Simulation +++'
        print
        print '----------------------------------'
        print
        print 'Dimension: ' +  str(self.dimension)
        print
        print 'Number of clocks: %d' % self.nClocks
        for i in range(self.nClocks):
            print
            print 'CLOCK %d' % (i+1)
            print Qobj(self.clockKets[i])
            if len(self.clockStates) > 0:
                print 'Currently in the following state:'
                print Qobj(self.clockStates[i])
        print
        print 'Hamiltonian specified: ' + str(self.hamiltonian is not None)
        if self.hamiltonian is not None:
            print
            print 'HAMILTONIAN'
            print Qobj(self.hamiltonian)
        print
        print 'Projectors:'
        if self.projectorMode == 'equal':
            if self.tickProjector is not None:
                print 'TICK PROJECTOR'
                print Qobj(self.tickProjector)
                print 'NO-TICK PROJECTOR'
                print Qobj(self.noTickProjector)
                print
            else:
                print 'not specified'
                print
        else:
            for i in range(self.nClocks):
                print 'CLOCK %s:' % (i)
                if self.differentTickProjectors[i] is not None:
                    print 'TICK PROJECTOR'
                    print Qobj(self.differentTickProjectors[i])
                    print 'NO-TICK PROJECTOR'
                    print Qobj(self.differentNoTickProjectors[i])
                    print
                else:
                    print 'not specified'
                    print
        print 'Entanglement: ' + str(self.entangled)
        if self.entangled:
            print
            print '----------------------------------'
            print
            print 'ENTANGLED CLOCK STATE'
            print Qobj(self.entangledClockKet)
            if self.entangledClockState is not None:
                print 'Currently in the following state:'
                print Qobj(self.entangledClockState)
            if self.entangledHamiltonian is not None:
                print
                print 'ENTANGLED HAMILTONIAN'
                print Qobj(self.entangledHamiltonian)
            if self.tickProjector is not None:
                for i in range(self.nClocks):
                    print
                    print 'ENTANGLED TICK-PROJECTOR %d' % (i + 1)
                    print Qobj(self.entangledTickProjectors[i])
                    print 'ENTANGLED NO-TICK-PROJECTOR %d' % (i + 1)
                    print Qobj(self.entangledNoTickProjectors[i])

    def initialize(self, veryEfficient = False, resetState = None):
        # check if simulation can be initialized
        if self.nClocks < 2:
            print 'Can\'t run the simulation yet. Not enough clocks.'
            return
        if self.hamiltonian is None:
            print 'Can\'t run the simulation yet. No Hamiltonian specified.'
            return
        if self.tickProjector is None and self.projectorMode == 'equal':
            print 'Can\'t run the simulation yet. No projectors sepecified.'
            return
        if None in self.differentTickProjectors and self.projectorMode == 'different':
            print 'Can\'t run the simulation yet. Projectors are not specified for all clocks.'
            return

        if veryEfficient:
            print 'Warning: the very efficient option has been chosen. This is only possible ' + \
                    'if the Peres Hamiltonian is used with theta = tau = 1 and the initial states ' + \
                    'are NOT superpositions in the computational basis. If this is not the case, the ' + \
                    'simulation will lead to wrong results.'
            self.veryEfficient = veryEfficient

        self.resetState = resetState

        if self.entangled:
            if self.veryEfficient:
                self.entangledClockState = self.entangledClockKet
                self.initialEntangledClockState = self.entangledClockKet
            else:
                self.entangledClockMat = np.outer(self.entangledClockKet,
                        self.entangledClockKet)
                self.entangledClockState = self.entangledClockMat
                self.initialEntangledClockState = self.entangledClockMat
        else:
            if self.veryEfficient:
                self.clockStates = [self.clockKets[i] for i in range(self.nClocks)]
                self.initialClockStates = [self.clockKets[i] for i in range(self.nClocks)]
            else:
                self.clockMats = [np.outer(ket, ket) for ket in self.clockKets]
                self.clockStates = [self.clockMats[i] for i in range(self.nClocks)]
                self.initialClockStates = [self.clockMats[i] for i in range(self.nClocks)]

        if self.label is None:
            self.label = 'dim' + str(self.dimension)

        if self.mode is None:
            if self.nClocks == 2:
                self.mode = 'strict'
            else:
                self.mode = 'normal'

        self.ready = True

    def getEntangledClockState(self):
        entangledKet = np.zeros(self.dimension ** self.nClocks, dtype = np.complex_)
        referenceKet = self.clockKets[0]
        for i in range(self.dimension):
            if referenceKet[i] == 0:
                continue
            t = np.zeros(self.dimension, dtype = np.complex_)
            t[i] = 1.
            prod = t
            for _ in range(self.nClocks-1):
                prod = np.kron(prod, t)
            entangledKet += referenceKet[i] * prod
        self.entangledClockKet = entangledKet
        print 'Entangled clock states.'

    def addProjectors(self, tick, noTick, clock = None):
        if clock is None and self.projectorMode == 'different':
            output = 'No clock specified, although done previously. For the following clocks, no projectors exist yet: '
            for i in range(self.nClocks):
                if self.differentTickProjectors[i] is None:
                    output += str(i) + ' '
            output += '.'
            output += 'Please enter the clock number you would like to use this projector for. (Possible answers: ' + \
                    str(range(self.nClocks)) + ') If you would like to use ' + \
                    'this projector for all clocks, just type \'all\'. Any other input will cancel the action.'
            print output
            ans = raw_input('> ')
            if ans.isdigit():
                ans = int(ans)
                if ans in range(self.nClocks):
                    clock = ans 
            else:
                if not ans == 'all':
                    print 'Canceled.'
                    return
                else:
                    self.differentTickProjectors = [None] * self.nClocks
                    self.differentNoTickProjectors = [None] * self.nClocks
                    self.differentTickProjectorsDagger = [None] * self.nClocks
                    self.differentNoTickProjectorsDagger = [None] * self.nClocks
        projectorsOk, tick, noTick = self.checkProjectors(tick, noTick)
        if projectorsOk:
            dimensionsOk = self.checkDimensionConsistency(tick)
            if dimensionsOk:
                self.checkForEfficiency(tick)
                tickProj = None
                noTickProj = None
                tickProjDag = None
                noTickProjDag = None
                if self.efficient:
                    tickProj = np.diag(tick)
                    noTickProj = np.diag(noTick)
                    tickProjDag = np.diag(tick).conj()
                    noTickProjDag = np.diag(noTick).conj()
                else:
                    tickProj = tick
                    noTickProj = noTick
                    tickProjDag = tick.T.conj()
                    noTickProjDag = noTick.T.conj()
                if clock is None:
                    self.projectorMode = 'equal'
                    self.tickProjector = tickProj
                    self.noTickProjector = noTickProj
                    self.tickProjectorDagger = tickProjDag
                    self.noTickProjectorDagger = noTickProjDag
                    if self.entangled:
                        self.getEntangledProjectors()
                else:
                    if clock >= self.nClocks:
                        print 'Warning %s clocks added so far (Indexing starting at 0). Starting the simulation now will lead to an error. ' % (self.nClocks) + \
                                'Cowardly refusing to perform action.'
                        return
                    else:
                        self.projectorMode = 'different'
                        self.differentTickProjectors[clock] = tickProj
                        self.differentNoTickProjectors[clock] = noTickProj
                        self.differentTickProjectorsDagger[clock] = tickProjDag
                        self.differentNoTickProjectorsDagger[clock] = noTickProjDag
                    if self.entangled:
                        if not None in self.differentTickProjectors:
                            self.getEntangledProjectors()

    def getEntangledProjectors(self):
        if self.projectorMode == 'equal':
            if self.efficient:
                self.entangledTickProjectors = [
                        np.kron(np.kron(np.ones(self.dimension ** i), self.tickProjector),
                            np.ones(self.dimension ** (self.nClocks - i - 1)))
                        for i in range(self.nClocks)]
                self.entangledNoTickProjectors = [
                        np.kron(np.kron(np.ones(self.dimension ** i), self.noTickProjector),
                            np.ones(self.dimension ** (self.nClocks - i - 1)))
                        for i in range(self.nClocks)]
                self.entangledTickProjectorsDagger = [projector.conj() for projector in self.entangledTickProjectors]
                self.entangledNoTickProjectorsDagger = [projector.conj() for projector in self.entangledNoTickProjectors]
            else:
                self.entangledTickProjectors = [
                        np.kron(np.kron(np.eye(self.dimension ** i), self.tickProjector),
                            np.eye(self.dimension ** (self.nClocks - i - 1)))
                        for i in range(self.nClocks)]
                self.entangledNoTickProjectors = [
                        np.kron(np.kron(np.eye(self.dimension ** i), self.noTickProjector),
                            np.eye(self.dimension ** (self.nClocks - i - 1)))
                        for i in range(self.nClocks)]
                self.entangledTickProjectorsDagger = [mat.T.conj() for mat in self.entangledTickProjectors]
                self.entangledNoTickProjectorsDagger = [mat.T.conj() for mat in self.entangledNoTickProjectors]
        else:
            if self.efficient:
                self.entangledTickProjectors = [
                        np.kron(np.kron(np.ones(self.dimension ** i), self.differentTickProjectors[i]),
                            np.ones(self.dimension ** (self.nClocks - i - 1)))
                        for i in range(self.nClocks)]
                self.entangledNoTickProjectors = [
                        np.kron(np.kron(np.ones(self.dimension ** i), self.differentNoTickProjectors[i]),
                            np.ones(self.dimension ** (self.nClocks - i - 1)))
                        for i in range(self.nClocks)]
                self.entangledTickProjectorsDagger = [projector.conj() for projector in self.entangledTickProjectors]
                self.entangledNoTickProjectorsDagger = [projector.conj() for projector in self.entangledNoTickProjectors]
            else:
                self.entangledTickProjectors = [
                        np.kron(np.kron(np.eye(self.dimension ** i), self.differentTickProjectors[i]),
                            np.eye(self.dimension ** (self.nClocks - i - 1)))
                        for i in range(self.nClocks)]
                self.entangledNoTickProjectors = [
                        np.kron(np.kron(np.eye(self.dimension ** i), self.differentNoTickProjectors[i]),
                            np.eye(self.dimension ** (self.nClocks - i - 1)))
                        for i in range(self.nClocks)]
                self.entangledTickProjectorsDagger = [mat.T.conj() for mat in self.entangledTickProjectors]
                self.entangledNoTickProjectorsDagger = [mat.T.conj() for mat in self.entangledNoTickProjectors]
        print 'Entangled projectors.'

    def unentangle(self):
        self.entangled = False
        self.entangledHamiltonian = None
        self.entangledUnitary = None
        self.entangledUnitaryDagger = None
        self.entangledClockState = None
        self.entangledClockKet = None
        self.entangledClockMat = None
        self.entangledTickProjectors = []
        self.entangledTickProjectorsDagger = []
        self.entangledNoTickProjectors = []
        self.entangledNoTickProjectorsDagger = []

    def checkForEfficiency(self, tick):
        # if the projectors are diagonal matrices, we can calculate everything significantly
        # more efficiently.
        diagMat = np.diag(np.diag(tick))
        if np.array_equal(tick, diagMat):
            self.efficient = True
        else:
            self.efficient = False

    def checkHamiltonian(self, hamiltonian):
        # Does the Hamiltonian have a supported format?
        try:
            hamiltonian = np.array(hamiltonian).squeeze().astype(np.complex_)
        except:
            print 'Warning: Hamiltonian does not have a supported format. Use list or numpy array. ' + \
                    'Cowardly refused to perform action.'
            return False, None

        # Is it a valid Hamiltonian? (2-dimensional array and hermitian)
        try:
            asMat = np.matrix(hamiltonian)
        except:
            print 'Warning: Hamiltonian has invalid dimensions. It needs to be an n x n matrix. ' + \
                    'Cowardly refused to perform action.'
            return False, None

        if not np.array_equal(asMat, asMat.getH()):
            print 'Warning: Hamiltonian is not hermitian. Cowardly refused to perform action.'
            return False, None
        return True, hamiltonian

    def checkClockState(self, clockState):
        # Does the clock have a supported format?
        try:
            clockState = np.array(clockState).squeeze().astype(np.complex_)
        except:
            print 'Warning: Clock State does not have a supported format. Use list or numpy array. ' + \
                    'Cowardly refused to perform action.'
            return False, None

        # Is it a valid vector in the Hilbert space?
        dim = max(clockState.shape)
        if not clockState.shape == (dim, ):
            print 'Warning: Clock State has invalid dimensions. It needs to be a 1 x n vector or matrix. ' + \
                    'Cowardly refused to perform action.'
            return False, None
        return True, clockState

    def checkProjectors(self, tick, noTick):
        # Do the projectors have a supported format?
        try:
            tick = np.array(tick).squeeze().astype(np.complex_)
            noTick = np.array(noTick).squeeze().astype(np.complex_)
        except:
            print 'Warning: Projectors do not have a supported format. Use list or numpy array. ' + \
                        'Cowardly refused to perform action.'
            return False, None, None

        # Are the projectors matrices?
        try:
            tickAsMat = np.matrix(tick)
            noTickAsMat = np.matrix(noTick)
        except:
            print 'Warning: Projectors have invalid dimensions. They need to be n x n matrices. ' + \
                        'Cowardly refused to perform action.'
            return False, None, None

        # Do the dimensions of the matrices match and if so, do their squares add up to unity?
        try:
            unity = tickAsMat.dot(tickAsMat) + noTickAsMat.dot(noTickAsMat)
            try:
                np.testing.assert_array_almost_equal(unity, np.eye(max(tickAsMat.shape), dtype = np.complex))
            except:
                print 'Warning: The two projectors do not add up to unity which makes them invalid ' + \
                            'for the use as a POVM. Cowardly refused to perfrom action.'
                return False, None, None
        except:
            print 'Warning: Dimension mismatch for projectors. Cowardly refused to perform action.'
            return False, None, None

        return True, tick, noTick

    def preview(self, style = '-', nSteps = 10000, clock = 0):
        if self.hamiltonian is None:
            print 'Please specify a Hamiltonian.'
            return
        if self.nClocks == 0:
            print 'Please specify at least one state.'
            return
        if self.entangled:
            print 'Can\'t animate an entangled clock.'
            return

        fig = plt.figure()

        clockKet = np.copy(self.clockKets[clock])
        frames = []
        x = np.array([i for i in range(self.dimension)])
        for i in range(nSteps+1):
            frame = None
            if i > 0:
                self.evolve(animationClock = clock)
            y = np.array([np.abs(self.clockKets[clock][j])**2 for j in range(self.dimension)])
            frames.append(plt.plot(x, y, style, color = 'b'))

        ani = animation.ArtistAnimation(fig, frames, interval = 50)
        plt.show(ani)
        
        self.clockKets[clock] = clockKet

    def setTau(self, tau):
        self.tau = tau
        if self.hamiltonian is not None:
            self.unitary = expm(-1.j * self.tau * self.hamiltonian)
            self.unitaryDagger = self.unitary.T.conj()
            if self.entangled:
                self.getEntangledHamiltonian()
