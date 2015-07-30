from qutip import *
import numpy as np
from scipy.linalg import expm2 as expm
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats as stats



class Simulation:
    """ The Simulation Class
        ____________________


        Accessible Attributes:
        _____________________

        nAverage (int): # of times the simulation is to get the average number of alternate ticks

        mode (string): Either 'strict' or 'normal', default: 'strict' for 2 clocks, 'normal' for more than 2 clocks 

        order (list) : If is chosen to be strict, a predfined order can be defined, in which the clocks have to tick. If not specified, order is assigned automatically

        label (string) : An identification label for easy-access when studying the simulation results. If not specified, label is assigned automatically.


        Functions
        _________

        setTau(tau) : change the value of tau used in the Hamiltonian (and recalculates the Hamiltonian if necessary)

        getTau(tau) : returns the value of tau used in the Hamiltonian

        setVeryEfficient() : enables very efficient computation (only possible in special cases) 

        unsetVeryEfficient() : disables very efficient computation

        setResetState(resetState) : defines the state a clock is reset to after ticking, resetState is the index of the clock (add order)

        getDimension() : returns the dimension of the current system (None if not specified yet)

        getNClocks() : returns the number of clocks in the simulation

        isReady() : checks if the simulation is ready to be run (not that it can only be ready after running initialize() )

        createUniformSuperpositionState(dim, left, right, phaseShift) : creates a uniform superposition state of dimension dim starting at left and ending at right

        createGaussianSuperpositionState(dim, cent, width, phaseShift) : creates a Gaussian superposition state of dimension dim centered around cent with width (3sigma) width

        createNaiveHamiltonian(dim, corners) : creates a naive hamiltonian of dimension dim, corners = True means that the corners of the Hamiltonian are set to 1

        createPeresHamiltonian(dim, tP) : creates a Peres Hamiltonian of dimension dim with Fourier parameters tP

        createStandardProjectors(dim, delta, d0, clock, loc) : creates the projectors as defined in the Rankovic paper. Clock is optional and can be used to assign
                                                               a projector to a specific clock, loc specifies at which state the support starts (going backwards),
                                                               e.g. loc = dim - 1 means that the states dim - d0 to dim - 1 have support

        addHamiltonian(hamiltonian) : adds a Hamiltonian to the system (not required if one of the standard Hamiltonians is created with one of the functions above)

        addProjectors(tick, noTick, clock) : adds projectors for tick and no tick to the system (not required if ")

        addClockState(clockState) : adds a Clock State to the system (not required if one of the standard states is created with one of the functions above) 

        removeClockState(ind) : removes clock ind

        clear() : resets the simulation to an empty system with default values

        entangle() : entangles the system, note that after entangling the system once, changing a state or Hamiltonian will immediately lead to recalculation
                     of the entanglement parameters

        unentanlge() : unentangles the system

        summary() : returns a summary of the system

        preview() : can be used to animate one of the clocks (without measurement) in the Hamiltonian

        initialize() : checks if everything is OK and sets the simulation to ready (is usually called by the SimulationsController)

        run() : runs the simulation until a tick occurs

"""        

    # private functions
    # _________________

    def __init__(self, tau = 1., nAverage = 100, nTicksBeforeSentToReferee = 1, mode = None, order = None, label = None):
        # private attributes
        # __________________
        # almost everything the simulation does is done internally
        self.__tau = tau
        self.__initialClockStates = []
        self.__clockKets = []
        self.__clockMats = []
        self.__clockStates = []
        self.__entangledClockKet = None
        self.__entangledClockMat = None
        self.__entangledClockState = None
        self.__initialEntangledClockState = None

        self.__resetState = None

        self.__hamiltonian = None
        self.__unitary = None
        self.__unitaryDagger = None
        self.__entangledHamiltonian = None
        self.__entangledUnitary = None
        self.__entangledUnitaryDagger = None

        self.__tickProjector = None
        self.__noTickProjector = None
        self.__tickProjectorDagger = None
        self.__noTickProjectorDagger = None

        self.__projectorMode = 'equal'
        self.__differentTickProjectors = []
        self.__differentNoTickProjectors = []
        self.__differentTickProjectorsDagger = []
        self.__differentNoTickProjectorsDagger = []

        self.__entangledTickProjectors = []
        self.__entangledNoTickProjectors = []
        self.__entangledTickProjectorsDagger = []
        self.__entangledNoTickProjectorsDagger = []

        self.__nClocks = 0

        self.__originalOrder = order

        self.__dimension = None

        self.__efficient = False
        self.__veryEfficient = False
        self.__entangled = False

        self.__ready = False

        # public attributes
        # _________________
        # these should be the attributes that can be specified in the constructor
        # so that they can easily be changed by the user without destroying the object
        # exception: tau, since changing tau requires an additional action which is:
        # recalculating the hamiltonian
        # tau is therefore changed with setTau()

        self.nTicksBeforeSentToReferee = nTicksBeforeSentToReferee
        self.label = label
        self.nAverage = nAverage
        self.mode = mode
        self.order = order

        self.__nTicks = []

    def __checkForEfficiency(self, tick):
        # if the projectors are already a vector, they represent a diagonal matrix and
        # we already know it's efficient
        dim = max(tick.shape)
        if not tick.shape == (dim, dim):
            self.__efficient = True
            return True # the return value indicates if the objects are already vectors

        # if the projectors are diagonal matrices, we can calculate everything significantly
        # more efficiently.
        diagMat = np.diag(np.diag(tick))
        if np.array_equal(tick, diagMat):
            self.__efficient = True
            return False
        else:
            self.__efficient = False
            return False

    def __checkHamiltonian(self, hamiltonian):
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

    def __checkClockState(self, clockState):
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

    def __checkProjectors(self, tick, noTick):
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
                np.testing.assert_array_almost_equal(unity, np.eye(max(tickAsMat.shape), dtype = np.complex_))
            except:
                print 'Warning: The two projectors do not add up to unity which makes them invalid ' + \
                        'for the use as a POVM. Cowardly refused to perfrom action.'
                return False, None, None
        except:
            # it could still be the case that the projectors were given as vectors
            try:
                unity = tick**2 + noTick**2 
                try:
                    np.testing.assert_array_almost_equal(unity, np.ones(max(tick.shape), dtype = np.complex_))
                except:
                    print 'Warning: The two projectors do not add up to unity which makes them invalid ' + \
                            'for the use as a POVM. Cowardly refused to perform action.'  
                    return False, None, None
            except:
                print 'Warning: Dimension mismatch for projectors. Cowardly refused to perform action.'
                return False, None, None

        return True, tick, noTick

    def __checkForEntanglement(self, potentialNewState = None):
        allKets = None
        if potentialNewState is None:
            allKets = self.__clockKets
        else:
            allKets = self.__clockKets + [potentialNewState]
        if self.__nClocks == 0:
            return True
        if not all(np.array_equal(allKets[0], state) for state in allKets):
            print 'Warning: Entanglement is ambiguous. All Clock State vectors should be identical. ' + \
                    'Cowardly refusing to perform action.'
            return False
        return True

    def __checkDimensionConsistency(self, obj):
        dim = max(obj.shape)
        if self.__dimension is None:
            self.__dimension = dim
            return True
        if not (dim == self.__dimension):
            print 'Warning: Dimension mismatch. Cowardly refused to perform action.'
            return False
        return True

    def __getEntangledHamiltonian(self):
        # if the Hamiltonian is added before the states,
        # this should just be ignored, because
        # entanglement does not make sense yet...
        if self.__nClocks == 0:
            return
        self.__entangledHamiltonian = sum([np.kron(np.kron(np.eye(self.__dimension ** i),
            self.__hamiltonian), np.eye(self.__dimension ** (self.__nClocks - i - 1))) \
                    for i in range(self.__nClocks)])
        self.__entangledUnitary = expm(-1.j * self.__tau * self.__entangledHamiltonian)
        self.__entangledUnitaryDagger = self.__entangledUnitary.T.conj()
        print 'Entangled Hamiltonian.'

    def __getEntangledClockState(self):
        entangledKet = np.zeros(self.__dimension ** self.__nClocks, dtype = np.complex_)
        referenceKet = self.__clockKets[0]
        for i in range(self.__dimension):
            if referenceKet[i] == 0:
                continue
            t = np.zeros(self.__dimension, dtype = np.complex_)
            t[i] = 1.
            prod = t
            for _ in range(self.__nClocks-1):
                prod = np.kron(prod, t)
            entangledKet += referenceKet[i] * prod
        self.__entangledClockKet = entangledKet
        print 'Entangled clock states.'

    def testEvolve(self):
        self.__evolve()

    def testMeasure(self):
        self.__measure()

    def __getEntangledProjectors(self):
        if self.__projectorMode == 'equal':
            if self.__efficient:
                self.__entangledTickProjectors = [
                        np.kron(np.kron(np.ones(self.__dimension ** i), self.__tickProjector),
                            np.ones(self.__dimension ** (self.__nClocks - i - 1)))
                        for i in range(self.__nClocks)]
                self.__entangledNoTickProjectors = [
                        np.kron(np.kron(np.ones(self.__dimension ** i), self.__noTickProjector),
                            np.ones(self.__dimension ** (self.__nClocks - i - 1)))
                        for i in range(self.__nClocks)]
                self.__entangledTickProjectorsDagger = [projector.conj() for projector in self.__entangledTickProjectors]
                self.__entangledNoTickProjectorsDagger = [projector.conj() for projector in self.__entangledNoTickProjectors]
            else:
                self.__entangledTickProjectors = [
                        np.kron(np.kron(np.eye(self.__dimension ** i), self.__tickProjector),
                            np.eye(self.__dimension ** (self.__nClocks - i - 1)))
                        for i in range(self.__nClocks)]
                self.__entangledNoTickProjectors = [
                        np.kron(np.kron(np.eye(self.__dimension ** i), self.__noTickProjector),
                            np.eye(self.__dimension ** (self.__nClocks - i - 1)))
                        for i in range(self.__nClocks)]
                self.__entangledTickProjectorsDagger = [mat.T.conj() for mat in self.__entangledTickProjectors]
                self.__entangledNoTickProjectorsDagger = [mat.T.conj() for mat in self.__entangledNoTickProjectors]
        else:
            if self.__efficient:
                self.__entangledTickProjectors = [
                        np.kron(np.kron(np.ones(self.__dimension ** i), self.__differentTickProjectors[i]),
                            np.ones(self.__dimension ** (self.__nClocks - i - 1)))
                        for i in range(self.__nClocks)]
                self.__entangledNoTickProjectors = [
                        np.kron(np.kron(np.ones(self.__dimension ** i), self.__differentNoTickProjectors[i]),
                            np.ones(self.__dimension ** (self.__nClocks - i - 1)))
                        for i in range(self.__nClocks)]
                self.__entangledTickProjectorsDagger = [projector.conj() for projector in self.__entangledTickProjectors]
                self.__entangledNoTickProjectorsDagger = [projector.conj() for projector in self.__entangledNoTickProjectors]
            else:
                self.__entangledTickProjectors = [
                        np.kron(np.kron(np.eye(self.__dimension ** i), self.__differentTickProjectors[i]),
                            np.eye(self.__dimension ** (self.__nClocks - i - 1)))
                        for i in range(self.__nClocks)]
                self.__entangledNoTickProjectors = [
                        np.kron(np.kron(np.eye(self.__dimension ** i), self.__differentNoTickProjectors[i]),
                            np.eye(self.__dimension ** (self.__nClocks - i - 1)))
                        for i in range(self.__nClocks)]
                self.__entangledTickProjectorsDagger = [mat.T.conj() for mat in self.__entangledTickProjectors]
                self.__entangledNoTickProjectorsDagger = [mat.T.conj() for mat in self.__entangledNoTickProjectors]
        print 'Entangled projectors.'

    def __evolve(self, animationClock = None):
        if animationClock is not None: # evolving for an animation (not a simulation)
            self.__clockKets[animationClock] = self.__unitary.dot(self.__clockKets[animationClock])
            return
        if self.__entangled:
            self.__entangledClockState = self.__entangledUnitary.dot(self.__entangledClockState)
            if not self.__veryEfficient:
                self.__entangledClockState = self.__entangledClockState.dot(self.__entangledUnitaryDagger)
        else:
            for i in range(self.__nClocks):
                if self.__veryEfficient:
                    # this is only possible for Peres Hamiltonians with tP = 1
                    # therefore, we don't need that matrix multiplication at all
                    # and can just move the individual states in the superposition
                    temp = self.__clockStates[i][0]
                    self.__clockStates[i][0] = self.__clockStates[i][self.__dimension-1]
                    for j in range(self.__dimension-2,0,-1):
                        self.__clockStates[i][j+1] = self.__clockStates[i][j]
                    self.__clockStates[i][1] = temp
                else:
                    self.__clockStates[i] = self.__unitary.dot(self.__clockStates[i]).dot(self.__unitaryDagger)

    def __measure(self):
        res = []
        for i in range(self.__nClocks):
            # calculate probabilities for a tick
            proj = None
            prob = 0.
            if self.__entangled:
                if self.__veryEfficient:
                    proj = self.__entangledTickProjectors[i] * self.__entangledClockState * \
                            self.__entangledTickProjectorsDagger[i]
                    prob = np.linalg.norm(proj)
                elif self.__efficient:
                    proj = np.copy(self.__entangledClockState)
                    for j in range(self.__dimension ** self.__nClocks):
                        proj[j,:] *= self.__entangledTickProjectors[i][j]
                        proj[:,j] *= self.__entangledTickProjectorsDagger[i][j]
                    prob = proj.trace()
                else:
                    proj = self.__entangledTickProjectors[i].dot(
                        self.__entangledClockState).dot(self.__entangledTickProjectorsDagger)
                    prob = proj.trace()
            else:
                if self.__projectorMode == 'equal':
                    if self.__veryEfficient:
                        proj = self.__tickProjector * self.__clockStates[i] * self.__tickProjectorDagger
                        prob = np.linalg.norm(proj)
                    elif self.__efficient:
                        proj = np.copy(self.__clockStates[i])
                        for j in range(self.__dimension):
                            proj[j,:] *= self.__tickProjector[j]
                            proj[:,j] *= self.__tickProjectorDagger[j]
                        prob = proj.trace()
                    else:
                        proj = self.__tickProjector.dot(self.__clockStates[i]).dot(
                                self.__tickProjectorDagger)
                        prob = proj.trace()
                else:
                    if self.__veryEfficient:
                        proj = self.__differentTickProjectors[i] * self.__clockStates[i] * self.__differentTickProjectorsDagger[i]
                        prob = np.linalg.norm(proj)
                    elif self.__efficient:
                        proj = np.copy(self.__clockStates[i])
                        for j in range(self.__dimension):
                            proj[j,:] *= self.__differentTickProjectors[i][j]
                            proj[:,j] *= self.__differentTickProjectorsDagger[i][j]
                        prob = proj.trace()
                    else:
                        proj = self.__differentTickProjectors[i].dot(self.__clockStates[i]).dot(
                                self.__differentTickProjectorsDagger[i])
                        prob = proj.trace()


            # simulate a measurement by drawing a random number
            # if the random number is smaller than the probability: tick
            rand = np.random.uniform()
            resetClock = False
            if rand <= prob:
                self.__nTicks[i] += 1
                if self.__nTicks[i] == self.nTicksBeforeSentToReferee:
                    resetClock = True
                    self.__nTicks[i] = 0
                    res.append(i)
                # reset the state (if that's an option)
                if self.__entangled:
                    if self.__resetState is not None and resetClock:
                        # this really does not make any sense and should never occur
                        proj = self.__initialEntangledState
                    else:
                        proj = (1. / prob) * proj
                else:
                    if self.__resetState is not None and resetClock:
                        proj = self.__initialClockStates[self.__resetState]
                    else:
                        proj = (1. / prob) * proj
            else:
                # unsucessful measurement, project the state to the new state
                if self.__entangled:
                    if self.__veryEfficient:
                        proj = self.__entangledNoTickProjectors[i] * self.__entangledClockState * \
                                self.__entangledNoTickProjectorsDagger[i]
                        proj = (1. / np.linalg.norm(proj)) * proj
                    elif self.__efficient:
                        proj = np.copy(self.__entangledClockState)
                        for j in range(self.__dimension ** self.__nClocks):
                            proj[j,:] *= self.__entangledNoTickProjectors[i][j]
                            proj[:,j] *= self.__entangledNoTickProjectorsDagger[i][j]
                        proj = (1. / proj.trace()) * proj
                    else:
                        proj = self.__entangledNoTickProjectors[i].dot(self.__entangledClockState).dot(
                                self.__entangledNoTickProjectorsDagger[i])
                        proj = (1. / proj.trace()) * proj
                else:
                    if self.__projectorMode == 'equal':
                        if self.__veryEfficient:
                            proj = self.__noTickProjector * self.__clockStates[i] *\
                                    self.__noTickProjectorDagger
                            proj = (1. / np.linalg.norm(proj)) * proj
                        elif self.__efficient:
                            proj = np.copy(self.__clockStates[i])
                            for j in range(self.__dimension):
                                proj[j,:] *= self.__noTickProjector[j]
                                proj[:,j] *= self.__noTickProjectorDagger[j]
                            proj = (1. / proj.trace()) * proj
                        else:
                            proj = self.__noTickProjector.dot(self.__clockStates[i]).dot(
                                    self.__noTickProjectorDagger)
                            proj = (1. / proj.trace()) * proj
                    else:
                        if self.__veryEfficient:
                            proj = self.__differentNoTickProjectors[i] * self.__clockStates[i] *\
                                    self.__differentNoTickProjectorsDagger[i]
                            proj = (1. / np.linalg.norm(proj)) * proj
                        elif self.__efficient:
                            proj = np.copy(self.__clockStates[i])
                            for j in range(self.__dimension):
                                proj[j,:] *= self.__differentNoTickProjectors[i][j]
                                proj[:,j] *= self.__differentNoTickProjectorsDagger[i][j]
                            proj = (1. / proj.trace()) * proj
                        else:
                            proj = self.__differentTickProjectors[i].dot(self.__clockStates[i]).dot(
                                    self.__differentNoTickProjectorsDagger[i])
                            proj = (1. / proj.trace()) * proj
            if self.__entangled:
                self.__entangledClockState = proj
            else:
                self.__clockStates[i] = proj
        return res

    def __reset(self):
        if self.__entangled:
            self.__entangledClockState = self.__initialEntangledClockState
        else:
            self.__clockStates = [self.__initialClockStates[i] for i in range(self.__nClocks)]
        self.order = self.__originalOrder

    # public functions
    # ________________

    def setTau(self, tau):
        self.__tau = __tau
        self.__ready = False
        if self.__hamiltonian is not None:
            self.__unitary = expm(-1.j * self.__tau * self.__hamiltonian)
            self.__unitaryDagger = self.__unitary.T.conj()
            if self.__entangled:
                self.__getEntangledHamiltonian()

    def getTau(self):
        return self.__tau

    def setVeryEfficient(self):
        print 'Warning: the very efficient option has been chosen. This is only possible ' + \
                    'if the Peres Hamiltonian is used with theta = tau = 1 and the initial states ' + \
                    'are NOT superpositions in the computational basis. If this is not the case, the ' + \
                    'simulation will lead to wrong results.'
        self.__ready = False
        self.__veryEfficient = True

    def unsetVeryEfficient(self):
        self.__ready = False
        self.__veryEfficient = False

    def setResetState(self, resetState):
        self.__ready = False
        self.__resetState = resetState

    def getDimension(self):
        return self.__dimension

    def getNClocks(self):
        return self.__nClocks

    def isReady(self):
        return self.__ready

    def createPeresHamiltonian(self, dim = None, tP = 1.):
        if dim is None:
            if self.__dimension is None:
                print 'Warning: The simulation is still dimensionless. If you want to create a predefined ' + \
                        'Hamiltonian, you have to specify a dimension. Cowardly refused to perform action.'
                return
            else:
                dim = self.__dimension
        else:
            if not self.__checkDimensionConsistency(np.zeros(dim)):
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
            if self.__dimension is None:
                print 'Warning: The simulation is still dimensionless. If you want to create a predefined ' + \
                        'Hamiltonian, you have to specify a dimension. Cowardly refused to perform action.'
                return
            else:
                dim = self.__dimension
        else:
            if not self.__checkDimensionConsistency(np.zeros(dim)):
                return
        h = np.diag(np.ones(dim-1, dtype = np.complex_),-1) + \
                np.diag(np.ones(dim-1, dtype = np.complex_),1)
        if corners:
            h[0,dim-1] = 1.
            h[dim-1,0] = 1.
        self.addHamiltonian(h)

    def createUniformSuperpositionState(self, dim = None, left = 0, right = 0, phaseFactor = None):
        if dim is None:
            if self.__dimension is None:
                print 'Warning: The simulation is still dimensionless. If you want to create a predefined ' + \
                        'initial state, you have to specify a dimension. Cowardly refused to perform action.'
                return
            else:
                dim = self.__dimension
        else:
            if not self.__checkDimensionConsistency(np.zeros(dim)):
                return
        state = np.zeros(dim, dtype = np.complex_)
        for pos in range(left, right+1):
            state[pos] = 1.
        if phaseFactor is not None:
            for i in range(right+1-left):
                state[left+i] *= phaseFactor ** i 
        state = state / np.linalg.norm(state)
        self.addClockState(state)

    def createGaussianSuperpositionState(self, dim = None, cent = 0, width = 1, phaseFactor = None):
        if dim is None:
            if self.__dimension is None:
                print 'Warning: The simulation is still dimensionless. If you want to create a predefined ' + \
                        'initial state, you have to specify a dimension. Cowardly refused to perform action.'
                return
            else:
                dim = self.__dimension
        else:
            if not self.__checkDimensionConsistency(np.zeros(dim)):
                return
        state = np.zeros(dim, dtype = np.complex_)
        for i in range(width + 1):
            leftPos = cent - i
            rightPos = (cent + i) % dim
            state[leftPos] = stats.norm(0,1).pdf(i * 3. / width)
            state[rightPos] = stats.norm(0,1).pdf(i * 3. / width)
        if phaseFactor is not None:
            for i in range(2*width + 1):
                state[(cent-width+i) % dim] *= phaseFactor**i 
        state = (1. / np.linalg.norm(state)) * state
        self.addClockState(state)

    def createStandardProjectors(self, dim = None, delta = 0.1, d0 = 1, clock = None, loc = None):
        if dim is None:
            if self.__dimension is None:
                print 'Warning: The simulation is still dimensionless. If you want to create a predefined ' + \
                        'initial state, you have to specify a dimension. Cowardly refused to perform action.'
                return
            else:
                dim = self.__dimension
        else:
            if not self.__checkDimensionConsistency(np.zeros(dim)):
                return
        if loc is None:
            loc = dim-1
        loc = loc % dim
        tick = np.zeros(dim, dtype = np.complex_)
        noTick = np.ones(dim, dtype = np.complex_)
        for i in range(d0):
            tick[loc-i] = np.sqrt(delta)
            noTick[loc-i] = np.sqrt(1. - delta)
        self.addProjectors(tick, noTick, clock = clock)

    def addHamiltonian(self, hamiltonian):
        hamiltonianOk, hamiltonian = self.__checkHamiltonian(hamiltonian)
        if hamiltonianOk:
            dimensionsOk = self.__checkDimensionConsistency(hamiltonian)
            if dimensionsOk:
                self.__ready = False
                self.__hamiltonian = hamiltonian
                self.__unitary = expm(-1.j * self.__tau * hamiltonian)
                self.__unitaryDagger = self.__unitary.T.conj()
                if self.__entangled:
                    self.__getEntangledHamiltonian()

    def addClockState(self, clockState):
        clockStateOk, clockState = self.__checkClockState(clockState)
        if clockStateOk:
            dimensionOk = self.__checkDimensionConsistency(clockState)
            if dimensionOk:
                # check if the state is already normalized.
                # if not, normalize it:
                norm = np.linalg.norm(clockState)
                if np.abs(norm - 1) >= 10**-1:
                    print 'Warning: You handed an unnormalized state to the simulation. State has been normalized.'
                    clockState = (1. / norm) * clockState
                # check if the clockworks are already entangled,
                # only add the state if it matches the other ones
                self.__ready = False
                if self.__entangled:
                    if self.__checkForEntanglement(potentialNewState = clockState):
                        self.__clockKets.append(clockState)
                        self.__nClocks += 1
                        # reentangle
                        self.unentangle()
                        self.entangle()
                else:
                    self.__clockKets.append(clockState)
                    self.__nClocks += 1
                    while len(self.__differentTickProjectors) < self.__nClocks:
                        self.__differentTickProjectors.append(None)
                        self.__differentNoTickProjectors.append(None)
                        self.__differentTickProjectorsDagger.append(None)
                        self.__differentNoTickProjectorsDagger.append(None)

    def addProjectors(self, tick, noTick, clock = None):
        if clock is None and self.__projectorMode == 'different':
            output = 'No clock specified, although done previously. For the following clocks, no projectors exist yet: '
            for i in range(self.__nClocks):
                if self.__differentTickProjectors[i] is None:
                    output += str(i) + ' '
            output += '.'
            output += 'Please enter the clock number you would like to use this projector for. (Possible answers: ' + \
                    str(range(self.__nClocks)) + ') If you would like to use ' + \
                    'this projector for all clocks, just type \'all\'. Any other input will cancel the action.'
            print output
            ans = raw_input('> ')
            if ans.isdigit():
                ans = int(ans)
                if ans in range(self.__nClocks):
                    clock = ans 
            else:
                if not ans == 'all':
                    print 'Canceled.'
                    return
                else:
                    self.__differentTickProjectors = [None] * self.__nClocks
                    self.__differentNoTickProjectors = [None] * self.__nClocks
                    self.__differentTickProjectorsDagger = [None] * self.__nClocks
                    self.__differentNoTickProjectorsDagger = [None] * self.__nClocks
        projectorsOk, tick, noTick = self.__checkProjectors(tick, noTick)
        if projectorsOk:
            dimensionsOk = self.__checkDimensionConsistency(tick)
            if dimensionsOk:
                self.__ready = False
                projectorsAlreadyDiagonal = self.__checkForEfficiency(tick)
                tickProj = None
                noTickProj = None
                tickProjDag = None
                noTickProjDag = None
                if self.__efficient and not projectorsAlreadyDiagonal:
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
                    self.__projectorMode = 'equal'
                    self.__tickProjector = tickProj
                    self.__noTickProjector = noTickProj
                    self.__tickProjectorDagger = tickProjDag
                    self.__noTickProjectorDagger = noTickProjDag
                    if self.__entangled:
                        self.__getEntangledProjectors()
                else:
                    if clock >= self.__nClocks:
                        print 'Warning %s clocks added so far (Indexing starting at 0). Starting the simulation now will lead to an error. ' % (self.__nClocks) + \
                                'Cowardly refusing to perform action.'
                        return
                    else:
                        self.__projectorMode = 'different'
                        self.__differentTickProjectors[clock] = tickProj
                        self.__differentNoTickProjectors[clock] = noTickProj
                        self.__differentTickProjectorsDagger[clock] = tickProjDag
                        self.__differentNoTickProjectorsDagger[clock] = noTickProjDag
                    if self.__entangled:
                        if not None in self.__differentTickProjectors:
                            self.__getEntangledProjectors()

    def removeClockState(self, ind = -1):
        if self.__nClocks == 0:
            print 'Nothing to remove...'
            return
        if ind < self.__nClocks:
            self__ready = False
            self.__clockKets.pop(ind)
            self.__differentTickProjectors.pop(ind)
            self.__differentTickProjectorsDagger.pop(ind)
            self.__differentNoTickProjectors.pop(ind)
            self.__differentNoTickProjectorsDagger.pop(ind)
            if self.__nClocks == 0:
                self.__projectorMode = 'equal'
            self.__nClocks -= 1
            # if the clocks are entangled: reentangle them
            if self.__entangled:
                self.unentangle()
                self.entangle()
            if self.__hamiltonian is None and self.__tickProjector is None and self.__nClocks == 0:
                self.__dimension = None

    def clear(self):
        self.__init__(tau = self.__tau, nAverage = self.nAverage,
                mode = self.mode, order = self.order)

    def entangle(self):
        if self.__checkForEntanglement():
            self.__ready = False
            self.__entangled = True
            if self.__hamiltonian is not None:
                self.__getEntangledHamiltonian()
            if self.__tickProjector is not None and self.__projectorMode == 'equal':
                self.__getEntangledProjectors()
            if None in self.__differentTickProjectors and self.__projectorMode == 'different':
                print 'Warning: Projectors have not been specified for all clocks. ' + \
                        'Will entangle the projectors as soon as they are specified for all clocks.'
            elif None not in self.__differentTickProjectors and self.__projectorMode == 'different':
                self.__getEntangledProjectors()
            if len(self.__clockKets) > 0:
                self.__getEntangledClockState()

    def unentangle(self):
        self__ready = False
        self.__entangled = False
        self.__entangledHamiltonian = None
        self.__entangledUnitary = None
        self.__entangledUnitaryDagger = None
        self.__entangledClockState = None
        self.__entangledClockKet = None
        self.__entangledClockMat = None
        self.__entangledTickProjectors = []
        self.__entangledTickProjectorsDagger = []
        self.__entangledNoTickProjectors = []
        self.__entangledNoTickProjectorsDagger = []

    def summary(self):
        output = ''
        output +=  '+++ Summary of this Simulation +++'
        output += '\n'
        output += '\n'
        output += '----------------------------------'
        output += '\n'
        output += '\n'
        output += 'Dimension: ' +  str(self.__dimension)
        output += '\n'
        output += '\n'
        output += 'Number of clocks: %d' % self.__nClocks
        for i in range(self.__nClocks):
            output += '\n'
            output += 'CLOCK %d' % (i+1)
            output += '\n'
            output += str(Qobj(self.__clockKets[i]))
            output += '\n'
            if len(self.__clockStates) > 0:
                output += 'Currently in the following state:'
                output += '\n'
                output += str(Qobj(self.__clockStates[i]))
                output += '\n'
        output += '\n'
        output += 'Hamiltonian specified: ' + str(self.__hamiltonian is not None)
        output += '\n'
        if self.__hamiltonian is not None:
            output += '\n'
            output += 'HAMILTONIAN'
            output += '\n'
            output += str(Qobj(self.__hamiltonian))
            output += '\n'
        output += '\n'
        output += 'Projectors:'
        output += '\n'
        if self.__projectorMode == 'equal':
            if self.__tickProjector is not None:
                output += 'TICK PROJECTOR'
                output += '\n'
                output += str(Qobj(self.__tickProjector))
                output += '\n'
                output += 'NO-TICK PROJECTOR'
                output += '\n'
                output += str(Qobj(self.__noTickProjector))
                output += '\n'
                output += '\n'
            else:
                output += 'not specified'
                output += '\n'
                output += '\n'
        else:
            for i in range(self.__nClocks):
                output += 'CLOCK %s:' % (i)
                output += '\n'
                if self.__differentTickProjectors[i] is not None:
                    output += 'TICK PROJECTOR'
                    output += '\n'
                    output += str(Qobj(self.__differentTickProjectors[i]))
                    output += '\n'
                    output += 'NO-TICK PROJECTOR'
                    output += '\n'
                    output += str(Qobj(self.__differentNoTickProjectors[i]))
                    output += '\n'
                    output += '\n'
                else:
                    output += 'not specified'
                    output += '\n'
                    output += '\n'
        output += 'Entanglement: ' + str(self.__entangled)
        if self.__entangled:
            output += '\n'
            output += '----------------------------------'
            output += '\n'
            output += '\n'
            output += 'ENTANGLED CLOCK STATE'
            output += '\n'
            output += str(Qobj(self.__entangledClockKet))
            output += '\n'
            if self.__entangledClockState is not None:
                output += 'Currently in the following state:'
                output += '\n'
                output += str(Qobj(self.__entangledClockState))
                output += '\n'
            if self.__entangledHamiltonian is not None:
                output += '\n'
                output += 'ENTANGLED HAMILTONIAN'
                output += '\n'
                output += str(Qobj(self.__entangledHamiltonian))
                output += '\n'
            if self.__tickProjector is not None:
                for i in range(self.__nClocks):
                    output += '\n'
                    output += 'ENTANGLED TICK-PROJECTOR %d' % (i + 1)
                    output += '\n'
                    output += str(Qobj(self.__entangledTickProjectors[i]))
                    output += '\n'
                    output += 'ENTANGLED NO-TICK-PROJECTOR %d' % (i + 1)
                    output += '\n'
                    output += str(Qobj(self.__entangledNoTickProjectors[i]))
                    output += '\n'
        return output

    def preview(self, style = '-', nSteps = 10000, clock = 0):
        if self.__hamiltonian is None:
            print 'Please specify a Hamiltonian.'
            return
        if self.__nClocks == 0:
            print 'Please specify at least one state.'
            return
        if self.__entangled:
            print 'Can\'t animate an entangled clock.'
            return
        fig = plt.figure()
        clockKet = np.copy(self.__clockKets[clock])
        frames = []
        x = np.array([i for i in range(self.__dimension)])
        for i in range(nSteps+1):
            frame = None
            if i > 0:
                self.__evolve(animationClock = clock)
            y = np.array([np.abs(self.__clockKets[clock][j])**2 for j in range(self.__dimension)])
            frames.append(plt.plot(x, y, style, color = 'b'))
        ani = animation.ArtistAnimation(fig, frames, interval = 50)
        plt.show(ani)
        self.__clockKets[clock] = clockKet

    def initialize(self):
        # if the simulation has been initialized before, reset it
        if self.__ready:
            self.__reset()
            return

        # check if simulation can be initialized
        if self.__nClocks < 2:
            print 'Can\'t run the simulation yet. Not enough clocks.'
            return
        if self.__hamiltonian is None:
            if not self.__veryEfficient:
                print 'Can\'t run the simulation yet. No Hamiltonian specified.'
                return
        if self.__tickProjector is None and self.__projectorMode == 'equal':
            print 'Can\'t run the simulation yet. No projectors sepecified.'
            return
        if None in self.__differentTickProjectors and self.__projectorMode == 'different':
            print 'Can\'t run the simulation yet. Projectors are not specified for all clocks.'
            return

        if self.__entangled:
            if self.__veryEfficient:
                self.__entangledClockState = self.__entangledClockKet
                self.__initialEntangledClockState = self.__entangledClockKet
            else:
                self.__entangledClockMat = np.outer(self.__entangledClockKet,
                        self.__entangledClockKet)
                self.__entangledClockState = self.__entangledClockMat
                self.__initialEntangledClockState = self.__entangledClockMat
        else:
            if self.__veryEfficient:
                self.__clockStates = [self.__clockKets[i] for i in range(self.__nClocks)]
                self.__initialClockStates = [self.__clockKets[i] for i in range(self.__nClocks)]
            else:
                self.__clockMats = [np.outer(ket, ket) for ket in self.__clockKets]
                self.__clockStates = [self.__clockMats[i] for i in range(self.__nClocks)]
                self.__initialClockStates = [self.__clockMats[i] for i in range(self.__nClocks)]
        if self.label is None:
            self.label = 'dim' + str(self.__dimension)
        if self.mode is None:
            if self.__nClocks == 2:
                self.mode = 'strict'
            else:
                self.mode = 'normal'

        # keep count of the number of ticks produced by each clock
        # this enables us to only send information to the referee once a
        # clock has ticked many times. this reduces the probabilistic effects
        # of quantum mechanics
        self.__nTicks = [0 for _ in range(self.__nClocks)]
        self.__ready = True #initialization successful

    def run(self, maxIterations = 50000):
        for _ in range(maxIterations):
            self.__evolve()
            ticks = self.__measure()
            if len(ticks) > 0:
                return ticks
        return []
