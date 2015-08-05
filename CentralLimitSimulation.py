from __future__ import print_function
import numpy as np
from sys import stdout

# parameters
maxOutputLength = 200
dimension = 200000
prob = 0.1

# calculated values
expectationSingle = 1/prob
varianceSingle = 1/prob**2 + 1/prob

p = - (25*varianceSingle + 4*dimension*expectationSingle)/(4*expectationSingle**2)
q = dimension**2/(4*expectationSingle**2)

nTicks = int(-p/2. - np.sqrt((p/2.)**2 - q)) + 1

position0 = nTicks*expectationSingle - 1
position1 = 0

overlapStartingAt = nTicks*expectationSingle

nTicks0 = 0
nTicks1 = 0

ticksString = ''

stop = False
consoleOutput = True
while True:
    position0 += 1
    position1 += 1

    if consoleOutput:
        string = ticksString 
        if len(ticksString) > maxOutputLength:
            string = '...' + ticksString[-(maxOutputLength-3):]
        end = '\r'
        if stop:
            end = '\n'
        print('[' + string + '-'*(maxOutputLength-len(string)) + '] -- Total: ' + str(len(ticksString)), end = end)
        stdout.flush()

        consoleOutput = False

    if stop:
        break
    
    if position0 >= overlapStartingAt:
        rand = np.random.uniform()
        if rand <= prob:
            nTicks0 += 1
    if position1 >= overlapStartingAt:
        rand = np.random.uniform()
        if rand <= prob:
            nTicks1 += 1

    if nTicks0 == nTicks:
        nTicks0 = 0
        position0 = 0
        ticksString += str(0)
        consoleOutput = True
    if nTicks1 == nTicks:
        nTicks1 = 0
        position1 = 0
        ticksString += str(1)
        consoleOutput = True

    if len(ticksString) >= 2:
        if ticksString[-2] == ticksString[-1]:
            stop = True

print('Finished. Total: ' + str(len(ticksString)))
