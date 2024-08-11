import sys
print(sys.version_info)
import numpy as np
import random 
import math
import pandas as pd
import Racecar
from random import randint

class ValueIteration:
    def __init__(self, fileName, learningRate, discountRate, crashVersion):
        self.race = Racecar.Race(fileName)
        self.rows = self.race.racetrack.rows
        self.cols = self.race.racetrack.cols
        self.actions = self.initActions()           # possible (ax,ay) action combinations
        self.states = self.initStates()             # possible (x,y,vx,vy) state combinations
        self.Vtable = self.initVTable()
        self.Qtable = self.initQTable()
        self.learningRate = learningRate
        self.discountRate = discountRate
        self.crashVersion = crashVersion
        self.fileName = fileName

    '''
    return list of possible (ax,ay) action combinations
    '''
    def initActions(self):
        accelerations = [-1, 0, 1]
        possibleActions = []
        for ax in accelerations:
            for ay in accelerations: 
                possibleActions.append((ax,ay))
        return possibleActions

    '''
    Get new velocity given vx,vy pair
    '''
    def getNewVelocity(self, vx,vy,ax,ay):
        newX = vx + ax   
        newY = vy + ay
        if newX < self.race.car.minV: newX = self.race.car.minV
        if newX > self.race.car.maxV: newX = self.race.car.maxV
        if newY < self.race.car.minV: newY = self.race.car.minV
        if newY > self.race.car.maxV: newY = self.race.car.maxV
        return newX, newY

    '''
    Car should only choose to accelerate 0.8 of the time
    '''
    def shouldTakeAction(self):
        probability = np.random.uniform(0,1)
        if probability < 0.8:
            return True
        return False

    '''
    Validate x,y coordinates
    Make sure we do not move outside of the track
    '''
    def validateSingleCoordinates(self, x, y):
        newX,newY = x,y
        if x < 0:
            newX = 0
        elif x >= self.race.racetrack.rows:
            newX = self.race.racetrack.rows-1
        if y < 0: 
            newY = 0
        elif y >= self.race.racetrack.cols:
            newY = self.race.racetrack.cols-1
        # if out of bounds, get closest on track point
        if x != newX or y != newY:
            return self.race.getClosestTrackPoint((newX,newY))
        return (x,y)

    '''
    return list of possible (x,y,vx,vy) state combinations
    '''
    def initStates(self):
        minVelocity, maxVelocity = -5, 5
        minX, maxX = 0, self.rows
        minY, maxY = 0, self.cols
        possibleStates = []

        for x in range(minX, maxX):
            for y in range(minY, maxY):
                for vx in range(minVelocity, maxVelocity+1):
                    for vy in range(minVelocity, maxVelocity+1):
                        possibleStates.append((x,y,vx,vy))
        return possibleStates

    def Q(self, state, action, expectedValue, reward):
        self.Qtable.loc[state][action] = reward + (self.discountRate * expectedValue)

    '''
    Initialze our V(s) table to hold 
    this will hold optimal Q value for each possible state S
    used to help us detemrine when to stop algorithm
    '''
    def initVTable(self):
        table = [random.uniform(0,0.1) for _ in self.states]
        dfTable = pd.DataFrame(table, index = pd.MultiIndex.from_tuples(self.states))
        print('v table: ', dfTable)
        return dfTable

    def initQTable(self):
        table = [[np.random.uniform(0,0.1) for col in range(len(self.actions))] for row in range(len(self.states))]
        dfTable = pd.DataFrame(table, index = pd.MultiIndex.from_tuples(self.states))
        dfTable.columns = pd.MultiIndex.from_tuples(self.actions)
        print('dfTable: ', dfTable)
        return dfTable

    '''
    Given a state (x,y,vx,vy) return the max reward action from Q table
    Input: state (x,y,vx,vy) 
    Output: max reward action (ax,ay)
    '''
    def getAction(self, state):
        row = self.Qtable.loc[state]
        if random.uniform(0,1) > 0.8: 
            print('using maximal action: ', row.argmax())
            return row.argmax()
        else:
            randIdx = random.choice(self.actions)
            print('taking random action: ', randIdx)
            return randIdx

    def validateCoordinates(self, state):
        x,y,vx,vy = eval(str(state))
        newX,newY = self.validateSingleCoordinates(x,y)
        return (newX, newY, vx, vy)

    '''
    returns new state based on state,action input
    updates Qtable with reward
    '''
    def takeActionGetReward(self, state, action):
        if self.shouldTakeAction():            
            ax,ay = action
            x,y,vx,vy = state

            # get new vx,vy velocities based on accelerations ax,ay
            vx_, vy_ = self.getNewVelocity(vx,vy,ax,ay)

            # get new x,y positions based on new velocity
            x_ = x + vx_
            y_ = y + vy_
            x_,y_ = self.validateSingleCoordinates(x_,y_)

            # check if this new position makes the car crash
            carCrashed, linePoints, crashCoordinates = self.race.carCrashed(x,y,x_,y_)

            if carCrashed:
                newX, newy, newVx, newVy = self.race.getNewCarStateAfterCrash(linePoints, crashCoordinates, self.crashVersion)
                newState = self.validateCoordinates((newX, newy, newVx, newVy))
                return newState
            else:
                newState = self.validateCoordinates((x_, y_, vx_, vy_))
                return newState
        else:
            return self.validateCoordinates(state)

    '''
    Get optimal policy based on Q table
    '''
    def getBestPolicy(self):
        bestPolicy = {}
        for state in self.states:
            bestPolicy[state] = self.Qtable.loc[state].argmax()
        return bestPolicy

    '''
    Set our intial state to any coordinate on starting line, with zero vel
    '''
    def setInitalState(self):
        x,y = random.choice(self.race.racetrack.startingLine)
        return (x,y,0,0)
    '''
    train q learning algorithm 
    '''
    def train(self, steps):
        prevVTable = self.Vtable.copy()
        print('prevVTable: ', prevVTable)
        steps=1
        for step in range(steps):
            self.race.time+=1
            if step % 1000 == 0:
                print('-----------step '+ str(step) + '/' + str(steps) + '-------------')
            for currState in self.states:
                x,y,vx,vy = currState
                currAction = self.getAction(currState)

                for action in self.actions:
                    if self.race.carCrossedFinishLine(str(currState)):
                        reward = 1
                    else:
                        reward = -1

                    # take the action and get the reward
                    newState = self.takeActionGetReward(currState, currAction)
                    valueNewState = prevVTable.loc[newState]

                    # calculate new state if action is no movevemnt
                    stateNoMovement = self.takeActionGetReward(currState, (0,0))

                    # Value for state failure, aka state of not moving 
                    valueNewStateFailureToMove = prevVTable.loc[stateNoMovement]

                    # transition function / expected value of possible values
                    expectedValue = (0.8 * valueNewState) + (0.2 * (valueNewStateFailureToMove))

                    # update Q table
                    self.Q(currState, action, expectedValue, reward)
                
                # Get the action with the highest Q value
                actionWithHighestQ = np.argmax(self.Qtable.loc[currState])

                print('Value before: ', self.Vtable.loc[currState])
                # update value table
                self.Vtable.loc[currState] = self.Qtable.loc[currState][actionWithHighestQ]
                print('Value after: ', self.Vtable.loc[currState])


        print('training finished! ')
        print('time: ', self.race.time)
        return self.getBestPolicy(), self.race.time

    '''
    Given a policy, output the path we have taken along with time
    '''
    def timeBestPolicy(self, policy):
        finalPath = []
        currState = self.setInitalState()
    
        # Keep track if we get stuck
        stop_clock = 0   
        maxSteps = 100
    
        # Begin time trial
        for step in range(maxSteps):   
 
            # Get the best action given the current state
            bestAction = policy[currState]
            
            # If we are at the finish line, stop the time trial
            if self.race.carCrossedFinishLine(str(currState)): 
                print('crossed finish line')
                print('finalPath after finishing policy: ', finalPath)
                return step
            
            # Take action and get new a new state s'
            newState = self.takeActionGetReward(currState, bestAction)

            # add to our path
            finalPath.append((currState, bestAction, newState))

            currState = newState
    
            # Determine if the car gets stuck
            _,_,vx,vy = currState
            if vy == 0 and vx == 0:
                stop_clock += 1
            else:
                stop_clock = 0
    
            # We have gotten stuck as the car has not been moving for 5 timesteps
            if stop_clock == 5:
                print('car stuck, returning now')
                print('finalPath: ', finalPath)
                return maxSteps
            
        # Program has timed out
        print('done going through policy')
        print('finalPath: ', finalPath)
        return maxSteps, finalPath

    # run the experiements
    def experiment(self, iterations):            
        policy,timeToBuildPolicy = self.train(iterations)
        time = self.timeBestPolicy(policy)
        print('-------------- Evaluation Results-----------------')
        print('track: ', self.fileName)
        print('time to build policy: ', timeToBuildPolicy)
        print('time using best policy: ', time)
        print('iterations: ', iterations)
        print('learning rate: ', self.learningRate)
        print('discount rate: ', self.discountRate)
        print('crash version: ', self.crashVersion)
        return time
                    

# generating graphs for pdf write up
# iterationList = [1,5,5]
# timeListForV1 = []
# timeListForV2 = []
# mapping = {}
# for iteration in iterationList:
#     # crash version 1
#     valueIteration = ValueIteration('L-track.txt', 0.3, 0.9, 'v1')
#     time = valueIteration.experiment(iteration)
#     timeListForV1.append(time)
    

#     # crash version 2
#     valueIteration.crashVersion = 'v2'
#     time2 = valueIteration.experiment(iteration)
#     timeListForV2.append(time2)

# mapping['crash version 1'] = timeListForV1
# mapping['crash version 2'] = timeListForV2
# print('final mapping: ', mapping)
# print('plotting the results...')
# plotMultiple(iterationList, mapping, 'Value Iteration Algorithm on L-track.txt')


# demo purposes 
# train for 1 iteration, crash version 1
# valueIteration = ValueIteration('sampleTrack.txt', 0.3, 0.9, 'v1')
# time = valueIteration.experiment(1)

# demo purposes 
# train for 1 iteration, crash version 2
# valueIteration.crashVersion = 'v2'
# valueIteration.experiment(1)