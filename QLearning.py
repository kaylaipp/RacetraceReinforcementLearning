import sys
import numpy as np
import random 
import math
import pandas as pd
import Racecar
from random import randint



class QLearning:
    def __init__(self, fileName, learningRate, discountRate, crashVersion):
        self.fileName = fileName
        self.race = Racecar.Race(fileName)
        self.rows = self.race.racetrack.rows
        self.cols = self.race.racetrack.cols
        self.actions = self.initActions()           # possible (ax,ay) action combinations
        self.states = self.initStates()             # possible (x,y,vx,vy) state combinations
        self.Qtable = self.initQTable()
        self.learningRate = learningRate
        self.discountRate = discountRate
        self.crashVersion = crashVersion

    '''
    return list of possible (ax,ay) action combinations
    '''
    def initActions(self):
        accelerations = [-1, 0, 1]
        possibleActions = []
        for ax in accelerations:
            for ay in accelerations: 
                possibleActions.append(str((ax,ay)))
        return possibleActions


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
                        possibleStates.append(str((x,y,vx,vy)))
        return possibleStates
    
    '''
    Initialze our Q-table to all zeros
    columns = possible actions (ax,ay = 0,1,-1)
    rows = number of states (x = num rows, y = num cols, vx = -5,5, vy = -5,5)
         = (x,y) = (0,0), (0,1), (0,2), (0,4)....
         = (vx,vy) = ()
    '''
    def initQTable(self):
        table = [[np.random.uniform(0,0.1) for col in range(len(self.actions))] for row in range(len(self.states))]
        # using regular strings
        dfTable = pd.DataFrame(table, index = self.states)
        dfTable.columns = self.actions
        
        print('dfTable: ', dfTable)
        return dfTable
    '''
    Update the Q table 
    maxQ(state',action') = max expected future reward given new state s' and all possible actions at this new state
    new Q(state',action') = Q(state,action) + LR(reward for (state',action') + DR*maxQ(state',action') - Q(state,action)))
    '''
    def Q(self, state, newState, action, reward):
        newX, newY, newVx, newVy = eval(newState)
        print('Q table before: ', self.Qtable.loc[state][action])
        self.Qtable.loc[state][action] = ((1 - self.learningRate)*self.Qtable.loc(axis=0)[state][action] +
                self.learningRate*(reward + self.discountRate*self.Qtable.loc(axis=0)[newState].max()))
        print('Q table after: ', self.Qtable.loc[state][action])

    '''
    Get a state randomly
    '''
    def getState(self):
        randomIndex = randint(0, 2)
        return self.states[randomIndex]

    '''
    Car should only choose to accelerate 0.8 of the time
    '''
    def shouldTakeAction(self):
        probability = np.random.uniform(0,1)
        if probability < 0.8:
            return True
        return False

    '''
    Get new velocities based on acc
    Make sure we dont go out of max/min velocities
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
    Validate coordinates in a state
    '''
    def validateCoordinates(self, state):
        x,y,vx,vy = eval(str(state))
        newX,newY = self.validateSingleCoordinates(x,y)
        return (newX, newY, vx, vy)

    '''
    Validate single x,y coordinates, make sure we dont go out of bounds
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
    returns new state based on state,action input
    updates Qtable with reward
    '''
    def takeActionGetReward(self, state, action):
        if self.shouldTakeAction():
            # print('taking action, state, action: ', state, action)
            
            ax,ay = eval(action)
            x,y,vx,vy = eval(state)
            # print('curr location: ', x,y)

            # get new vx,vy velocities based on accelerations ax,ay
            vx_, vy_ = self.getNewVelocity(vx,vy,ax,ay)
            # print('new velocities: ', vx_, vy_)

            # get new x,y positions based on new velocity
            x_ = x + vx_
            y_ = y + vy_
            x_,y_ = self.validateSingleCoordinates(x_,y_)

            # print('new location: ', x_, y_)

            # check if this new position makes the car crash
            carCrashed, linePoints, crashCoordinates = self.race.carCrashed(x,y,x_,y_)
            # print('did car crash: ', carCrashed)
            # print('linePoints: ', linePoints)
            if carCrashed:
                # print('case 1 car crashed!')
                newX, newy, newVx, newVy = self.race.getNewCarStateAfterCrash(linePoints, crashCoordinates, self.crashVersion)
                newState = self.validateCoordinates((newX, newy, newVx, newVy))
                self.Q(state, str(newState), action, reward = -100)
                # print('returning new state: ', (newX, newy, newVx, newVy))
                return newState
            else:
                # print('case 2 car didnt crash, returning new state: ', (x_, y_, vx_, vy_))
                newState = self.validateCoordinates((x_, y_, vx_, vy_))
                self.Q(state, str(newState), action, reward = -1)
                # print('returning new state: ', (x_, y_, vx_, vy_))
                return newState
        else:
            # self.Qtable.loc[state] = 0
            self.Q(state, state, action, reward = 0)
            return self.validateCoordinates(state)

    '''
    Given a state (x,y,vx,vy) return the max reward action from Q table
    Input: state (x,y,vx,vy) 
    Output: max reward action (ax,ay)
    '''
    def getAction(self, state):
        row = self.Qtable.loc[state]
        if random.uniform(0,1) > 0.8: 
            print('using maximal action: ', row.argmax())
            return str(row.argmax())
        else:
            randIdx = random.choice(self.actions)
            print('taking random action: ', randIdx)
            return str(randIdx)

    '''
    return postion on starting line and zero velocity, (x,y,vx,vy)
    '''
    def setInitalState(self):
        x,y = random.choice(self.race.racetrack.startingLine)
        return str((x,y,0,0))

    '''
    Get optimal policy based on Q table
    '''
    def getBestPolicy(self):
        bestPolicy = {}
        for state in self.states:
            bestPolicy[state] = self.Qtable.loc[state].argmax()
        return bestPolicy

    '''
    train q learning algorithm 
    '''
    def train(self, steps):
        # intiialze state to starting line position, with velocity 0

        count = 0
        for step in range(steps):
            currState = self.setInitalState()
            if step % 1000 == 0:
                print('-----------step '+ str(step) + '/' + str(steps) + '-------------')
            # print('initial state: ', currState)
            for s in range(10):
                # base case: car crosses finsish line
                if not self.race.carCrossedFinishLine(currState):
                    self.race.time+=1
                    # print('')
                    # print('---------')
                    # print('time now: ', self.race.time)
                    # print('curr state: ', currState)
                    # choose action from state via epsilon greedy strategy
                    # currAction = max(self.Qtable[currState])
                    currAction = self.getAction(currState)
                    # print('currAction: ', currAction)

                    # take the action & observe reward
                    # get the new state, aka new position & velocity (x,y,vx,vy) 
                    newState = self.takeActionGetReward(currState, currAction)
                    currState = str(newState)
                    # print('newState: ', newState)
                    count += 1
                else:
                    break

        print('training finished! ')
        print('time: ', self.race.time)
        return self.getBestPolicy(), self.race.time


    '''
    Given a policy, output the path we have taken along with time
    '''
    def timeBestPolicy(self, policy):
        currState = self.setInitalState()
        # print('initial state: ', currState)

        finalPath = []
    
        # Keep track if we get stuck
        stop_clock = 0   
        maxSteps = 100
    
        # Begin time trial
        for step in range(maxSteps):        
            # Get the best action given the current state
            bestAction = policy[currState]
    
            # If we are at the finish line, stop the time trial
            if self.race.carCrossedFinishLine(currState): 
                print('found finish line!')
                print('final path: ',finalPath)
                return step
            
            # Take action and get new a new state s'
            newState = self.takeActionGetReward(currState, bestAction)
            currState = str(newState)
    
            # Determine if the car gets stuck
            _,_,vx,vy = eval(currState)
            if vy == 0 and vx == 0:
                stop_clock += 1
            else:
                stop_clock = 0

            finalPath.append((currState, bestAction, newState))
    
            # case if car gets stuck in corner
            if stop_clock == 10:
                # print('car stuck, returning now')
                print('final path: ',finalPath)
                return maxSteps
            
        # Program has timed out
        # print('program timed out, returning now')
        print('final path: ',finalPath)
        return maxSteps



    # run the experiements
    def experiment(self, iterations):            
        policy,timeToBuildPolicy = self.train(iterations)
        time = self.timeBestPolicy(policy)
        print('--------------Evaluation Results-----------------')
        print('track: ', self.fileName)
        print('time to build policy: ', timeToBuildPolicy)
        print('time using best policy: ', time)
        print('iterations: ', iterations)
        print('learning rate: ', self.learningRate)
        print('discount rate: ', self.discountRate)
        print('crash version: ', self.crashVersion)
        return time


# Generating graphs for pdf
# iterationList = [600000,800000,1000000]
# timeListForV1 = []
# timeListForV2 = []
# mapping = {}
# for iteration in iterationList:
#     # crash version 1
#     q = QLearning('L-track.txt', 0.3, 0.9, 'v1')
#     time = q.experiment(iteration)
#     timeListForV1.append(time)
    

#     # crash version 2
#     q.crashVersion = 'v2'
#     time2 = q.experiment(iteration)
#     timeListForV2.append(time2)

# mapping['crash version 1'] = timeListForV1
# mapping['crash version 2'] = timeListForV2
# print('final mapping: ', mapping)
# print('plotting the results...')
# plotMultiple(iterationList, mapping, 'Q-Learning Algorithm on L-track.txt')



# demo purposes 
# train for 1 iteration, crash version 1
# q = QLearning('L-track.txt', 0.5, 0.8, 'v1')
# q.experiment(1)

# train for 1 iteration, crash version 2
# q.crashVersion = 'v2'
# q.experiment(1)