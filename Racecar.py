# coding: utf-8
import numpy as np
import random 
import math
# -*- coding: utf-8 -*-
class RaceTrack: 
    def __init__(self, fileName):
        self.track = self.buildTrack(fileName)
        self.getStartAndFinishLines()
        self.t = 0

    def buildTrack(self, fileName): 
        # read in file
        with open(fileName) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines] 
        self.rows = int(lines[0].split(',')[0])
        self.cols = int(lines[0].split(',')[1])
        # print('rows, cols: ', self.rows, self.cols)
        # print('returning: ', lines[1:])
        return lines[1:]


    def getStartAndFinishLines(self): 
        self.startingLine, self.finshLine, self.walls = [],[],[]
        for x in range(self.rows):
            for y in range(self.cols):
                if self.track[x][y] == 'S':
                    self.startingLine.append((x,y))
                elif self.track[x][y] == 'F':
                    self.finshLine.append((x,y))
                elif self.track[x][y] == '#':
                    self.walls.append((x,y))
        # print('staringLine: ', self.startingLine)
        # print('finishLine; ', self.finshLine)


# raceTrack = RaceTrack('L-track.txt')

class RaceCar: 
    def __init__(self):
        # coordinates of car at time t
        self.x = 0
        self.y = 0
        # velocity of car at time t (limted to ranges between -5,5)
        self.vx = 0
        self.vy = 0
        # acceration of car at time t (car only has control of acceleration vars = 0,1,-1)
        self.ax = 0
        self.ay = 0
        self.minV = -5
        self.maxV = 5

    '''
    probability of accelerating = 80%
    probabiity of not accelerating = 20%

    As an example, if at time t = 0 your car is at location (2, 2) with velocity (1, 0), it is essentially moving
    towards the east. If you apply an acceleration of (1, 1), then at timestep t = 1 your position will be (4, 3)
    and your velocity will be (2, 1). At each time step, your acceleration is applied to your velocity before your
    position is updated.

    t = 0
    (x,y) = (2,2)
    (x_v, y_v) = (1,0)
    (a_x, a_y) = (1,1)

    t = 1
    (x,y) = (4,3) = x + (x_v)*(a_x), y + (y_v)(a_y) = (2 + 1*1, 2 + 0*1) = 
    (x_v, y_v) = intital v + accerlation*time = (x_v, y_v) + (a_x, a_y) = (1,0) + (1,1) = (2,1)
    '''
    def move(self, ax, ay):
        probability = np.random.uniform(0,1)
        shouldAccelerate = True if probability > 0.2 else False



class Race: 
    def __init__(self, fileName): 
        # initialze the track
        self.racetrack = RaceTrack(fileName)

        # initialize the car 
        self.car = RaceCar()

        self.time = 0

    def getRandomState(self):

        pass


    def carCrossedFinishLine(self, state):
        x,y,_,_ = eval(state)
        # print('in carCrossedFinishLine, x,y: ', (x,y))
        # print('self.racetrack.finshLine: ', self.racetrack.finshLine)
        if (x,y) in self.racetrack.finshLine:
            # print('CROSSED FINSIH LINE')
            return True
        # print('didnot cross finsh line')
        return False

    '''
    Input: (x,y) (newX, newY) line cooridnates
    Output: list of [(x,y)] coordinates that fall on this line
    '''
    # def getPointsOnLine(self, x, y, newX, newY):
    #     dx = newX - x
    #     dy = newY - y
    #     D = 2*dy - dx
    #     linePoints = []
    #     print('x, newX: ', x, newX)
    #     for x in range(x, newX):
    #         linePoints.append((x,y))
    #         if D > 0:
    #             y += 1
    #             D = D - 2*dx
    #         D = D + 2*dy
    #     return linePoints
    def getPointsOnLine(self, x0, y0, x1, y1):
        # print('in getPoints on line: ', (x0, y0), (x1,y1))
        if abs(y1 - y0) < abs(x1 - x0):
            if x0 > x1:
                return self.plotLineLow(x1, y1, x0, y0)
            else:
                return self.plotLineLow(x0, y0, x1, y1)
        else:
            if y0 > y1:
                return self.plotLineHigh(x1, y1, x0, y0)
            else:
                return self.plotLineHigh(x0, y0, x1, y1)
    
    def plotLineHigh(self,x0, y0, x1, y1):
        res = [(x0,y0)]
        dx = x1 - x0
        dy = y1 - y0
        xi = 1
        if dx < 0:
            xi = -1
            dx = -dx
        D = (2 * dx) - dy
        x = x0

        for y in range(y0, y1):
            res.append((x,y))
            if D > 0:
                x = x + xi
                D = D + (2 * (dx - dy))
            else:
                D = D + 2*dx
        return res

    def plotLineLow(self, x0, y0, x1, y1):
        res = [(x0,y0)]
        dx = x1 - x0
        dy = y1 - y0
        yi = 1
        if dy < 0:
            yi = -1
            dy = -dy
        D = (2 * dy) - dx
        y = y0

        for x in range(x0, x1):
            res.append((x,y))
            if D > 0:
                y = y + yi
                D = D + (2 * (dy - dx))
            else:
                D = D + 2*dy
        return res

    def outOfBounds(self, x,y): 
        if x < 0:
            return True
            x = 0
        elif x >= self.racetrack.rows:
            return True
            x = self.racetrack.rows-1
        if y < 0: 
            return True
            y = 0
        elif y >= self.racetrack.cols:
            return True
            y = self.racetrack.cols-1
        return False


    def lineContainsWall(self, linePoints):
        # return any(point in self.racetrack.walls for point in linePoints)
        for point in linePoints:
            x,y = point
            if self.outOfBounds(x,y) or self.racetrack.track[x][y] == '#':
                return True, (x,y)
        return False, None
    '''
    Returns T/F if car has crashed
        * v1: if crashed, place car back to nearest place on track where car crashed, set vel = (0,0)
        * v2: if crashed, place car back to starting postiion, with vel = (0,0)

    Uses Bresenham’s Algorithm to see if car has intercepted with boundary of track
    '''
    def carCrashed(self, x, y, newX, newY):
        linePoints = self.getPointsOnLine(x,y,newX,newY)
        # print('linePoints in carCrashed: ', linePoints)
        carCrashed, crashCoordinates = self.lineContainsWall(linePoints)
        if carCrashed:
            # print('CRASH DETECTED, getting nearest on track point')
            return True, linePoints, crashCoordinates
        return False, linePoints, crashCoordinates


    def getNewCarStateAfterCrash(self, linePoints, crashCoordinates, version = 'v1'):
        # get closest track coordinates to where this car crashed
        print('car crashed!')
        print('line points: ', linePoints)
        print('crashCoordinates: ', crashCoordinates)
        x_,y_ = self.getClosestTrackPoint(crashCoordinates)
        print('closest on track point: ', x_,y_)
        if version == 'v1':
            print('v1 crash, setting car back to closest on track point, state now = ', (x_, y_, 0, 0))
            return (x_, y_, 0, 0)
        else: 
            startX, startY = random.choice(self.racetrack.startingLine)
            print('v2 crash, setting car back to starting line, state now = ', (startX, startX, 0, 0))
            return (startX, startX, 0, 0)

    '''
    return (x,y) coordinates of closest track points to where car crashed initially
    use euclidean distance measure & BFS
    '''
    def getClosestTrackPoint(self, crashCoordiantes):
        x,y = crashCoordiantes
        queue = [(x,y)]
        visited = set()
        directions = [(1,0), (0,1), (-1,0), (0,-1)]
        # print('getting closest track point')
        while queue:
            curr = queue.pop(0)
            currx, curry = curr
            # print('self.racetrack.track[currx][curry]: ', self.racetrack.track[currx][curry])
            
            if self.racetrack.track[currx][curry] == '.': 
                # print('returning nearest on track point!!!!!!: ', curr)
                return curr
            if curr not in visited:
                visited.add((currx, curry))
                for dx,dy in directions: 
                    # print('currx+dx, curry+dy: ', currx+dx, curry+dy)
                    if 0 <= currx+dx < self.racetrack.rows and 0 <= curry+dy < self.racetrack.cols:
                        queue.append((currx+dx, curry+dy))
                        # print('queue now: ', queue)
                        
        

            


        

