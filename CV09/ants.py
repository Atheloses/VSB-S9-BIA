import matplotlib.pyplot as plt
import numpy as np
import random
import math
from matplotlib.path import Path
import matplotlib.patches as patches
import sys

class Plotting:
    def __init__(self, lB, uB, fitness):
        self.lB = lB
        self.uB = uB
        plt.ion()
        self.figure = 0
        self.fitness = fitness

    def init2D(self, title):
        self.title = title
        self.patches = []

        self.fig, self.ax = plt.subplots()

        self.fig.canvas.manager.set_window_title(self.title)    

        self.ax.set_xlim(self.lB[0], self.uB[0])
        self.ax.set_ylim(self.lB[1], self.uB[1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot2D(self, path, cities, name, end=False):
        if(end):
            plt.ioff()
        
        for patch in self.patches:
            patch.remove()
        self.patches = []

        verts = []
        codes = [Path.MOVETO,]
        for city in range(len(cities)):
            verts.append(tuple(cities[path[city]]))
            codes.append(Path.LINETO)
        verts.append(tuple(cities[path[0]]))

        self.ax.set_title(name)

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=1)
        self.ax.add_patch(patch)
        self.patches.append(patch)

        self.ax.set_xlim(self.lB[0], self.uB[0])
        self.ax.set_ylim(self.lB[1], self.uB[1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if(end):
            plt.show()
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

class Particle:
    def __init__(self, lB, uB):
        self.pos = np.zeros(len(lB))
        for dim in range(len(lB)):
            self.pos[dim] = random.uniform(lB[dim], uB[dim])
        self.value = 0

def TSPInit(lB,uB,citiesCount):
    output = []
    dims = len(lB)
    for city in range(citiesCount):
        newCity = []
        for dim in range(dims):
            newCity.append(random.uniform(lB[dim], uB[dim]))
        output.append(newCity)
    return output

def pathLength(path, distances):
    pathLength = 0
    for i in range(len(path)-1):
        pathLength += distances[path[i]][path[i+1]]
    pathLength += distances[path[len(path)-1]][path[0]]

    return pathLength

class Solution:
    def __init__(self, lower_bound, upper_bound, maximize, fitness="empty"):
        self.dims = len(lower_bound)
        self.lB = lower_bound  # we will use the same bounds for all parameters
        self.uB = upper_bound
        self.fitness = fitness
        self.generations = []

    def ants(self, G, D, antsCount):
        plot = Plotting(self.lB, self.uB, self.fitness)
        plot.init2D("Ant Colony Optimization")

        matrixVisited = np.zeros((antsCount,D,D))
        matrixVisibility = np.zeros((D,D))
        startPheromon = 1
        evaporation = 0.5
        pheromones = np.zeros((D, D)) + startPheromon
        antsPath = np.zeros((antsCount,D),np.int32)
        antsPathValues = np.zeros(antsCount)
        cities = TSPInit(self.lB,self.uB,D)
        alpha = 1
        beta = 1
        Q = 1
        matrixDistance = [
            [0, 10, 12, 11, 14],
            [10, 0, 13, 15, 8],
            [12, 13, 0, 9, 14],
            [11, 15, 9, 0, 16],
            [14, 8, 14, 16, 0]
        ]
        matrixDistance = np.zeros((D,D))

        globalMinValue = sys.maxsize
        globalMinPath = []

        for x in range(antsCount):
            antsPath[x][0] = x % D # starting city

        for y in range(D):
            for x in range(D): # distance between cities
                matrixDistance[x][y] = math.dist(cities[x],cities[y])
                if(matrixDistance[x][y]):
                    matrixVisibility[x][y] = 1/matrixDistance[x][y]

        for gen in range(G):
            for antIndex in range(antsCount): # going over each ant
                matrixVisibilityCopy = np.copy(matrixVisibility) # backup original values
                for antPos in range(D-1): # looking for path from first city
                    antPosFrom = antsPath[antIndex][antPos] # first city is pre-set
                    # rule out posibility of chosing this city again
                    matrixVisibilityCopy[:, antPosFrom] = 0

                    # calculating posible paths based on distance
                    posibilities = pheromones[antPosFrom]**alpha * matrixVisibilityCopy[antPosFrom]**beta
                    probabilities = posibilities/np.sum(posibilities)

                    # weighted probability
                    antPosTo = np.random.choice(range(D), 1, p=probabilities)[0]

                    # set the next city
                    antsPath[antIndex][antPos+1] = antPosTo 
                    matrixVisited[antIndex][antPosFrom][antPosTo] = 1
                
                # finish the circle
                antPosFrom = antsPath[antIndex][0]
                matrixVisited[antIndex][antPosTo][antPosFrom] = 1
            
            # calculating path's distance value
            for antIndex in range(antsCount):
                antsPathValues[antIndex] = pathLength(antsPath[antIndex],matrixDistance)

            # most visited paths
            usedPaths = np.zeros((D,D))
            for ant in range(antsCount):
                usedPaths += matrixVisited[ant]*(1/antsPathValues[ant])
            # evaporation
            pheromones = (1-evaporation)*pheromones + usedPaths
            matrixVisited = np.zeros((antsCount,D,D))     
            
            # presenting data
            min_index = np.argmin(antsPathValues)
            plot.plot2D(antsPath[min_index],cities,str(gen+1)+": "+'{:8.4f}'.format(antsPathValues[min_index]))
            print('{:3.0f}'.format(gen+1)+": "+'{:8.4f}'.format(antsPathValues[min_index]))
            if(globalMinValue>antsPathValues[min_index]):
                globalMinPath = np.copy(antsPath[min_index])
                globalMinValue = antsPathValues[min_index]
        
        plot.plot2D(globalMinPath,cities,"best: "+'{:8.4f}'.format(globalMinValue), end=True)
        print("best: "+'{:8.4f}'.format(globalMinValue))


tsp = Solution([0,0],[10,10], False)
tsp.ants(50, 20, 10)