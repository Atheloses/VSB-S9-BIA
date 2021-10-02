import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import random
import math

class Plotting:
    def plot(self, generations):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
        X = np.arange(-1, 1.01, 0.1)
        Y = np.arange(-1, 1.01, 0.1)
        X, Y = np.meshgrid(X, Y)
        #R = np.sqrt(X**2 + Y**2)
        #Z = np.sin(R)
        Z = X + Y

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)

        for generation in generations:
            for jedinec in generation:
                ax.scatter(jedinec[0], jedinec[1], jedinec[2], marker='o')

        # Customize the z axis.
        ax.set_zlim(-2.01, 2.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

class Solution:
    def __init__(self, dimension, lower_bound, upper_bound, generations, maximize):
        self.dims = dimension
        self.lB = lower_bound  # we will use the same bounds for all parameters
        self.uB = upper_bound
        self.params = np.zeros(self.dims) #solution parameters
        self.f = np.inf  # objective function evaluation
        self.fitness = Fitness()
        self.numberOfGens = generations
        self.generations = []
        self.maximize = maximize

    def plot(self):
        Plotting().plot(self.generations)

    def hill_climbing(self):
        sigma = (min(self.uB)-max(self.lB))/5

        for dim in range(self.dims):
            self.params[dim] = random.uniform(self.lB[dim], self.uB[dim])
        print("Starting position: " + str(self.params))

        lastFitness = self.fitness.hill_climbing(self.params)
        self.generations.append([np.append(self.params,lastFitness),])
        for gen in range(self.numberOfGens):
            tempParams = np.random.normal(self.params,sigma)
            
            for dim in range(self.dims):
                if(tempParams[dim]<self.lB[dim]):
                    tempParams[dim]=self.params[dim]
                if(tempParams[dim]>self.uB[dim]):
                    tempParams[dim]=self.params[dim]

            tempFitness = self.fitness.hill_climbing(tempParams)
            if(self.maximize and tempFitness > lastFitness or not self.maximize and tempFitness < lastFitness):
                print("Generated: " + str(tempParams) + ", fitness: " + str(tempFitness))
                lastFitness = tempFitness
                self.params = tempParams
                self.generations.append([np.append(self.params,lastFitness),])

        print("Final: " + str(self.params) + ", fitness: " + str(self.fitness.hill_climbing(self.params)))
        self.plot()

    def sim_annealing(self):
        sigma = 0.5
        temp = 500
        tempMin = 0.001
        alpha = 0.95

        for dim in range(self.dims):
            self.params[dim] = random.uniform(self.lB[dim], self.uB[dim])
        print("Starting position: " + str(self.params))

        lastFitness = self.fitness.hill_climbing(self.params)
        self.generations.append([np.append(self.params,lastFitness),])
        while temp > tempMin:
            tempParams = np.random.normal(self.params,sigma)
            
            for dim in range(self.dims):
                if(tempParams[dim]<self.lB[dim]):
                    tempParams[dim]=self.params[dim]
                if(tempParams[dim]>self.uB[dim]):
                    tempParams[dim]=self.params[dim]
            
            tempFitness = self.fitness.hill_climbing(tempParams)
            if(tempFitness < lastFitness):
                print("Generated: " + str(tempParams) + ", fitness: " + str(tempFitness) + ", better fitness")
                lastFitness = tempFitness
                self.params = tempParams
                self.generations.append([np.append(self.params,lastFitness),])
            else:
                r = np.random.uniform(0,1)
                annealing = math.e**(-((tempFitness-lastFitness)/temp))
                if( r < annealing):
                    print("Generated: " + str(tempParams) + ", fitness: " + str(tempFitness) + ", annealing with " + str(round(annealing,2)))
                    lastFitness = tempFitness
                    self.params = tempParams
                    self.generations.append([np.append(self.params,lastFitness),])
            temp = temp*alpha
        print("Final: " + str(self.params) + ", fitness: " + str(self.fitness.hill_climbing(self.params)))
        self.plot()
        #e umocněné na delta f lomeno teplota

class Fitness:
    #def __init__(self, name):
    #    self.name = name

    def hill_climbing(self, params):
        sum = 0    
        for p in params:  
            sum += p

        return sum

solution = Solution(2,[-1,-1],[1,1],100,True)
#solution.hill_climbing()
solution.sim_annealing()
