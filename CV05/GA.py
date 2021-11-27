import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import random
import math
import time

from numpy.lib.function_base import append

class Plotting:
    def __init__(self, lB, uB, fitness):
        self.lB = lB
        self.uB = uB
        plt.ion()
        self.figure = 0
        self.fitness = fitness

    def initPath(self, title):
        self.title = title
        self.patches = []

        self.fig, self.ax = plt.subplots()

        self.fig.canvas.manager.set_window_title(self.title)    

        self.ax.set_xlim(self.lB[0], self.uB[0])
        self.ax.set_ylim(self.lB[1], self.uB[1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plotPath(self, path, cities, name, end=False):
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
    def __init__(self, lower_bound, upper_bound, fitness):
        self.dims = len(lower_bound)
        self.lB = lower_bound  # we will use the same bounds for all parameters
        self.uB = upper_bound
        self.params = np.zeros(self.dims) #solution parameters
        self.f = np.inf  # objective function evaluation
        self.fitness = fitness
        self.generations = []

    def ga(self):
        plot = Plotting(self.lB, self.uB, self.fitness)
        plot.initPath("Generic Algorithm")

        NP = 20
        G = 100
        D = 10  # In TSP, it will be a number of cities

        population = []
        pathLengths = []
        cities = TSPInit(self.lB,self.uB,D)

        matrixDistance = np.zeros((D,D))
        for y in range(D):
            for x in range(D): # distance between cities
                matrixDistance[x][y] = math.dist(cities[x],cities[y])

        for j in range(NP):
            randomPath = [*range(len(cities))]
            np.random.shuffle(randomPath)
            population.append(randomPath)
            pathLengths.append(pathLength(population[j],matrixDistance))

        for gen in range(G):
            new_population = np.copy(population)  # Offspring is always put to a new population

            for firstParent in range(NP):
                parent_A = population[firstParent]
                secondParent = np.random.choice([*range(0,firstParent),*range(firstParent+1,NP)])
                parent_B = population[secondParent]

                offspring_AB = np.copy(parent_A)
                split = np.random.randint(0,D)
                found = offspring_AB[:split]

                appnd = []
                for parentB in parent_B:
                    if parentB not in found:
                        appnd.append(parentB)

                for i in range(D-split):
                    offspring_AB[split+i] = appnd[i]

                if np.random.uniform() < 0.5:
                    firstMutate = np.random.randint(0,D)
                    left = [*range(0,firstMutate),*range(firstMutate+1,D)]
                    secondMutate = np.random.choice(left,1)

                    temp = np.copy(offspring_AB[firstMutate])
                    offspring_AB[firstMutate] = offspring_AB[secondMutate]
                    offspring_AB[secondMutate] = temp
                
                
                newPath = pathLength(offspring_AB, matrixDistance)
                if(newPath < pathLengths[firstParent]):
                    pathLengths[firstParent] = newPath
                    new_population[firstParent] = offspring_AB

            population = new_population

            min_index = pathLengths.index(min(pathLengths))
            # plot.plotPath(population[min_index],cities,"GA, generation: " + str(gen)+", best: "+'{:8.4f}'.format(pathLengths[min_index]))
            # print('{:3.0f}'.format(gen)+": "+'{:8.4f}'.format(pathLengths[min_index]))
        plot.plotPath(population[min_index],cities,"GA, result, best: "+'{:8.4f}'.format(pathLengths[min_index]), True)

class Fitness:
    def rovina(params):
        sum = 0    
        for p in params:  
            sum += p

        return sum

    def sphere(params):
        sum1=0
        for p in params:
            sum1+=p**2
        return sum1

    def ackley(params):
        a = 20
        b = 0.2
        c = 2*math.pi
        sum1=0
        sum2=0
        for p in params:
            sum1+=p**2
            sum2+=np.cos(c*p)
        return -a*np.exp(-b*np.sqrt(sum1))-np.exp(1/len(params)*sum2)+a+np.exp(1)

    def schwefel(params):
        sum1=0
        for p in params:
            sum1+=p*np.sin(np.sqrt(np.abs(p)))
        return 418.9829*len(params)-sum1

    def rosenbrock(params):
        (x1,x2) = params
        return 100*(x2-x1**2)**2+(x1-1)**2

    def zakharov(params):
        (x1,x2) = params
        a = 0.5*x1+x2
        b = x1*x1+x2*x2 + pow(a,2) + pow(a,4)
        return b

    def griewank(params):
        (x1,x2) = params
        return ((x1**2)/4000 + (x2**2)/4000) - (np.cos(x1/np.sqrt(1)) * np.cos(x2/np.sqrt(2)))
        
    def rastrigin(params):
        (x1,x2) = params
        return 10*2 + (x1**2-10*np.cos(2*math.pi*x1) + x2**2-10*np.cos(2*math.pi*x2))

    def levy(params):
        (x1,x2) = params
        w1 = 1+(x1-1)/4
        w2 = 1+(x2-1)/4
        return np.sin(math.pi*w1)**2 + (w1-1)**2 * (1+10*np.sin(math.pi*w1+1)**2) + (w2-1)**2 * (1+np.sin(2*math.pi*w2)**2)

    def michalewicz(params):
        (x1,x2) = params
        m = 10
        return -(np.sin(x1)*np.sin( 1*x1**2 /math.pi)**(2*m) + np.sin(x2)*np.sin( 2*x2**2 /math.pi)**(2*m))
    
    def dummy(params):
        pass

rovina = Solution([-1,-1],[1,1], Fitness.rovina)
ackley = Solution([-32.768,-32.768],[32.768,32.768], Fitness.ackley)
sphere = Solution([-5.12,-5.12],[5.12,5.12], Fitness.sphere)
schwefel = Solution([-500,-500],[500,500], Fitness.schwefel)
rosenbrock = Solution([-10,-10],[10,10], Fitness.rosenbrock)
zakharov = Solution([-10,-10],[10,10], Fitness.zakharov)
griewank = Solution([-600,-600],[600,600], Fitness.griewank)
griewankDetail = Solution([-5,-5],[5,5], Fitness.griewank)
rastrigin = Solution([-5.12,-5.12],[5.12,5.12], Fitness.rastrigin)
levy = Solution([-10,-10],[10,10], Fitness.levy)
michalewicz = Solution([0,0],[math.pi,math.pi], Fitness.michalewicz)
ag_tsp = Solution([0,0],[10,10], Fitness.dummy)

ag_tsp.ga()
time.sleep(5)

pass