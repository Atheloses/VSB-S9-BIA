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

    def initHeatMap(self, title):
        self.title = title
        self.X, self.Y = np.meshgrid(np.linspace(self.lB[0], self.uB[0], 100), np.linspace(self.lB[1], self.uB[1], 100))
        self.Z = self.fitness([self.X,self.Y])
        self.z_min, self.z_max = self.Z.min(), self.Z.max()

        self.fig, self.ax = plt.subplots()

        self.ax.set_title(self.title)
        c = self.ax.pcolormesh(self.X, self.Y, self.Z, vmin=self.z_min, vmax=self.z_max)
        self.ax.axis([self.lB[0], self.uB[0], self.lB[1], self.uB[1]])
        self.fig.colorbar(c, ax=self.ax)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def init3D(self):
        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})
        self.ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        self.ax.zaxis.set_major_formatter('{x:.02f}')

        # Make data.
        X = np.arange(self.lB[0], self.uB[0], (self.uB[0]-self.lB[0])/50)
        Y = np.arange(self.lB[1], self.uB[1], (self.uB[1]-self.lB[1])/50)
        self.X, self.Y = np.meshgrid(X, Y)
        #R = np.sqrt(X**2 + Y**2)
        #Z = np.sin(R)
        self.Z = self.fitness([self.X,self.Y])

        # Plot the surface.
        surf = self.ax.plot_surface(self.X, self.Y, self.Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)
        # Add a color bar which maps values to colors.
        self.fig.colorbar(surf, shrink=0.5, aspect=5)
        minZ = np.min(self.Z)
        maxZ = np.max(self.Z)
        # Customize the z axis.
        self.ax.set_zlim(minZ, maxZ)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot3D(self, generations, name):
        self.ax.clear()
        self.fig.canvas.set_window_title(name)
        self.ax.plot_surface(self.X, self.Y, self.Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)

        for generation in generations:
            for jedinec in generation:
                self.ax.scatter(jedinec[0], jedinec[1], jedinec[2], marker='o')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plotHeatMap(self, generations, name, end=False):
        if(end):
            plt.ioff()

        self.ax.clear()
        self.fig.canvas.set_window_title(name)        
        self.ax.set_title(self.title)
        c = self.ax.pcolormesh(self.X, self.Y, self.Z,  vmin=self.z_min, vmax=self.z_max)
        self.ax.axis([self.lB[0], self.uB[0], self.lB[1], self.uB[1]])

        for generation in generations:
            for jedinec in generation:
                self.ax.scatter(jedinec[0], jedinec[1], marker='o')

        if(end):
            plt.show()
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def plot2D(self, cities, name):
        verts = []
        codes = [Path.MOVETO,]
        for city in range(len(cities)):
            verts.append(tuple(cities[city]))
            codes.append(Path.LINETO)
        verts.append(tuple(cities[0]))

        self.ax.clear()
        self.fig.canvas.set_window_title(name)

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=1)
        self.ax.add_patch(patch)

        #xs, ys = zip(*verts)
        #self.ax.plot(xs, ys, 'x--', lw=2, color='black', ms=10)

        #for city in range(len(cities)):
        #    self.ax.text(cities[city][0], cities[city][1], city)

        self.ax.set_xlim(self.lB[0], self.uB[0])
        self.ax.set_ylim(self.lB[1], self.uB[1])
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

def pathLength(cities):
    pathLength = 0
    for i in range(len(cities)-1):
        pathLength += math.dist(cities[i],cities[i+1])
    pathLength += math.dist(cities[len(cities)-1],cities[0])

    return pathLength

class Solution:
    def __init__(self, lower_bound, upper_bound, maximize, fitness):
        self.dims = len(lower_bound)
        self.lB = lower_bound  # we will use the same bounds for all parameters
        self.uB = upper_bound
        self.params = np.zeros(self.dims) #solution parameters
        self.f = np.inf  # objective function evaluation
        self.fitness = fitness
        self.generations = []
        self.maximize = maximize

    def hill_climbing(self):
        sigma = (min(self.uB)-max(self.lB))/5
        numberOfGens = 100

        for dim in range(self.dims):
            self.params[dim] = random.uniform(self.lB[dim], self.uB[dim])
        print("Starting position: " + str(self.params))

        lastFitness = self.fitness.hill_climbing(self.params)
        self.generations.append([np.append(self.params,lastFitness),])
        for gen in range(numberOfGens):
            tempParams = np.random.normal(self.params,sigma)
            
            for dim in range(self.dims):
                if(tempParams[dim]<self.lB[dim]):
                    tempParams[dim]=self.params[dim]
                if(tempParams[dim]>self.uB[dim]):
                    tempParams[dim]=self.params[dim]

            tempFitness = self.fitness(tempParams)
            if(self.maximize and tempFitness > lastFitness or not self.maximize and tempFitness < lastFitness):
                print("Generated: " + str(tempParams) + ", fitness: " + str(tempFitness))
                lastFitness = tempFitness
                self.params = tempParams
                self.generations.append([np.append(self.params,lastFitness),])

        print("Final: " + str(self.params) + ", fitness: " + str(self.fitness(self.params)))

        plot = Plotting(self.lB, self.uB, False, self.fitness)
        plot.plot(self.generations)

    def evolution(self):
        plot = Plotting(self.lB, self.uB, False, self.fitness)
        params = []
        sigma = 0.5
        numberOfGens = 100
        populationSize = 20
        lastFitness = []
        konstanta = 0.817

        for i in range(populationSize):
            location = []
            for dim in range(self.dims):
                location.append(random.uniform(self.lB[dim], self.uB[dim]))
            params.append(location)
            lastFitness.append(self.fitness(location))
            #self.generations.append([np.append(location,self.fitness(location)),])

        for gen in range(numberOfGens):
            better = 0
            for jedinec in range(populationSize):
                tempParams = np.random.normal(params[jedinec], sigma)
                for dim in range(self.dims):
                    if(tempParams[dim]<self.lB[dim]):
                        tempParams[dim]=params[jedinec][dim]
                    if(tempParams[dim]>self.uB[dim]):
                        tempParams[dim]=params[jedinec][dim]

                tempFitness = self.fitness(tempParams)
                if(self.maximize and tempFitness > lastFitness[jedinec] or not self.maximize and tempFitness < lastFitness[jedinec]):
                    print("Generated: " + str(tempParams) + ", fitness: " + str(tempFitness))
                    lastFitness[jedinec] = tempFitness
                    params[jedinec] = tempParams
                    better = better + 1
                    #self.generations.append([np.append(tempParams,tempFitness),])

            if(better < populationSize/5):
                sigma = konstanta*sigma
            elif(better > populationSize/5):
                sigma = sigma/konstanta
            
        for jedinec in range(populationSize):
            print("Final: " + str(params[jedinec]) + ", fitness: " + str(lastFitness[jedinec]))
            self.generations.append([np.append(params[jedinec],lastFitness[jedinec]),])
        
        plot.plot(self.generations)

    def sim_annealing(self):
        sigma = (min(self.uB)-max(self.lB))/5
        temp = 500
        tempMin = 0.01
        alpha = 0.97

        for dim in range(self.dims):
            self.params[dim] = random.uniform(self.lB[dim], self.uB[dim])
        print("Starting position: " + str(self.params))

        lastFitness = self.fitness(self.params)
        self.generations.append([np.append(self.params,lastFitness),])
        nth = 0
        while temp > tempMin:
            tempParams = np.random.normal(self.params,sigma)
            
            for dim in range(self.dims):
                if(tempParams[dim]<self.lB[dim]):
                    tempParams[dim]=self.params[dim]
                if(tempParams[dim]>self.uB[dim]):
                    tempParams[dim]=self.params[dim]
            
            tempFitness = self.fitness(tempParams)
            if(self.maximize and tempFitness > lastFitness or not self.maximize and tempFitness < lastFitness):
                print('{:4.0f}'.format(nth) + ": Generated: [" + '{:8.4f}'.format(tempParams[0]) + "; " + '{:8.4f}'.format(tempParams[1]) + "], fitness: " + '{:8.4f}'.format(tempFitness) + ", better fitness")
                lastFitness = tempFitness
                self.params = tempParams
                self.generations.append([np.append(self.params,lastFitness),])
            else:
                r = np.random.uniform(0,1)
                if(self.maximize):
                    annealing = math.e**(-((lastFitness-tempFitness)/temp))
                else:
                    annealing = math.e**(-((tempFitness-lastFitness)/temp))
                if( r < annealing):
                    print('{:4.0f}'.format(nth) + ": Generated: [" + '{:8.4f}'.format(tempParams[0]) + "; " + '{:8.4f}'.format(tempParams[1]) + "], fitness: " + '{:8.4f}'.format(tempFitness) + ", annealing with " + str(round(annealing,2)))
                    lastFitness = tempFitness
                    self.params = tempParams
                    self.generations.append([np.append(self.params,lastFitness),])
            temp = temp*alpha
            nth+=1
        print('{:4.0f}'.format(nth) + ": Final: " + str(self.params) + ", fitness: " + str(self.fitness(self.params)))
        
        plot = Plotting(self.lB, self.uB, False, self.fitness)
        plot.plot(self.generations)

    def ga(self):
        NP = 20
        G = 200
        D = 20  # In TSP, it will be a number of cities

        plot = Plotting(self.lB, self.uB, True, self.fitness)
        population = []
        pathLengths = []
        jedinec = TSPInit(self.lB, self.uB, D)

        for j in range(NP):
            np.random.shuffle(jedinec)
            population.append(np.copy(jedinec))
            pathLengths.append(pathLength(population[j]))

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
                
                
                newPath = pathLength(offspring_AB)
                if(newPath < pathLengths[firstParent]):
                    pathLengths[firstParent] = newPath
                    new_population[firstParent] = offspring_AB

            population = new_population

            min_index = pathLengths.index(min(pathLengths))
            plot.plot2D(population[min_index],str(gen)+": "+str(pathLengths[min_index]))
            print('{:3.0f}'.format(gen)+": "+'{:8.4f}'.format(pathLengths[min_index]))
            #time.sleep(0.2)

    def de(self):
        plot = Plotting(self.lB, self.uB, self.fitness)
        plot.initHeatMap("Differential Evolution")
        NP = 10
        G = 20
        # D = 20  # In TSP, it will be a number of cities
        F = 0.5
        CR = 0.5

        #plot = Plotting(self.lB, self.uB, True)
        fitnessResults = []
        population = []
        jedinec = np.zeros(self.dims)

        for j in range(NP):
            for dim in range(self.dims):
                jedinec[dim] = random.uniform(self.lB[dim], self.uB[dim])
            population.append(np.copy(jedinec))
            fitnessResults.append(self.fitness(jedinec))

        for jedinec in range(NP):
            print("Starting: " + str(population[jedinec]) + ", fitness: " + str(fitnessResults[jedinec]))

        for gen in range(G):
            new_population = np.copy(population)  # Offspring is always put to a new population

            for firstParent in range(NP):
                parent_A = population[firstParent]
                restParents = np.random.choice([*range(0,firstParent),*range(firstParent+1,NP)], size=3)
                parent_B = population[restParents[0]]
                parent_C = population[restParents[1]]
                parent_D = population[restParents[2]]

                mutation = np.zeros(self.dims)
                # mutation vector
                for dim in range(self.dims):
                    mutation[dim] = (parent_B[dim] - parent_C[dim]) * F + parent_D[dim]

                    # boundaries
                    if(mutation[dim] < self.lB[dim] or mutation[dim] > self.uB[dim]):
                        mutation[dim] = parent_A[dim]

                randomInt = np.random.randint(0, self.dims)
                offspring = np.zeros(self.dims)
                # combine mutation with parent_A
                for dim in range(self.dims):
                    if(np.random.uniform() < CR or dim == randomInt):
                        offspring[dim] = mutation[dim]
                    else:
                        offspring[dim] = parent_A[dim]
                
                newFitness = self.fitness(offspring)
                if(newFitness <= fitnessResults[firstParent]):
                    fitnessResults[firstParent] = newFitness
                    new_population[firstParent] = offspring

            population = new_population
            newGen = []
            for jedinec in range(NP):
                newGen.append([np.append(population[jedinec],fitnessResults[jedinec]),])
            self.generations.append(newGen)
            plot.plotHeatMap(newGen, "Generation: " + str(gen+1))

        for jedinec in range(NP):
            print("Final: " + str(population[jedinec]) + ", fitness: " + str(fitnessResults[jedinec]))
        
        plot.plotHeatMap(newGen, "Result", end = True)

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

rovina = Solution([-1,-1],[1,1], False, Fitness.rovina)
ackley = Solution([-32.768,-32.768],[32.768,32.768], False, Fitness.ackley)
sphere = Solution([-5.12,-5.12],[5.12,5.12], False, Fitness.sphere)
schwefel = Solution([-500,-500],[500,500], False, Fitness.schwefel)
rosenbrock = Solution([-10,-10],[10,10], False, Fitness.rosenbrock)
zakharov = Solution([-10,-10],[10,10], False, Fitness.zakharov)
griewank = Solution([-600,-600],[600,600], False, Fitness.griewank)
griewankDetail = Solution([-5,-5],[5,5], False, Fitness.griewank)
rastrigin = Solution([-5.12,-5.12],[5.12,5.12], False, Fitness.rastrigin)
levy = Solution([-10,-10],[10,10], False, Fitness.levy)
michalewicz = Solution([0,0],[math.pi,math.pi], False, Fitness.michalewicz)
ag_tsp = Solution([0,0],[10,10], False, Fitness.dummy)

#rovina.sim_annealing() 
#ackley.de() # global min 0 [0;0]
#sphere.de() # global min 0 [0;0]
#schwefel.de() # global min 0 [420.9;420.9]
#rosenbrock.de() # global min 0 [1;1]
#zakharov.de() # global min 0 [0;0]
#griewank.de() # global min 0 [0;0]
#griewankDetail.de() # global min 0 [0;0]
#rastrigin.de() # global min 0 [0;0]
#levy.de() # global min 0 [1;1]
michalewicz.de() # global min -1.8013 [2.2;1.57]