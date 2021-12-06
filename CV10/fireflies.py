import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import random
import math

class Plotting:
    def __init__(self, lB, uB, fitness):
        self.lB = lB
        self.uB = uB
        plt.ion()
        self.figure = 0
        self.fitness = fitness
        self.scatter = []
        
    def initHeatMap(self, title):
        self.title = title
        self.scatter = []
        self.X, self.Y = np.meshgrid(np.linspace(self.lB[0], self.uB[0], 100), np.linspace(self.lB[1], self.uB[1], 100))
        self.Z = self.fitness([self.X,self.Y])
        self.z_min, self.z_max = self.Z.min(), self.Z.max()

        self.fig, self.ax = plt.subplots()

        self.fig.canvas.manager.set_window_title(self.title)      
        c = self.ax.pcolormesh(self.X, self.Y, self.Z, vmin=self.z_min, vmax=self.z_max, shading='auto')
        self.ax.axis([self.lB[0], self.uB[0], self.lB[1], self.uB[1]])
        self.fig.colorbar(c, ax=self.ax)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plotHeatMap(self, generation, name, end=False, leaderIndex = -1):
        if(end):
            plt.ioff()

        self.ax.set_title(name)       

        for sc in self.scatter:
            sc.remove()
        self.scatter = []

        for index in range(len(generation)):
            if(leaderIndex != index):
                self.scatter.append(self.ax.scatter(generation[index][0], generation[index][1], marker='o', color='black'))

        if(leaderIndex >= 0):
            self.scatter.append(self.ax.scatter(generation[leaderIndex][0], generation[leaderIndex][1], marker='o', color='red'))

        if(end):
            plt.show()
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

class Solution:
    def __init__(self, lower_bound, upper_bound, fitness):
        self.dims = len(lower_bound)
        self.lB = lower_bound  # we will use the same bounds for all parameters
        self.uB = upper_bound
        self.params = [] #solution parameters
        self.f = np.inf  # objective function evaluation
        self.fitness = fitness
        self.generations = []

    def fireflies(self):
        redrawGen = 10
        numberOfGens = 150
        populationSize = 20
        lastFitness = []
        absorption = 1
        attractivness = 1

        plot = Plotting(self.lB, self.uB, self.fitness)
        plot.initHeatMap("Firefly Algorithm")

        leaderIndex = 0
        leaderFitness = 100000000

        self.params = np.zeros((populationSize,self.dims))
        for i in range(populationSize): # generate random locations
            for dim in range(self.dims):
                self.params[i][dim] = random.uniform(self.lB[dim], self.uB[dim])
            lastFitness.append(self.fitness(self.params[i]))
            if(lastFitness[i] < leaderFitness):
                leaderFitness = lastFitness[i]
                leaderBackup = np.copy(self.params[i])
                leaderIndex = i
            
        for gen in range(numberOfGens):
            for i in range(populationSize):
                if(i == leaderIndex):
                    leftFireflies = [*range(0,leaderIndex),*range(leaderIndex+1,populationSize)]
                    randomFirefly = np.random.choice(leftFireflies,1)[0]
                else:
                    randomFirefly = -1

                for j in range(populationSize):
                    if(randomFirefly >= 0): # Leader choses randomly
                        j = randomFirefly
                    else:
                        if(i == j): continue

                    # Calculate light intensity for both fireflies
                    distance_ij = np.linalg.norm(self.params[i]-self.params[j])
                    light_i = (-lastFitness[i])*math.e**(-absorption*distance_ij)
                    light_j = (-lastFitness[j])*math.e**(-absorption*distance_ij)
                    
                    if(light_i < light_j or randomFirefly >= 0):
                        alpha = np.zeros(len(self.lB))
                        epsilon = np.zeros(len(self.lB))
                        for dim in range(self.dims):
                            alpha[dim] = random.uniform(0, (self.uB[dim]-self.lB[dim])/20)
                            epsilon[dim] = np.random.normal(0, 1)

                        self.params[i] += attractivness*math.e**(-absorption*distance_ij**2)*(self.params[j]-self.params[i]) 
                        self.params[i] += alpha*epsilon
                        
                        # Check boundaries
                        for dim in range(self.dims):
                            if(self.params[i][dim]<self.lB[dim]):
                                self.params[i][dim]=self.lB[dim]
                            if(self.params[i][dim]>self.uB[dim]):
                                self.params[i][dim]=self.uB[dim]

                        lastFitness[i] = self.fitness(self.params[i])
                    
                    if(randomFirefly >= 0): # Leader moves only when fitness is better
                        if(lastFitness[i] > leaderFitness):
                            lastFitness[i] = leaderFitness
                            self.params[i] = np.copy(leaderBackup)
                        break
            
            for firefly in range(populationSize):
                if(lastFitness[firefly] < lastFitness[leaderIndex]):
                    leaderIndex = firefly

            leaderFitness = lastFitness[leaderIndex]
            leaderBackup = np.copy(self.params[leaderIndex])

            #print("Best: "+ '{:8.4f}'.format(lastFitness[leaderIndex]) + ", index: " + str(leaderIndex))
            if gen % redrawGen == 0:
                plot.plotHeatMap(self.params, "Fireflies, generation: " + str(gen+1) + ", best: " + '{:8.4f}'.format(lastFitness[leaderIndex]), False, leaderIndex)
            
        for firefly in range(populationSize):
            print("Final: " + str(self.params[firefly]) + ", fitness: " + str(lastFitness[firefly]))
            self.generations.append([np.append(self.params[firefly],lastFitness[firefly]),])
        plot.plotHeatMap(self.params, "Fireflies, last generation, best: " + '{:8.4f}'.format(lastFitness[leaderIndex]), True, leaderIndex)

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

#rovina.fireflies() 
#ackley.fireflies() # global min 0 [0;0]
#sphere.fireflies() # global min 0 [0;0]
#schwefel.fireflies() # global min 0 [420.9;420.9]
#rosenbrock.fireflies() # global min 0 [1;1]
#zakharov.fireflies() # global min 0 [0;0]
#griewank.fireflies() # global min 0 [0;0]
#griewankDetail.fireflies() # global min 0 [0;0]
rastrigin.fireflies() # global min 0 [0;0]
#levy.fireflies() # global min 0 [1;1]
#michalewicz.fireflies() # global min -1.8013 [2.2;1.57]