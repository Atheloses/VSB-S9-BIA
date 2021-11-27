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

    def init3D(self, title):
        self.title = title

        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})

        X = np.arange(self.lB[0], self.uB[0], (self.uB[0]-self.lB[0])/50)
        Y = np.arange(self.lB[1], self.uB[1], (self.uB[1]-self.lB[1])/50)
        X, Y = np.meshgrid(X, Y)
        Z = self.fitness([X,Y])

        # Plot the surface.
        surf = self.ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)

        self.fig.colorbar(surf, shrink=0.5, aspect=5)

        self.fig.canvas.manager.set_window_title(self.title)    

        self.ax.set_xlim(self.lB[0], self.uB[0])
        self.ax.set_ylim(self.lB[1], self.uB[1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot3D(self, params, fitnessValues, name, end=False):
        if(end):
            plt.ioff()

        self.ax.set_title(name)       

        for sc in self.scatter:
            sc.remove()
        self.scatter = []

        self.scatter.append(self.ax.scatter(params[0], params[1], fitnessValues, marker='o', color='black'))

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
        self.params = np.zeros(self.dims) #solution parameters
        self.f = np.inf  # objective function evaluation
        self.fitness = fitness
        self.generations = []

    def hill_climbing(self):
        plot = Plotting(self.lB, self.uB, self.fitness)
        plot.init3D("Hill Climbing")

        generations = 20
        sigma = (min(self.uB)-max(self.lB))/5

        for dim in range(self.dims):
            self.params[dim] = random.uniform(self.lB[dim], self.uB[dim])
        #print("Starting position: " + str(self.params))

        lastFitness = self.fitness(self.params)
        self.generations.append([np.append(self.params,lastFitness),])
        for gen in range(generations):
            tempParams = np.random.normal(self.params,sigma)
            
            for dim in range(self.dims):
                if(tempParams[dim]<self.lB[dim]):
                    tempParams[dim]=self.params[dim]
                if(tempParams[dim]>self.uB[dim]):
                    tempParams[dim]=self.params[dim]

            tempFitness = self.fitness(tempParams)
            if(tempFitness < lastFitness):
                #print("Generated: " + str(tempParams) + ", fitness: " + str(tempFitness))
                lastFitness = tempFitness
                self.params = tempParams
                self.generations.append([np.append(self.params,lastFitness),])

            plot.plot3D(self.params, lastFitness, "HC, value: " + '{:8.4f}'.format(lastFitness), False)

        print("Final: " + str(self.params) + ", fitness: " + str(self.fitness(self.params)))
        plot.plot3D(self.params, lastFitness, "HC, result: " + '{:8.4f}'.format(self.fitness(self.params)), True)

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

rovina.hill_climbing() 
#ackley.hill_climbing() # global min 0 [0;0]
#sphere.hill_climbing() # global min 0 [0;0]
#schwefel.hill_climbing() # global min 0 [420.9;420.9]
#rosenbrock.hill_climbing() # global min 0 [1;1]
#zakharov.hill_climbing() # global min 0 [0;0]
#griewank.hill_climbing() # global min 0 [0;0]
#griewankDetail.hill_climbing() # global min 0 [0;0]
#rastrigin.hill_climbing() # global min 0 [0;0]
#levy.hill_climbing() # global min 0 [1;1]
#michalewicz.hill_climbing() # global min -1.8013 [2.2;1.57]
