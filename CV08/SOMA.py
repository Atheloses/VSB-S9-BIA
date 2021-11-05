import matplotlib.pyplot as plt
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

    def initHeatMap(self, title):
        self.title = title
        self.scatter = []
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

    def plotHeatMap(self, generation, name, end=False):
        if(end):
            plt.ioff()

        self.fig.canvas.set_window_title(name)        

        for sc in self.scatter:
            sc.remove()
        self.scatter = []

        for jedinec in generation:
            self.scatter.append(self.ax.scatter(jedinec[0], jedinec[1], marker='o'))

        if(end):
            plt.show()
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

class Particle:
    def __init__(self, lB, uB, Vmin, Vmax):
        self.pos = np.zeros(len(lB))
        for dim in range(len(lB)):
            self.pos[dim] = random.uniform(lB[dim], uB[dim])

        self.pBest = np.copy(self.pos)
        self.pBestValue = 0

        self.vel = np.zeros(len(lB))
        for dim in range(len(lB)):
            self.vel[dim] = random.uniform(Vmin[dim], Vmax[dim])

class Solution:
    def __init__(self, lower_bound, upper_bound, maximize, fitness):
        self.dims = len(lower_bound)
        self.lB = lower_bound  # we will use the same bounds for all parameters
        self.uB = upper_bound
        self.fitness = fitness
        self.generations = []

    def soma(self):
        pop_size = 20
        pertProb = 0.4
        pathLength = 3.0
        step = 0.11
        M_max = 100
        # perturbace přepočítat při každém skoku
        # perturbace vektor 1 nebo 0
        pass

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
michalewicz.soma() # global min -1.8013 [2.2;1.57]