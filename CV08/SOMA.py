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

        self.fig.canvas.manager.set_window_title(self.title)      
        c = self.ax.pcolormesh(self.X, self.Y, self.Z, vmin=self.z_min, vmax=self.z_max, shading='auto')
        self.ax.axis([self.lB[0], self.uB[0], self.lB[1], self.uB[1]])
        self.fig.colorbar(c, ax=self.ax)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plotHeatMap(self, generation, name, end = False, leader = -1):
        if(end):
            plt.ioff()

        self.ax.set_title(name)       

        for sc in self.scatter:
            sc.remove()
        self.scatter = []

        for index in range(len(generation)):
            if(index != leader):
                self.scatter.append(self.ax.scatter(generation[index][0], generation[index][1], marker='o', color='black'))

        if(leader>=0):
            self.scatter.append(self.ax.scatter(generation[leader][0], generation[leader][1], marker='o', color='red'))

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

class Solution:
    def __init__(self, lower_bound, upper_bound, fitness):
        self.dims = len(lower_bound)
        self.lB = lower_bound  # we will use the same bounds for all parameters
        self.uB = upper_bound
        self.fitness = fitness
        self.generations = []

    def soma(self):
        plot = Plotting(self.lB, self.uB, self.fitness)
        plot.initHeatMap("Self-Organizing Migrating Algorithm")
        pop_size = 20
        pertProb = 0.4
        pathLength = 3.0
        stepLength = 0.22
        M_max = 10

        leader = "empty"
        newGenPlot = [] 
        swarm = []
        for pop in range(pop_size):
            newParticle = Particle(self.lB,self.uB)
            newParticle.value = self.fitness(newParticle.pos)
            newGenPlot.append(newParticle.pos)
            swarm.append(newParticle)
            if(leader == "empty" or newParticle.value < leader.value):
                leader = newParticle

        plot.plotHeatMap(newGenPlot, "SOMA: Initialization")

        
        for m in range(M_max):
            for particle in swarm:
                if(particle == leader):
                    continue

                stepToLeader = []
                for dim in range(self.dims):
                    stepToLeader.append(leader.pos[dim] - particle.pos[dim])
                stepToLeader = np.array(stepToLeader) * stepLength

                bestPos = np.copy(particle.pos)
                bestValue = particle.value
                for step in range(math.ceil(pathLength/stepLength)):
                    perturbation = []
                    for dim in range(self.dims):
                        if(random.random() < pertProb):
                            perturbation.append(1)
                        else:
                            perturbation.append(0)
                        
                    particle.pos += stepToLeader * np.array(perturbation)

                    for pos in range(len(particle.pos)): # check for position boundaries
                        if(particle.pos[pos] < self.lB[pos]):
                            particle.pos[pos] = self.lB[pos]
                        if(particle.pos[pos] > self.uB[pos]):
                            particle.pos[pos] = self.uB[pos]

                    particle.value = self.fitness(particle.pos)
                    if(particle.value < bestValue):
                        bestValue = particle.value
                        bestPos = np.copy(particle.pos)

                particle.pos = bestPos
                particle.value = bestValue

            newGenPlot = [] 
            leader = swarm[0]
            for particle in swarm:
                newGenPlot.append(particle.pos)
                if(particle.value < leader.value):
                    leader = particle
                
            plot.plotHeatMap(newGenPlot, "SOMA, generation: " + str(m+1) + ", best: " + '{:8.4f}'.format(leader.value), leader = swarm.index(leader))
        plot.plotHeatMap(newGenPlot, "SOMA, result, best: " + '{:8.4f}'.format(leader.value), end = True, leader = swarm.index(leader))

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

#rovina.sim_annealing() 
#ackley.soma() # global min 0 [0;0]
#sphere.soma() # global min 0 [0;0]
#schwefel.soma() # global min 0 [420.9;420.9]
#rosenbrock.soma() # global min 0 [1;1]
#zakharov.soma() # global min 0 [0;0]
#griewank.soma() # global min 0 [0;0]
#griewankDetail.soma() # global min 0 [0;0]
rastrigin.soma() # global min 0 [0;0]
#levy.soma() # global min 0 [1;1]
#michalewicz.soma() # global min -1.8013 [2.2;1.57]