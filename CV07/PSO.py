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

    def pso(self):
        plot = Plotting(self.lB, self.uB, self.fitness)
        plot.initHeatMap("Particle Swarm Optimization")
        pop_size = 15
        M_max = 50
        c1 = 2.0 
        c2 = 2.0
        Vmini = []
        Vmaxi = []
        ws = 0.9
        we = 0.4
        gBest = np.zeros(self.lB)
        gBestValue = 100000000

        for dim in range(self.dims):
            newMax = (self.uB[dim]-self.lB[dim])/20
            Vmaxi.append(newMax)
            Vmini.append(-newMax)

        newGen = [] 
        swarm = []
        for pop in range(pop_size):
            swarm.append(Particle(self.lB,self.uB,Vmini,Vmaxi))
            swarm[pop].pBestValue = self.fitness(swarm[pop].pos)
            newGen.append(swarm[pop].pos)
            if(swarm[pop].pBestValue < gBestValue):
                gBest = np.copy(swarm[pop].pos)
                gBestValue = swarm[pop].pBestValue 

        plot.plotHeatMap(newGen, "Generation: 0")

        for m in range(M_max):
            newGen = [] 
            for particle in swarm:
                r1 = np.random.uniform()
                w = ws - (ws-we)*m/M_max # inertia
                newVel = particle.vel*w + r1*c1*(particle.pBest-particle.pos) + r1*c2*(gBest-particle.pos) # new velocity

                for vel in range(len(newVel)): # check for velocity boundaries
                    if(newVel[vel] < Vmini[vel]):
                        newVel[vel] = Vmini[vel]
                    if(newVel[vel] > Vmaxi[vel]):
                        newVel[vel] = Vmaxi[vel]

                particle.vel = newVel
                newPos = particle.pos + particle.vel

                for pos in range(len(particle.pos)): # check for position boundaries
                    if(newPos[pos] < self.lB[pos]):
                        newPos[pos] = self.lB[pos]
                    if(newPos[pos] > self.uB[pos]):
                        newPos[pos] = self.uB[pos]
                particle.pos = newPos

                pValue = self.fitness(particle.pos)
                if(pValue < particle.pBestValue):
                    particle.pBestValue = pValue
                    particle.pBest = particle.pos
                    if(particle.pBestValue < gBestValue):
                        gBest = np.copy(particle.pos)
                        gBestValue = particle.pBestValue
                newGen.append(particle.pos)
                
            plot.plotHeatMap(newGen, "Generation: " + str(m+1))
        plot.plotHeatMap(newGen, "Result", end = True)

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
#ackley.pso() # global min 0 [0;0]
#sphere.pso() # global min 0 [0;0]
#schwefel.pso() # global min 0 [420.9;420.9]
#rosenbrock.pso() # global min 0 [1;1]
#zakharov.pso() # global min 0 [0;0]
#griewank.pso() # global min 0 [0;0]
#griewankDetail.pso() # global min 0 [0;0]
#rastrigin.pso() # global min 0 [0;0]
#levy.pso() # global min 0 [1;1]
michalewicz.pso() # global min -1.8013 [2.2;1.57]