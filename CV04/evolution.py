import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import random
import math

class Plotting:
    def plot(self, generations, lB, uB, fitness):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
        X = np.arange(lB[0], uB[0], (uB[0]-lB[0])/50)
        Y = np.arange(lB[1], uB[1], (uB[1]-lB[1])/50)
        X, Y = np.meshgrid(X, Y)
        #R = np.sqrt(X**2 + Y**2)
        #Z = np.sin(R)
        Z = fitness([X,Y])

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)

        for generation in generations:
            for jedinec in generation:
                ax.scatter(jedinec[0], jedinec[1], jedinec[2], marker='o')
        minZ = np.min(Z)
        maxZ = np.max(Z)
        # Customize the z axis.
        ax.set_zlim(minZ, maxZ)
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

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

    def plot(self):
        Plotting().plot(self.generations, self.lB, self.uB, self.fitness)

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
        self.plot()

    def evolution(self):
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
        self.plot()

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
        self.plot()

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

#rovina.sim_annealing() 
#ackley.sim_annealing() # global min 0 [0;0]
#sphere.sim_annealing() # global min 0 [0;0]
#schwefel.sim_annealing() # global min 0 [420.9;420.9]
#rosenbrock.sim_annealing() # global min 0 [1;1]
#zakharov.sim_annealing() # global min 0 [0;0]
#griewank.sim_annealing() # global min 0 [0;0]
#griewankDetail.sim_annealing() # global min 0 [0;0]
#rastrigin.sim_annealing() # global min 0 [0;0]
levy.evolution() # global min 0 [1;1]
#michalewicz.sim_annealing() # global min -1.8013 [2.2;1.57]