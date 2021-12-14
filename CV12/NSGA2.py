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

    def initPareto(self, title):
        self.title = title
        self.scatter = []

        self.fig, self.ax = plt.subplots()

        self.fig.canvas.manager.set_window_title(self.title)    

        self.ax.set_xlim(self.lB[0], self.uB[0])
        self.ax.set_ylim(self.lB[1], self.uB[1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plotPareto(self, generation, name, plotColour=[], end=False):
        if(end):
            plt.ioff()

        self.ax.set_title(name)

        for sc in self.scatter:
            sc.remove()
        self.scatter = []

        if len(plotColour) == len(generation):
            for index in range(len(generation)):
                self.scatter.append(self.ax.scatter(generation[index][0], generation[index][1], marker='o', color=plotColour[index]))
        else:
            for index in range(len(generation)):
                self.scatter.append(self.ax.scatter(generation[index][0], generation[index][1], marker='o', color='black'))

        if(end):
            plt.show()
        else:
            try:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except:
                return False
        return True

def paretoRank(population, minimize):
    populationSize = len(population)
    S = []
    fitDim = len(population[0])

    for pop in range(populationSize):
        S.append([])

    n = np.zeros(populationSize, np.int32)
    for first in range(populationSize):
        for second in range(populationSize):
            if first == second:
                continue
            
            fitFirst = np.array(population[first])
            fitSecond = np.array(population[second])

            lesser = True
            greater = True
            for index in range(fitDim):
                if minimize and fitFirst[index] < fitSecond[index] or not minimize and fitFirst[index] > fitSecond[index]:
                    greater = False
                elif minimize and fitFirst[index] > fitSecond[index] or not minimize and fitFirst[index] < fitSecond[index]:
                    lesser = False

            if lesser:
                n[first] += 1
            elif greater:
                S[first].append(second)

    Q = np.zeros((populationSize,populationSize),np.int32)

    for index in range(populationSize):
        index2=0
        for value in range(populationSize):
            if n[value] == 0:
                Q[index][index2] = value + 1
                n[value] -= 1
                index2 += 1

        for i in Q[index]:
            if i == 0: break
            for j in S[i - 1]:
                n[j] -= 1
                
    output = []
    for rank in Q:
        if rank[0] == 0: break
        newRank = []
        for index in rank:
            if index == 0: break
            newRank.append(index-1)
        output.append(newRank)
    return output

class Person:
    def __init__(self, lB, uB):
        self.pos = np.zeros(len(lB))
        for dim in range(len(lB)):
            self.pos[dim] = random.uniform(lB[dim], uB[dim])
        self.fitness = []
        self.shown = False

class Solution:
    def __init__(self, lower_bound, upper_bound, fitness):
        self.dim = len(lower_bound)
        self.lB = lower_bound  # we will use the same bounds for all parameters
        self.uB = upper_bound
        self.fitness = fitness
        self.generations = []

    def nsga(self, plotL, plotU, minimize = True):
        plot = Plotting(plotL, plotU, self.fitness)
        plot.initPareto("Non-Dominated Sorting Genetic Algorithm")
        G = 20
        P = 30
        population = []

        #P = 6
        #population2 = [-2,-1,0,2,4,1]

        for pop in range(P):
            newPerson = Person(self.lB,self.uB)
            #newPerson.pos = population2[pop]
            newPerson.fitness = self.fitness(newPerson.pos)
            population.append(newPerson)

        plotPos = []
        plotColours = []

        for g in range(G):
            for firstPerson in range(P):
                firstPerson = secondPerson = population[firstPerson]
                while firstPerson == secondPerson:
                    secondPerson = population[random.choice(range(len(population)))]
                newPerson = Person(self.lB,self.uB)

                # crossover
                if np.random.uniform() < 0.5:
                    newPerson.pos = (firstPerson.pos + secondPerson.pos)/2
                else:
                    newPerson.pos = (firstPerson.pos - secondPerson.pos)/2

                # mutation
                if np.random.uniform() < 0.5:
                    newPerson.pos += np.random.uniform(0,1,self.dim)

                for dim in range(self.dim):
                    if(newPerson.pos[dim]<self.lB[dim]):
                        newPerson.pos[dim]=self.lB[dim]
                    if(newPerson.pos[dim]>self.uB[dim]):
                        newPerson.pos[dim]=self.uB[dim]
                
                newPerson.fitness = self.fitness(newPerson.pos)
                population.append(newPerson)
            
            Q = paretoRank([x.fitness for x in population], minimize=minimize)
            
            oldPopulation = population
            # take best solutions to the new population
            population = []
            added = 0
            for rank in Q:
                for index in rank:
                    if added == P:
                        continue
                    added += 1
                    population.append(oldPopulation[index])

            # data presentation
            for pop in population:
                if pop.shown: continue
                plotPos.append(np.array(pop.fitness))
                plotColours.append('black')
                pop.shown = True

            colours = []
            for pop in range(len(oldPopulation)):
                if pop in Q[0]:
                    colours.append('red')
                else:
                    colours.append('black')
            #plot.plotPareto([x.fitness for x in oldPopulation], "NSGA, generation: " + str(g+1) + ", with " + str(len(oldPopulation)) + " dots", colours)
        
        plotQ = paretoRank(plotPos, minimize=minimize)
        for index in plotQ[0]:
            plotColours[index] = 'red'

        if g != G-1:
            plot.plotPareto(plotPos, "NSGA, Losing population in infinity", plotColours, end = True)
        else:
            plot.plotPareto(plotPos, "NSGA, result with " + str(len(plotPos)) + " dots", plotColours, end = True)
        return 0

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

    def nsga2(params):
        (x1,x2) = params
        return [x1,(1+x2)/x1]

    def nsga1(params):
        (x) = params
        return [-x**2,-(x-2)**2]

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
nsga1 = Solution([-55],[55], Fitness.nsga1)
nsga2 = Solution([0.1,0],[1,5], Fitness.nsga2)

#rovina.nsga() 
#ackley.nsga() # global min 0 [0;0]
#sphere.nsga() # global min 0 [0;0]
#schwefel.nsga() # global min 0 [420.9;420.9]
#rosenbrock.nsga() # global min 0 [1;1]
#zakharov.nsga() # global min 0 [0;0]
#griewank.nsga() # global min 0 [0;0]
#griewankDetail.nsga() # global min 0 [0;0]
#rastrigin.nsga() # global min 0 [0;0]
#levy.nsga() # global min 0 [1;1]
#michalewicz.nsga() # global min -1.8013 [2.2;1.57]
nsga1.nsga([-20,-20],[20,20],True)
nsga1.nsga([-3000,-3000],[200,200],False)
nsga2.nsga([0.1,0],[1,20],True)
nsga2.nsga([0.1,0],[1,20],False)