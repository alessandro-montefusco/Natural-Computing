#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

import operator
import math
import random
import numpy
import matplotlib.pyplot as plt
import multiprocessing

'''
 Il dataset ha x sulla prima colonna, y sulla seconda, il risultato di f(xy) sulla terza.
 Lo scopo è quello di scoprire la relazione fra il glucosio interstiziale (a livello della pelle) e il glucosio sanguigno. 
 Bisogna trovare la legge esplicita attraverso cui è possibile calcoalre il livello del glucosio nel sangue partendo dal 
 livello di glucosio interstiziale rilevato attraverso dei sensori sulla pelle.
'''

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

'''
    Si utilizza un tipo di divisione protetta in modo che non ci siano divisioni per 0
    o comunque operazioni non matematicamente possibili.
'''
# Define new functions
def protectedDiv(left, right):
    with numpy.errstate(divide='ignore', invalid='ignore'):
        x = numpy.divide(left, right)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = 1
            x[numpy.isnan(x)] = 1
        elif numpy.isinf(x) or numpy.isnan(x):
            x = 1
    return x


pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(numpy.add, 2, name="add")
pset.addPrimitive(numpy.subtract, 2, name="sub")
pset.addPrimitive(numpy.multiply, 2, name="mul")
#pset.addPrimitive(protectedDiv, 2)
#pset.addPrimitive(numpy.negative, 1, name="neg")
pset.addPrimitive(numpy.square, 1, name="sqr")
pset.addPrimitive(numpy.cos, 1, name="cos")
pset.addPrimitive(numpy.sin, 1, name="sin")
pset.addEphemeralConstant("rand101", lambda: random.uniform(-20, 20)) #costanti generate in maniera randomica
#pset.addEphemeralConstant("rand100", lambda: random.random() * 1000)
pset.renameArguments(ARG0='IG', ARG1='dIG') #variabili incognite della funzione che vogliamo trovare, in questo caso x e y

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

#samples = numpy.linspace(-1, 1, 10000)
#values = samples ** 4 + samples ** 3 + samples ** 2 + samples
#samples_x = numpy.loadtxt('data.dat')

# Load DataSet
data = numpy.loadtxt('dataset.txt')
data_nonzero_rows = data[numpy.all(data != 0, axis=1)]
samples_x = data_nonzero_rows[:,[0,1]]
values = data_nonzero_rows[:,[3]]

#data = numpy.loadtxt('dataset.txt')
#data = data[:,[0,1,2]]
#data = data[numpy.all(data != 0, axis=1)]
#data_nonzero_rows = data[0:500,[0,1,2]]
#samples_x = data_nonzero_rows[:,[0,1]]
#values = data_nonzero_rows[:,[2]]

def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    const = func(samples_x[0,0],samples_x[0,1])
    n = len(values)
    y_pred = numpy.array([])
    for i in range(n):
        y_pred = numpy.append(y_pred, const)

    y_pred.tolist()

    variability = numpy.sqrt(((func(samples_x[:,0],samples_x[:,1]) - y_pred) ** 2).mean())

    # Evaluate the RMSE between the expression and the real function values
    diff = numpy.sqrt(((func(samples_x[:,0],samples_x[:,1]) - values) ** 2).mean())

    if variability > 60.0:
        return diff,
    else:
        return 1000.0,


def staticLimitCrossover(ind1, ind2, heightLimit, toolbox):
    # Store a backup of the original individuals
    keepInd1, keepInd2 = toolbox.clone(ind1), toolbox.clone(ind2)

    # Mate the two individuals
    # The crossover is done in place (see the documentation)
    gp.cxOnePoint(ind1, ind2)

    # If a child is higher than the maximum allowed, then
    # it is replaced by one of its parent
    if ind1.height > heightLimit:
        ind1 = keepInd1
    if ind2.height > heightLimit:
        ind2 = keepInd2

    return ind1, ind2

def staticLimitMutation(individual, expr, heightLimit, toolbox):
    # Store a backup of the original individual
    keepInd = toolbox.clone(individual)

    # Mutate the individual
    # The mutation is done in place (see the documentation)
    gp.mutUniform(individual, expr, pset=pset)

    # If the mutation sets the individual higher than the maximum allowed,
    # replaced it by the original individual
    if individual.height > heightLimit:
        individual = keepInd

    return individual,

toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selDoubleTournament, fitness_size=20, parsimony_size=1.8, fitness_first=True)
toolbox.register("mate", staticLimitCrossover, heightLimit=4, toolbox=toolbox)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register('mutate', staticLimitMutation, expr=toolbox.expr_mut, heightLimit=8, toolbox=toolbox)

#toolbox.register("select", tools.selTournament, tournsize=20)
#toolbox.register("mate", gp.cxOnePoint)
#toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
#toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

expr = toolbox.individual()
nodes, edges, labels = gp.graph(expr)


def main():
    fig, ax = plt.subplots()
    line1, = ax.plot(values, color='red', linewidth=1, label='BG')
    line2, = ax.plot(samples_x[:, 0], color='blue', linewidth=1, label='IG')
    ax.legend(loc='lower right')
    plt.show()

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    random.seed(319)

    pop = toolbox.population(n=2000)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 350, stats=mstats, halloffame=hof, verbose=True)
    print(hof[0])
    final_func = toolbox.compile(expr=hof[0])
    final_data = final_func(samples_x[:,0],samples_x[:,1])
    print(final_data)

    fig, ax = plt.subplots()
    line1, = ax.plot(values, color='red', linewidth=1, label='BG')

    line2, = ax.plot(samples_x[:,0], color='blue', linewidth=1, label='IG')

    line3, = ax.plot(final_data, color='green', linewidth=1, label='Predicted')

    ax.legend(loc='lower right')
    plt.show()

    return pop, log, hof


if __name__ == "__main__":
    main()