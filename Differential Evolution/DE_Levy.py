import numpy as np
import matplotlib.pyplot as plt

from yabox.problems import Levy

'''
 Definiamo la funzione obiettivo rispetto al nostro problema di ottimizzazione,
 che consiste nel trovare il minimo globale. In questo caso, il valore del 
 minimo vale 0. Bisogna trovare una configurazione degli iper-parametri tale per
 cui ci si avvicini il più possibile al valore ottimo. 
'''

def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    dimensions = len(bounds)
    np.random.seed(1234)
    pop = np.random.rand(popsize, dimensions)  #NOTA BENE: ogni volta in cui è presente un generatore random, bisogna impostare il seed. Vedi appunti.
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]

# Plot della funzione: landscape del problema
problem = Levy()
problem.plot3d()

'''
Le variabili in ingresso definite sono 10, e per ognuna di esse vengono definiti
bound: [-10; 10]. Il numero di generazioni scelto è 3000
'''
result = list(de(Levy(), bounds=[(-10, 10)] * 10, its=1000))
print("Risultato con 10 variabili: ", result[-1])
x, f = zip(*result)
plt.plot(f)
plt.title("Funzione con 10 variabili")
plt.show()  # plot del risultato con 10 variabili

'''
In questo caso vado ad ottimizzare su un problema considerando un numero di variabli
diverso: 8, 16, 32, 64; e che hanno un nuovo range di definizione
'''
for d in [8, 16, 32, 64]:
    it = list(de(Levy(), [(-100, 100)] * d, mut=0.5, crossp=0.1, popsize=20, its=1000))
    print("Risultato con diverse variabili: ", it[-1])
    x, f = zip(*it)
    plt.plot(f, label='d={}'.format(d))
plt.legend()
plt.title("Funzione con 8, 16, 32, 64 variabili")
plt.show()  # plot del risultato con diverse variabili