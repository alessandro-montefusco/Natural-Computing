import numpy as np
import matplotlib.pyplot as plt


'''
    Il modello definito si basa su un polinomio del quinto ordine
'''
def fmodel(x, w):
    return w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 + w[5] * x**5

'''
    Errore quadratico medio che viene minimizzato tra il modello e quello originale
'''
def rmse(w):
    y_pred = fmodel(x, w)
    return np.sqrt(sum((y - y_pred)**2) / len(y))

def de(fobj, bounds, mut=0.5, crossp=0.8, popsize=100, its=1000):
    dimensions = len(bounds)
    np.random.seed(1234)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
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


'''
    Il problema che si vuole risolvere è un problema di regressione: trovare il miglior fit per i dati e tracciare questo fit
    per vedere se riesco a tracciare la funzione orginiale (Schwefel). Voglio trovare l'rmse più piccolo possibile che approssimi
    il modello creato a quello originale della funzione di Schwefel.
'''
x = np.linspace(0, 100, 500)
y = 418.9829 - x * np.sin(np.sqrt(np.abs(x))) + np.random.normal(0, 10.0, 500) #Funzione Schwefel sporcata con del rumore gaussiano
plt.scatter(x, y)
plt.plot(x, 418.9829 - x * np.sin(np.sqrt(np.abs(x))), '-r', label='Schwefel')
plt.legend()
plt.show()

#In questo problema è importantissimo definire il giusto range entro cui spaziano i pesi che vengono usati nel modello polinomiale.
best_result = list(de(rmse, [(-500, 500)] * 6, mut=0.5, crossp=0.8, popsize=100, its=2000))
print(best_result[-1])
plt.scatter(x, y)
plt.plot(x, 418.9829 - x * np.sin(np.sqrt(np.abs(x))), '-r', label='Schwefel')
plt.plot(x, fmodel(x, best_result[-1][0]), '-g', label='result')
plt.legend()
plt.show()