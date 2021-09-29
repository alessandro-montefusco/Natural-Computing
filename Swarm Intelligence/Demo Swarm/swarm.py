import sys
# Change directory to access the pyswarms module
sys.path.append('../')

# Import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Mesher

rc('animation', html='html5')

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx


# Initialize mesher with sphere function
# La funzione rastrigin è una funzione nota a priori, utilizzata per testare gli algoritmi di evoluzione naturale
# che hanno un numero di parametri molto maggiore di quelli utilizzati in questo caso (2D-dimension).
m = Mesher(func=fx.rastrigin)

# Set-up hyperparameters
# w è il peso che viene adto al vettore di base
# k, p sono due parametri dell'algoritmo di "local best" che stiamo utilizzando.
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}

# Call instance of PSO
# dimensions rappresenta ???
optimizer = ps.single.LocalBestPSO(n_particles=50, dimensions=2, options=options)

# Perform optimization
cost, pos = optimizer.optimize(fx.rastrigin, iters=1000)

print(cost)

plot_cost_history(cost_history=optimizer.cost_history)
plt.show()

animation = plot_contour(pos_history=optimizer.pos_history, mesher=m,mark=(0,0))
# Enables us to view it in a Jupyter notebook
HTML(animation.to_html5_video())
# Il video mostra al centro il minimo assoluto della funzione, punto che lo swarm cerca di identificare.
animation.save('dynamic_images.mp4')
