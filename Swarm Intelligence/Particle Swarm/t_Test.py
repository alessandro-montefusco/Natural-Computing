# Import modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
import xlwt
from xlwt import Workbook


# Import PySwarms
import pyswarms as ps
import numpy as np

# Load the iris dataset
data = load_breast_cancer()

# Store the features as X and the labels as y
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

'''
    Particle Swarm Optimizazion: otimizzazione.
    Questo script viene utilizzato per far evolvere i pesi di una rete neuronica che effettua l'addestramento su 
    un dataset (530 campioni): 30 parametri per ogni campione, che vanno a labellare un campione come tumore al polmone
    positivo o tumore al polmone negativo.
'''

# Forward propagation
def forward_prop(params, n_inputs, n_hidden, n_classes):
    """Forward propagation as objective function

    This computes for the forward propagation of the neural network, as
    well as the loss. It receives a set of parameters that must be
    rolled-back into the corresponding weights and biases.

    Inputs
    ------
    params: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    float
        The computed negative log-likelihood loss given the parameters
    """

    # Roll-back the weights and biases
    W1 = params[0:n_inputs*n_hidden].reshape((n_inputs, n_hidden))
    b1 = params[n_inputs*n_hidden:n_inputs*n_hidden+n_hidden].reshape((n_hidden,))
    W2 = params[n_inputs*n_hidden+n_hidden:n_inputs*n_hidden+n_hidden+n_classes*n_hidden].reshape((n_hidden,n_classes))
    b2 = params[n_inputs*n_hidden+n_hidden+n_classes*n_hidden:n_inputs*n_hidden+n_hidden+n_classes*n_hidden+n_classes].reshape((n_classes,))

    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    logits = z2          # Logits for Layer 2

    # Compute for the softmax of the logits
    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute for the negative log likelihood
    N = len(X) # Number of samples
    corect_logprobs = -np.log(probs[range(N), y])
    loss = np.sum(corect_logprobs) / N

    return loss

def f(x):
    """Higher-level method to do forward_prop in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [forward_prop(x[i], n_inputs, n_hidden, n_classes) for i in range(n_particles)]
    return np.array(j)


def predict(Test, pos, n_inputs, n_hidden, n_classes):
    """
    Use the trained weights to perform class predictions.

    Inputs
    ------
    X: numpy.ndarray
        Input Iris dataset
    pos: numpy.ndarray
        Position matrix found by the swarm. Will be rolled
        into weights and biases.
    """
    # Neural network architecture
    #n_inputs = 4
    #n_hidden = 20
    #n_classes = 3

    # Roll-back the weights and biases
    W1 = pos[0:n_inputs * n_hidden].reshape((n_inputs, n_hidden))
    b1 = pos[n_inputs * n_hidden:n_inputs * n_hidden + n_hidden].reshape((n_hidden,))
    W2 = pos[n_inputs * n_hidden + n_hidden:n_inputs * n_hidden + n_hidden + n_classes * n_hidden].reshape(
        (n_hidden, n_classes))
    b2 = pos[n_inputs * n_hidden + n_hidden + n_classes * n_hidden:n_inputs * n_hidden + n_hidden + n_classes * n_hidden + n_classes].reshape(
        (n_classes,))

    # Perform forward propagation
    z1 = Test.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    logits = z2          # Logits for Layer 2

    y_pred = np.argmax(logits, axis=1)
    return y_pred

# Neural network architecture
n_inputs = 30
n_hidden = 20
# un livello hidden soltanto con 20 neuroni. Tale parametro pu?? essere cambiato. Per avere pi?? hidden layer
# devo andare a modificare la funzione predict, aggiungendo pi?? livelli.
n_classes = 2

# Call instance of PSO
# rappresenta il numero di pesi che la PSO deve andare ad ottimizzare.
dimensions = (n_inputs * n_hidden) + (n_hidden * n_classes) + n_hidden + n_classes


rows = []
const_random = 0

while const_random <= 200:
    # Seed per l'inizializzazione
    np.random.seed(1234 + const_random)

    # Set-up hyperparameters
    options = {'c1': 0, 'c2': 1.1, 'w': 0.7298}
    optimizer = ps.single.GlobalBestPSO(n_particles=200, dimensions=dimensions, options=options)

    # Perform optimization
    cost, pos = optimizer.optimize(f, iters=200)
    rows.append((predict(X_test, pos, n_inputs, n_hidden, n_classes) == y_test).mean())
    const_random +=20

print(rows)

