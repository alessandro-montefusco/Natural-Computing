import numpy as np
from numpy.random import uniform


class Clonalg:
    def __init__(self, P=1, ap_cost=200, ap_rad=50, ds=None, aps=None, mst=None, k_cost=None, k_n_client=None):
        self._P = P
        self._ap_cost = ap_cost
        self._ap_rad = ap_rad
        self._ds = ds
        self._aps = aps
        self._mst = mst
        self._k_n_client = k_n_client
        self._k_cost = k_cost

    def affinity(self, p_i):
        """
        Description
        -----------
        Return the affinity of one subject.

        Parameters
        -----------
        p_i: numpy.array
            Subject of a population.

        Return
        -----------
        return: float
            Affinity of the subject passed as parameter.

        """

        # - Compute number of APs in subject's range
        # - Compute intersection area with other AP's ranges
        intersect_area = 0
        n_aps = 0
        for ap in self._aps:
            inside, distance = self.is_inside(p_i, 2 * self._ap_rad, ap)
            if inside:
                n_aps += 1
                intersect_area += self.intersection_area(distance ** (0.5), self._ap_rad)

        # - Compute number of clients in subject's range
        # - Compute signal provided to clients
        n_clients = 0
        sig = 0
        for client in self._ds:
            inside, distance = self.is_inside(p_i, self._ap_rad, client)
            if inside:
                n_clients += 1
                sig += self._P / (4 * np.pi * distance)

        # Find index of subject in APs list
        if type(p_i) is tuple:
            index = [np.array_equal(p_i[0], x) for x in self._aps].index(True)
        else:
            index = [np.array_equal(p_i, x) for x in self._aps].index(True)

        # Find length of wires
        row = self._mst[index, :]
        column = self._mst[:, index]
        mst_p_i = row + column.T

        # Compute affinity
        if n_clients == 0 or self._mst.sum() == 0:
            aff = np.inf
        else:
            # aff = ((n_aps/len(self._aps)) * (mst_p_i.sum()/self._mst.sum())) / (100*n_clients/(len(self._ds)))  # +
            # sig / n_clients
            aff = self._k_cost * ((intersect_area - np.pi * (self._ap_rad ** 2)) / (
                    n_aps * np.pi * (self._ap_rad ** 2)) + n_aps / len(
                self._aps) + mst_p_i.sum() / self._mst.sum()) - self._k_n_client * n_clients
        return aff

    # Check if "other" is inside ap's range
    def is_inside(self, ap, rad, other):
        if type(ap) is tuple:
            ap = ap[0]
        if type(other) is tuple:
            other = other[0]
        # Compare radius of circle with distance of its center from given point
        # print('ap0', ap[0], 'ap1', ap[1])
        # print('other0', other[0], 'other1', other[1])
        sq_dist = (other[0] - ap[0]) * (other[0] - ap[0]) + (other[1] - ap[1]) * (other[1] - ap[1])
        # print('sq', sq_dist)
        if sq_dist <= rad * rad:
            return (True, sq_dist)
        else:
            return (False, None)

    # Create random antibodies
    def create_random_cells(self, population_size, problem_size, b_lo, b_up):
        population = [uniform(low=b_lo, high=b_up, size=problem_size) for x in range(population_size)]

        return population

    # Create a number of clones (lower affinity, more clones)
    # Please note:
    # - lower bound is used to avoid a matrix-related issue
    # - upper bound is used to avoid extremely slow execution time
    def clone(self, p_i, clone_rate):
        clone_num = int(clone_rate / p_i[1])
        if clone_num > 20:
            clone_num = 20  
        clones = [(p_i[0], p_i[1]) for x in range(clone_num)]

        return clones

    # Mutate antibody (lower affinity, lighter mutation)
    def hypermutate(self, p_i, mutation_rate, b_lo, b_up):
        ind_tmp = []
        k = p_i[1] * mutation_rate
        for gen in p_i[0]:
            new = gen * k
            if new < b_lo:
                new = b_lo
            elif new > b_up:
                new = b_up
            ind_tmp.append(new)
        return np.array(ind_tmp)

    # Selection
    def select(self, pop, pop_clones, pop_size):
        population = pop + pop_clones
        population = sorted(population, key=lambda x: x[1])[:pop_size]

        return population

    # Replacement (same code as select, indeed)
    def replace(self, population, population_rand, population_size):
        population = population + population_rand
        population = sorted(population, key=lambda x: x[1])[:population_size]

        return population

    # Set APs list
    def set_aps(self, aps):
        self._aps = aps

    # Set MST
    def set_mst(self, mst):
        self._mst = mst

    # Compute intersection area between 2 circles at distance d from each other, both having radius r
    def intersection_area(self, d, r):
        """Return the area of intersection of two circles.

        The circles have radii R and r, and their centres are separated by d.

        """

        if d <= 0:
            # Circles completely overlap
            return np.pi * r ** 2
        if d >= 2 * r:
            # Circles don't overlap at all
            return 0

        r2, d2 = r ** 2, d ** 2
        alpha = np.arccos(d2 / (2 * d * r))
        return r2 * alpha + r2 * alpha - 0.5 * (r2 * np.sin(2 * alpha) + r2 * np.sin(2 * alpha))
