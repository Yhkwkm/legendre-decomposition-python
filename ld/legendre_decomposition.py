""" Legendre Decomposition
"""
import numpy as np
import itertools
from scipy import linalg
#import numba
#from sklearn.base import TransformerMixin, BaseEstimator

# TODO: Support scipy sparse matrix.
# TODO: Improve performance to use dynamic programming in compute_P or compute_eta.
# TODO: Update docstring and comments.

#class LegendreDecomposition(TransformerMixin, BaseEstimator):
class LegendreDecomposition:
    """
    """
    def __init__(self, core_size=2, depth_size=4, solver='ng',
                 tol=1e-4, max_iter=5, learning_rate=0.001,
                 random_state=None, verbose=0, shuffle=False):
        self.core_size = core_size
        self.depth_size = depth_size
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose
        self.shuffle = shuffle

    def fit_transform(self, X, y=None):
        self.theta = self._legendre_decomposition(X)
        Q = self._reconstruct(self.theta) * X.sum()
        self.reconstruction_err_ = self._calc_rmse(X, Q)

        return Q

    def fit(self, X, y=None, **params):
        """Learn a NMF model for the data X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed
        y : Ignored
        Returns
        -------
        self
        """
        self.fit_transform(X, **params)
        return self

    def transform(self):
        """Transform the data X according to the fitted NMF model
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be transformed by the model
        Returns
        -------
        W : array, shape (n_samples, n_components)
            Transformed data
        """
        #check_is_fitted(self)
        self._check_is_fitted()

        Q = self._reconstruct(self.theta)
        return Q

    def _fit_gradient_descent(self, P, beta):
        """
        """
        theta = self._initialize()
        self.eta_hat = self._compute_eta(P)

        for n_iter in range(self.max_iter):
            violation = 0.
            for v in beta:
                # \theta_v \gets \theta_v - \epsilon \times (\eta_v - \hat{\eta_v})
                grad = self._compute_eta(self._reconstruct(theta)) - self.eta_hat
                theta[v] -= self.learning_rate * grad[v]
                #if n_iter == 0:
                #    violation_init = violation
                # TODO: set convergence condition.
                #if violation / violation_init <= self.tol:
                #    if self.verbose:
                #        print("Converged at iteration", n_iter + 1)
                #    break
        return theta

    def _fit_natural_gradient(self, P, beta):
        """
        """
        theta = self._initialize()
        # convert to 1d array (vector).
        theta_vec = np.array([theta[v] for v in beta]).ravel()
        self.eta_hat = self._compute_eta(P)

        for n_iter in range(self.max_iter):
            violation = 0.

            eta = self._compute_eta(self._reconstruct(theta))
            grad = eta - self.eta_hat
            grad_vec = np.array([grad[v] for v in beta]).ravel()

            g = self._compute_jacobian(eta, beta)

            # TODO: check performance of different way to calculate inverse matrix.
            # TODO: Algorithm 7, Information Geometric Approaches for Neural Network Algorithms
            #theta_vec -= np.linalg.solve(g, grad_vec)
            #theta_vec -= linalg.inv(g) * grad_vec
            try:
                theta_vec -= np.dot(np.linalg.inv(g), grad_vec)
            except:
                theta_vec -= np.dot(np.linalg.pinv(g), grad_vec)
            # Update theta
            for n, v in enumerate(beta):
                theta[v] = theta_vec[n]
            #if n_iter == 0:
            #    violation_init = violation
            # TODO: convergence condition.
            #if violation / violation_init <= self.tol:
            #    if self.verbose:
            #        print("Converged at iteration", n_iter + 1)
            #    break
        return theta

    # TODO: change the way of generation basis, obey the author's implementation.
    def _make_basis(self, shape):
        """
        """
        if len(shape) == 2:
            beta = [(i,j) for i, j in itertools.product(range(shape[0]), range(shape[1]))]
        elif len(shape) == 3:
            # TODO: check pattern of basis in the paper.
            beta = [(i,j,k) for i, j, k in itertools.product(range(shape[0]), range(shape[1]), range(shape[2]))]
        else:
            raise NotImplementedError('Order of input tensor should be 2 or 3.')
        return beta

    def _reconstruct(self, theta, b=None):
        """
        """
        # TODO: need to improve perfomances; use dictionary or numpy function.
        shape = theta.shape
        order = len(shape)
        exp_theta = np.exp(theta)
        Q = self.prev_Q
        if b == None:
            b = [0 for i in range(order)]
        if order == 2:
            for i, j in itertools.product(range(b[0], shape[0]), range(b[1], shape[1])):
                Q[i, j] = exp_theta[np.arange(0, i+1)][:, np.arange(0, j+1)].prod()
        elif order == 3:
            for i, j, k in itertools.product(range(b[0], shape[0]), range(b[1], shape[1]), range(b[2], shape[2])):
                Q[i, j, k] = exp_theta[np.arange(0, i+1)][:, np.arange(0, j+1)][:, :, np.arange(0, k+1)].prod()
        else:
            raise NotImplementedError('Order of input tensor should be 2 or 3.')
        psi = Q.sum()
        Q /= psi
        self.prev_Q = Q.copy()
        return Q

    def _compute_eta(self, Q, b=None):
        """
        """
        # TODO: need to improve perfomances; use dictionary or numpy function.
        shape = Q.shape
        order = len(Q.shape)
        eta = self.prev_eta
        if b == None:
            b = [0 for i in range(order)]
        if order == 2:
            for i, j in itertools.product(range(b[0], shape[0]), range(b[1], shape[1])):
                eta[i, j] = Q[np.arange(i, shape[0])][:, np.arange(j, shape[1])].sum()
        elif order == 3:
            for i, j, k in itertools.product(range(b[0], shape[0]), range(b[1], shape[1]), range(b[2], shape[2])):
                eta[i, j, k] = Q[np.arange(i, shape[0])][:, np.arange(j, shape[1])][:, :, np.arange(k, shape[2])].sum()
        else:
            raise NotImplementedError('Order of input tensor should be 2 or 3.')
        self.prev_eta = eta.copy()
        return eta

    def _compute_jacobian(self, eta, beta):
        """Compute jacobian, this is what we call Fisher information matrix.
        """
        size = len(beta)
        g = np.zeros((size, size))
        order = len(eta.shape)
        # TODO: consider the way of creating vector.
        # In author's code, the indexes of vector are randomly generated.
        # https://github.com/mahito-sugiyama/Legendre-decomposition/blob/master/src/cc/legendre_decomposition.h#L211
        if order == 2:
            for i, j in itertools.product(range(size), range(size)):
                g[i, j] = eta[np.max((beta[i][0], beta[j][0])), \
                              np.max((beta[i][1], beta[j][1]))
                             ] \
                            - eta[beta[i]] * eta[beta[j]]
        elif order == 3:
            for i, j in itertools.product(range(size), range(size)):
                g[i, j] = eta[np.max((beta[i][0], beta[j][0])), \
                              np.max((beta[i][1], beta[j][1])), \
                              np.max((beta[i][2], beta[j][2]))
                             ] \
                            - eta[beta[i]] * eta[beta[j]]
        else:
            raise NotImplementedError('Order of input tensor should be 2 or 3.')
        return g

    def _calc_rmse(self, P, Q):
        """
        """
        return np.sqrt(np.mean(np.square(P - Q)))

    def _check_is_fitted(self):
        """
        """
        raise NotImplementedError()

    def _normalizer(self, X):
        """
        """
        # TODO: check if tensor has NaN values.
        return X / np.sum(X)

    def _initialize(self):
        """
        """
        theta = np.zeros(self.shape)
        self.prev_Q = np.zeros(self.shape)
        self.prev_eta = np.zeros(self.shape)
        return theta

    def _init_test_tensor(self, random_state):
        """
        """
        np.random.seed(random_state)
        test_tensor = np.random.rand(*self.shape)
        return test_tensor

    def _legendre_decomposition(self, P):
        """
        """
        # TODO: need to separately declare P shape and basis shape.
        self.shape = P.shape
        order = len(P.shape)
        if order >= 4:
            raise NotImplementedError("Order of input tensor should be 2 or 3. Order: '%s'." % order)
        # normalize tensor
        P = self._normalizer(P)
        beta = self._make_basis(self.shape)
        if self.solver == 'ng':
            theta = self._fit_natural_gradient(P, beta)
        elif self.solver == 'gd':
            theta = self._fit_gradient_descent(P, beta)
        else:
            raise ValueError("Invalid solver parameter '%s'." % self.solver)

        return theta


def main():
    np.random.seed(2020)
    P = np.random.rand(8, 5, 3)
    ld = LegendreDecomposition(solver='ng', max_iter=5, learning_rate=1)
    reconst_tensor = ld.fit_transform(P)
    print('Reconstruction error(RMSE): %s' % ld.reconstruction_err_)
    np.set_printoptions(threshold=200)
    print('============= Original Tensor =============')
    print(P)
    print('============= Reconstructed Tensor =============')
    print(reconst_tensor)

if __name__ == '__main__':
    main()