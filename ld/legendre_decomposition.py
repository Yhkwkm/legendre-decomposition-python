""" Legendre Decomposition
"""
import numpy as np
import itertools
from scipy import linalg
from enum import Enum
#import numba
#from sklearn.base import TransformerMixin, BaseEstimator

# TODO: Support scipy sparse matrix.
# TODO: Improve performance to use dynamic programming in compute_P or compute_eta, using cython
# TODO: Update docstring and comments.

class Constants(Enum):
    EPSILON = 1e-100


#class LegendreDecomposition(TransformerMixin, BaseEstimator):
class LegendreDecomposition:
    """Legendre Decomposition

    Find non-negative decomposable tensor Q whose multiplicative
    combination of parameters approximates the non-
    negative input tensor P, solving as a convex optimization
    problem thanks to information geometric formulation.

    Parameters
    ----------
    solver : 'ng' | 'gd'
        Type of solver.

        - 'ng': natural gradient method.
        - 'gd': gradient descent method.

    Attributes
    ----------

    Examples
    --------
    >>> import numpy as np
    >>> P = np.random.rand(8, 5, 3)
    >>> from ld import LegendreDecomposition
    >>> ld = LegendreDecomposition(solver='ng', max_iter=5)
    >>> Q = ld.fit_transform(P)

    References
    ----------
    M. Sugiyama, H. Nakahara, K. Tsuda.
    "Legendre Decomposition for Tensors".
    Advances in Neural Information Processing Systems 31(NeurIPS2018),
    pages 8825-8835, 2018.
    https://papers.nips.cc/paper/8097-legendre-decomposition-for-tensors
    """

    def __init__(self, core_size=2, depth_size=4, solver='ng',
                 tol=1e-4, max_iter=5, learning_rate=0.1,
                 random_state=None, shuffle=False, verbose=0):
        self.core_size = core_size
        self.depth_size = depth_size
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose
        if self.verbose:
            np.set_printoptions(threshold=200)

    def fit_transform(self, P, y=None):
        r"""Learn a Legendre Decomposition model for the data P and
        returns the transformed data.
        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        y : Ignored

        Returns
        -------
        Q : array
            second/third-order tensor.
            Transformed data reconstructed by parameter \theta.
        """
        self.theta = self._legendre_decomposition(P)
        Q = self._compute_Q(self.theta) * P.sum()
        self.reconstruction_err_ = self._calc_rmse(P, Q)

        return Q

    def fit(self, P, y=None, **params):
        """Learn a Legendre Decomposition model for the data.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        y : Ignored

        Returns
        -------
        self
        """
        self.fit_transform(P, **params)

        return self

    def transform(self):
        r"""Transform the data P according to the fitted Legendre Decomposition model.

        Parameters
        ----------
        None

        Returns
        -------
        Q : array
            second/third-order tensor.
            Transformed data reconstructed by parameter \theta.
        """
        self._check_is_fitted()

        return self._compute_Q(self.theta)

    def _compute_Q(self, theta, b=None):
        r"""Compute decomposable tensor Q from parameter \theta.

        Parameters
        ----------
        theta : array
            second/third-order tensor.
            Same shapes as input tensor P.

        b : array
            set of decomposition basis B.

        Returns
        -------
        Q : array
            second/third-order tensor.
            Decomposable tensor.
        """
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
            raise NotImplementedError("Order of input tensor should be 2 or 3. Order: {}.".format(order))
        psi = Q.sum()
        Q /= psi
        self.prev_Q = Q.copy()
        return Q

    def _compute_eta(self, Q, b=None):
        r"""Compute parmaters \eta from decomposable tensor Q.

        Parameters
        ----------
        Q : array
            second/third-order tensor.
            Decomposable tensor.

        b : array
            set of decomposition basis B.

        Returns
        -------
        eta : array
            second/third-order tensor.
            parameter \eta.
            Same shapes as input tensor P.
        """
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
            raise NotImplementedError("Order of input tensor should be 2 or 3. Order: {}.".format(order))
        self.prev_eta = eta.copy()

        return eta

    def _compute_jacobian(self, eta, beta):
        """Compute jacobian matrix, this is what we call Fisher information matrix.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        y : Ignored

        Returns
        -------
        g : array, shape (cardinality of basis, cardinality of basis)
            Fisher information matrix.
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
                              np.max((beta[i][1], beta[j][1]))] \
                            - eta[beta[i]] * eta[beta[j]]
        elif order == 3:
            for i, j in itertools.product(range(size), range(size)):
                g[i, j] = eta[np.max((beta[i][0], beta[j][0])), \
                              np.max((beta[i][1], beta[j][1])), \
                              np.max((beta[i][2], beta[j][2]))] \
                            - eta[beta[i]] * eta[beta[j]]
        else:
            raise NotImplementedError("Order of input tensor should be 2 or 3. Order: {}.".format(order))

        return g

    def _compute_residual(self, eta, beta):
        """Compute root mean squared error(rmse),
        which is reconstructed error between input tensor P and reconstrcted tensor Q.

        Parameters
        ----------
        eta : array
            second/third-order tensor.
            Same shapes as input tensor P.

        Returns
        -------
        residual : float
        """
        res = np.sqrt(np.sum([(eta[v] - self.eta_hat[v])**2 for v in beta]))
        return res

    def _calc_rmse(self, P, Q):
        """Compute root mean squared error(rmse),
        which is reconstructed error between input tensor P and reconstrcted tensor Q.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        Q : array
            second/third-order tensor.
            Decomposable tensor.

        Returns
        -------
        rmse : float
            rmse of reconstructed error.
        """
        return np.sqrt(np.mean(np.square(P - Q)))

    def _check_is_fitted(self):
        """Check if fit() has already been executed.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def _normalizer(self, P):
        """normalize input tensor P by summation of P.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        Returns
        -------
        norm_P : array
            second/third-order tensor.
            Normalized data tensor to be decomposed.
        """
        # TODO: check if tensor has NaN values.
        return P / np.sum(P)

    def _initialize(self):
        r"""Initialize paramters \theta, \eta and decomposable tensor Q.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        y : Ignored

        Returns
        -------
        theta : array
            second/third-order tensor.
            Same shapes as input tensor P.

        eta : array
            second/third-order tensor.
            parameter \eta.
            Same shapes as input tensor P.

        Q : array
            second/third-order tensor.
            Decomposable tensor.
        """
        theta = np.zeros(self.shape)
        eta = np.zeros(self.shape)
        Q = np.zeros(self.shape)

        return theta, eta, Q

    def _gen_norm(self, shape):
        pass

    def _sort_basis(self, v):
        return self.P[v]

    def _gen_core(self, shape):
        order = (len(shape))
        beta = []
        for i in range(shape[0]):
            temp_beta = []
            c_size = self.core_size
            if order == 2:
                for j in range(shape[1]):
                    if self.basis_index[i, j] == 0:
                        temp_beta.append((i, j))
            elif order == 3:
                for j, k in itertools.product(range(shape[1]), range(shape[2])):
                    if self.basis_index[i, j, k] == 0:
                        temp_beta.append((i, j, k))
            if self.shuffle:
                np.random.shuffle(temp_beta)
            else:
                temp_beta.sort(key=self._sort_basis)
            if c_size > len(temp_beta):
                c_size = len(temp_beta)
            for c in range(c_size):
                beta.append(temp_beta[c])
                self.basis_index[temp_beta[c]] = 1

        return beta

    def _gen_basis_2(self, shape):
        """
        """
        beta = []
        self.basis_index = np.zeros(shape)
        if self.solver == 'ng':
            beta += self._gen_norm(shape)
        beta += self._gen_core(shape)

        return beta

    def _gen_basis(self, shape):
        """Generate set of decomposition basis B,
        which are used for reconstructing tensor Q.
        This basis are paramters how decomposed input tensor P,
        thus the basis are directory related to complexity of models.

        Parameters
        ----------
        shape : int
            order of tensor.

        Returns
        -------
        beta : list
            sets of decomposition basis vectors.
        """
        if len(shape) == 2:
            beta = [(i,j) for i, j in itertools.product(range(shape[0]), range(shape[1]))]
        elif len(shape) == 3:
            beta = [(i,j,k) for i, j, k in itertools.product(range(shape[0]), range(shape[1]), range(shape[2]))]
        else:
            raise NotImplementedError("Order of input tensor should be 2 or 3. Order: {}.".format(len(shape)))
        if self.verbose:
            print("\n\n============= set of basis =============")
            print(beta)
        return beta

    def _fit_gradient_descent(self, P, beta):
        r"""Compute parameter \theta using Gradient Descent-based optimization algorithms.

        The objective function is KL divergence(P, Q) and is minimized with
        Gradient Descent method.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        beta : list
            sets of decomposition basis vectors.

        Returns
        -------
        theta : array
            second/third-order tensor.
            Same shapes as input tensor P.
        """
        theta, self.prev_eta, self.prev_Q = self._initialize()
        self.eta_hat = self._compute_eta(P)
        self.res = 0.

        for n_iter in range(self.max_iter):
            eta = self._compute_eta(self._compute_Q(theta))
            if self.verbose:
                print("\n\n============= iteration: {}, eta =============".format(n_iter))
                print(eta)

            prev_res = self.res
            self.res = self._compute_residual(eta, beta)
            if self.verbose:
                print("n_iter: {}, Residual: {}".format(n_iter, self.res))
            # check convergence
            if (self.res <= self.tol) or (prev_res <= self.res and Constants.EPSILON.value <= prev_res):
                self.converged_n_iter = n_iter
                print("Convergence of theta at iteration: {}".format(self.converged_n_iter))
                break

            for v in beta:
                # \theta_v \gets \theta_v - \epsilon \times (\eta_v - \hat{\eta_v})
                grad = self._compute_eta(self._compute_Q(theta)) - self.eta_hat
                theta[v] -= self.learning_rate * grad[v]

            if self.verbose:
                print("\n\n============= iteration: {}, theta =============".format(n_iter))
                print(theta)

        return theta

    def _fit_natural_gradient(self, P, beta):
        r"""Compute parameter \theta using Natural Gradient-based optimization algorithms.

        The objective function is KL divergence(P, Q) and is minimized with
        Natural Gradient method.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        beta : list
            sets of decomposition basis vectors.

        Returns
        -------
        theta : array
            second/third-order tensor.
            Same shapes as input tensor P.
        """
        theta, self.prev_eta, self.prev_Q = self._initialize()
        self.eta_hat = self._compute_eta(P)
        self.res = 0.
        theta_vec = np.array([theta[v] for v in beta])

        for n_iter in range(self.max_iter):
            eta = self._compute_eta(self._compute_Q(theta))
            if self.verbose:
                print("\n\n============= iteration: {}, eta =============".format(n_iter))
                print(eta)

            prev_res = self.res
            self.res = self._compute_residual(eta, beta)
            if self.verbose:
                print("n_iter: {}, Residual: {}".format(n_iter, self.res))
            # check convergence
            if (self.res <= self.tol) or (prev_res <= self.res and Constants.EPSILON.value <= prev_res):
                self.converged_n_iter = n_iter
                print("Convergence of theta at iteration: {}".format(self.converged_n_iter))
                break

            # compute \eta_delta and Fisher information matrix.
            grad = eta - self.eta_hat
            grad_vec = np.array([grad[v] for v in beta])
            g = self._compute_jacobian(eta, beta)

            # TODO: check performance of different way to calculate inverse matrix.
            # TODO: Algorithm 7, Information Geometric Approaches for Neural Network Algorithms
            #theta_vec -= np.linalg.solve(g, grad_vec)
            #theta_vec -= np.dot(linalg.inv(g), grad_vec)
            try:
                theta_vec -= np.dot(np.linalg.inv(g), grad_vec)
            except:
                theta_vec -= np.dot(np.linalg.pinv(g), grad_vec)
            # Update theta
            for n, v in enumerate(beta):
                theta[v] = theta_vec[n]

            if self.verbose:
                print("\n\n============= iteration: {}, theta =============".format(n_iter))
                print(theta)

        return theta

    def _legendre_decomposition(self, P):
        """Compute Legendre decomposition.

        Parameters
        ----------
        P : array
            second/third-order tensor.
            Data tensor to be decomposed.

        Returns
        -------
        theta : array
            second/third-order tensor.
            Same shapes as input tensor P.
            Used for reconstructing decomposable tensor Q.
        """
        # TODO: need to separately declare P shape and basis shape.
        self.shape = P.shape
        order = len(P.shape)
        if order not in (2, 3):
            raise NotImplementedError("Order of input tensor should be 2 or 3. Order: {}.".format(order))
        # normalize tensor
        self.P = self._normalizer(P)
        beta = self._gen_basis(self.shape)
        if self.solver == 'ng':
            theta = self._fit_natural_gradient(self.P, beta)
        elif self.solver == 'gd':
            theta = self._fit_gradient_descent(self.P, beta)
        else:
            raise ValueError("Invalid solver parameter {}.".format(self.solver))

        return theta


def main():
    np.random.seed(2020)
    P = np.random.rand(8, 5, 3)
    ld = LegendreDecomposition(solver='ng', max_iter=5)
    reconst_tensor = ld.fit_transform(P)
    print('Reconstruction error(RMSE): {}'.format(ld.reconstruction_err_))
    np.set_printoptions(threshold=200)
    print("\n\n============= Original Tensor =============")
    print(P)
    print("\n\n============= Reconstructed Tensor =============")
    print(reconst_tensor)

if __name__ == '__main__':
    main()