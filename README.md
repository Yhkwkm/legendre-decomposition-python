legendre-decomposition-python
====

An implementation of Legendre Decomposition in Python.

## Description

An implementation of the Legendre Decomposition[[1]](https://papers.nips.cc/paper/8097-legendre-decomposition-for-tensors) in Python(>=3.6), which is non-negative tensor decomposition method using information geometric formulation of the log-linear model which we can solve as a convex optimization problem.

This is not an official implementation and the author's implementation in C++ is available here [[2]](https://github.com/mahito-sugiyama/Legendre-decomposition).

## Usage
```python
import numpy as np
from ld import LegendreDecomposition

# generate third-order tensor.
P = np.random.rand(8, 5, 3)

# create an instance of Legendre Decomposition.
ld = LegendreDecomposition(solver='ng', max_iter=5)

# compute and get reconstructed tensors using scikit-learn like API.
reconst_tensor = ld.fit_transform(P)
```

`LegendreDecomposition` offers some options including the following:

- `core_size` : integer, default: 2
  - The parameter for a decomposition basis.

- `solver` : `ng` | `gd`, default: `ng`
  - Type of solver.
    - `ng`: natural gradient method.
    - `gd`: gradient descent method.

- `tol` : float, default: 1e-4
  - Tolerance of the stopping condition.

- `max_iter` : integer, default: 10
  - Maximum number of iterations before timing out.

- `learning_rate` : float, default: 0.1
  - The learning rate used in gradient descent method.

- `random_state` : int, RandomState instance, default=None
  - Used to randomize selection of a decomposition basis, when
    ``shuffle`` is set to ``True``. Pass an int for reproducible
    results across multiple function calls.
    See :term:`Glossary <random_state>`.

- `shuffle` : boolean, default: False
  - If true, randomize selection of a decomposition basis.

- `verbose` : integer, default: 0
  - The verbosity level.

As of now, it doesn't support order of input tensor more than 4, means it supports second or third order.

## Licence

[MIT](https://github.com/Yhkwkm/legendre-decomposition-python/blob/master/LICENSE)

## References
[[1] M. Sugiyama, H. Nakahara, K. Tsuda. Legendre Decomposition for Tensors. Advances in Neural Information Processing Systems 31(NeurIPS2018), pages 8825-8835, 2018.](https://papers.nips.cc/paper/8097-legendre-decomposition-for-tensors)

[[2] Legendre Decomposition implementaion in C++.](https://github.com/mahito-sugiyama/Legendre-decomposition)
