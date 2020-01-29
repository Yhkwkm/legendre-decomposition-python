legendre-decomposition-python
====

An implementation of Legendre Decomposition in Python.

### Description

An implementation of the Legendre Decomposition[[1]](https://papers.nips.cc/paper/8097-legendre-decomposition-for-tensors) in Python(>=3.6), which is non-negative tensor decomposition method using information geometry formulation of the log-linear mode which we can solve as a convex optimization problem.

This is not an official implementation. The author publish C++ implementaion[[2]](https://github.com/mahito-sugiyama/Legendre-decomposition).

### Usage
```python
# create an instance of Legendre Decomposition.
from ld import LegendreDecomposition
ld = LegendreDecomposition(solver='ng', max_iter=5)
# compute and get reconstructed tensors using scikit-learn like API.
reconst_tensor = ld.fit_transform(X)
```
`LegendreDecomposition` offers some options including the following:

- `solver`: Type of solver:
  - `'ng'`: natural gradient method.
  - `'gd'`: gradient descent method.
- `max_iter`: The number of iterations of the solver.
- `learning_rate`: Learning rate for gradient descent method.


As of now, it doesn't support order of input tensor more than 4, means it support second or third order.


### Licence

[MIT](https://github.com/Yhkwkm/legendre-decomposition-python/LICENCE)

### References
[[1] Sugiyama, M., Nakahara, H., Tsuda, K. Legendre Decomposition for Tensors. Advances in Neural Information Processing Systems 31(NeurIPS2018), pages 8825-8835, 2018.](https://papers.nips.cc/paper/8097-legendre-decomposition-for-tensors)

[[2] C++ implementaion of Legendre Decomposition.](https://github.com/mahito-sugiyama/Legendre-decomposition)
