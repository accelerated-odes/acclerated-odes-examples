# Chemical Kinetics Example

By default, this will compile to run in serial on a CPU.

To compile for integration on a GPU, do `cmake -DENABLE_CUDA=ON`.

The chemical kinetics system is 3 equations, and this test integrates
N such systems, where N = grid_size**3.

grid_size can be changed in `main.cpp`, it is 32 by default.

Uses analytic solution for the sparse linear solve computed with https://github.com/accelerated-odes/gauss-jordan-solver
