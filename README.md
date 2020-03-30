# multi_period_liability_clearing

Code for the paper [Multi-period liability clearing via convex optimal control](http://web.stanford.edu/~boyd/papers/multi_period_liability_clearing.html).

## Dependencies

The code is written in the Python language, and
has the following dependencies:
```
numpy
scipy
cvxpy
networkx
matplotlib
mosek
```

To use the MOSEK solver, you will need to follow instructions listed [here](https://docs.mosek.com/9.1/install/installation.html).
Otherwise, you will have to remove all instances of `solver="MOSEK"`.

## Examples
The examples in section 6.1 are in `liability_clearing.py`.
The example in section 6.2 is in `liability_reduction.py`.
The example in section 6.3 is in `liability_mpc.py`.
The example in section 7.5 is in `non_cleared_liabilities.py`.

To run, for example, the examples in 6.1, run
```
$ python liability_clearing.py 
```

## License
This repository carries an Apache 2.0 license.

## Citing
If you use cvxpylayers for research, please cite our accompanying [NeurIPS paper](http://web.stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf):

```
@inproceedings{cvxpylayers2019,
  author={Agrawal, A. and Amos, B. and Barratt, S. and Boyd, S. and Diamond, S. and Kolter, Z.},
  title={Differentiable Convex Optimization Layers},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019},
}
```
