import torch

# importing the utilities module from the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utilities as utils

# Experiment 1:
# Pattern in solutions for different sourcing term values:
# Keeping 'N' and 'boundaryValues' constant and observing the effect of
# 'd_const'
u_init = 0
for N in range(10, 100):
    boundaryValues = torch.zeros((1, 4, N))

    d_const = -10
    solvedPDE1 = utils.solvePDE(
        boundaryValues, d_const, u_init
    )[:, 1:-1, 1:-1]
    for i in range(2, 6):
        d_const = -10 * i
        solvedPDE = utils.solvePDE(
            boundaryValues, d_const, u_init
        )[:, 1:-1, 1:-1]
        assert(torch.allclose(solvedPDE1 * i, solvedPDE, rtol=1e-04))
print('Experiment 1: Done')

# Experiment 2:
# For a constant 'd_const' value, exploring the relationship between solutions
# for zero boundary condition and random boundary conditions
u_init = 0
for N in range(10, 100):
    for i in range(1, 6):
        d_const = -10 * i

        zeroBoundaryValues = torch.zeros((1, 4, N))
        solvedPDEZeroBoundaryValues = utils.solvePDE(
            zeroBoundaryValues, d_const, u_init
        )[:, 1:-1, 1:-1]

        randomBoundaryValues = torch.rand((1, 4, N))
        solvedPDERandom = utils.solvePDE(
            randomBoundaryValues, d_const, u_init
        )[:, 1:-1, 1:-1]
        solvedPDERandomWithZeroDConst = utils.solvePDE(
            randomBoundaryValues, 0, u_init
        )[:, 1:-1, 1:-1]

        assert(
            torch.allclose(
                solvedPDEZeroBoundaryValues + solvedPDERandomWithZeroDConst,
                solvedPDERandom, rtol=1e-04
            )
        )
print('Experiment 2: Done')
