import torch
import pandas
import random

# importing the utilities module from the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utilities as utils

# Seeds for reproducibility
random.seed(1)
#np.random.seed(1)
torch.manual_seed(1)

# Hyperparams
TRAINING_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# Importing the data from CSV file
df = pandas.read_csv('data/data.csv')
ipoptInfoDf = pandas.read_csv('experiment/ipoptInformation.csv')

# Input generator
def getEquationAt(index):
    N = df.iloc[index].N
    alpha = df.iloc[index].alpha
    lb_y = df.iloc[index].lb_y
    ub_y = df.iloc[index].ub_y
    lb_u = df.iloc[index].lb_u
    ub_u = df.iloc[index].ub_u
    d_const = df.iloc[index].d_const
    target_profile_equation = df.iloc[index].target_profile_equation
    costIPOPT = df.iloc[index].cost

    d_const = torch.tensor([[d_const]], dtype=torch.float)
    bounds = torch.tensor([[lb_y, ub_y, lb_u, ub_u]])
    array = utils.generateProfile(target_profile_equation, N)
    u_init = 0

    return N, array, bounds, d_const, u_init, alpha

# Input generator
def generateInputToNetwork(mode, useIPOPTConvergedProblems=True):
    # Sample an index
    trainDataEnd = int(len(df) * TRAINING_SPLIT)
    validDataEnd = trainDataEnd + int(len(df) * VALIDATION_SPLIT)
    if mode == 'training':
        index = random.randint(0, trainDataEnd - 1)
    elif mode == 'validation':
        index = random.randint(trainDataEnd, validDataEnd - 1)
    elif mode == 'testing':
        index = random.randint(validDataEnd, len(df) - 1)
    else:
        raise Exception("Undefined/unexpected mode value")

    if useIPOPTConvergedProblems and ipoptInfoDf.iloc[index].convergedToOptimalFlag != 1:
        # The problem had not converged in IPOPT; Let's get another one.
        index, N, array, bounds, d_const, u_init, alpha = generateInputToNetwork(
            mode, useIPOPTConvergedProblems=useIPOPTConvergedProblems)
    else:
        # Get the equation at the index
        N, array, bounds, d_const, u_init, alpha = getEquationAt(index)

    return index, N, array, bounds, d_const, u_init, alpha

# Solve using optimizer model
def solveWithOptimizerModel(N, model, guessBVal, array, bounds, d_const, u_init, alpha, loss_fn, max_steps):
    model.reset()

    x = torch.zeros((1, 4, N), requires_grad=True)
    loss_min = float('inf')

    for _ in range(max_steps):
        outputs = guessBVal + x

        # Compute the loss
        innerLoss = loss_fn(outputs, array, bounds, d_const, u_init, alpha)

        if (loss_min > innerLoss.item()):
            loss_min = innerLoss.item()
            x_min = x.clone()

        # Get the gradients
        grads = torch.autograd.grad(innerLoss, outputs)
        assert len(grads) == 1
        grad = grads[0]

        # Use the custom optimizer model
        x = model(x, grad)

    # Calculate the loss
    loss = loss_fn(guessBVal + x, array, bounds, d_const, u_init, alpha)
    if (loss_min > loss.item()):
        loss_min = loss.item()
        x_min = x.clone()

    return x_min, loss_min


# Solve using optimizer model
def solveWithOptimizerModelAndCollectStatistics(N, model, guessBVal, array, bounds, d_const, u_init, alpha, loss_fn, max_steps, costIPOPT):
    model.reset()

    x = torch.zeros((1, 4, N), requires_grad=True)
    minNetCost = float('inf')
    minNetCostIter = -1

    minCostWithoutCV = float('inf')
    minCostWithoutCVIter = -1

    beatIPOPTIter = -1
    beatIPOPTWithZeroViolationIter = -1

    for iter in range(max_steps + 1):
        outputs = guessBVal + x

        # Compute the loss
        innerLoss, cvCost = loss_fn(outputs, array, bounds, d_const, u_init, alpha, retCVCostFlag=True)

        if (minNetCost > innerLoss.item()):
            minNetCost = innerLoss.item()
            minNetCostIter = iter

            # Divide the cost with 'utils.constraintViolationCost' to get the squared constraint violation.
            # Take square root of that to get the absolute constraint violation.
            minNetMaxAbsCV = torch.sqrt(cvCost.max() / utils.constraintViolationCost)
            minNetMeanSquaredCV = cvCost.mean() / utils.constraintViolationCost

            if beatIPOPTIter == -1:
                if minNetCost < costIPOPT:
                    beatIPOPTIter = iter

        if (cvCost.sum() == 0):
            if (minCostWithoutCV > innerLoss.item()):
                minCostWithoutCV = innerLoss.item()
                minCostWithoutCVIter = iter

                if beatIPOPTWithZeroViolationIter == -1:
                    if minCostWithoutCV < costIPOPT:
                        beatIPOPTWithZeroViolationIter = iter

        # This is much easier than having all of the above code written again at the end
        if iter == max_steps:
            break

        # Get the gradients
        grads = torch.autograd.grad(innerLoss, outputs)
        assert len(grads) == 1
        grad = grads[0]

        # Use the custom optimizer model
        x = model(x, grad)

    return (
        minNetCost, minNetCostIter, minNetMaxAbsCV, minNetMeanSquaredCV,
        minCostWithoutCV, minCostWithoutCVIter,
        beatIPOPTIter, beatIPOPTWithZeroViolationIter
    )
