import torch
import random
import pandas
import re
from tqdm import tqdm

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
NUM_PROBLEMS = 10000

FOLDER_NAME = 'experiment'
OUTPUT_FILE_NAME = FOLDER_NAME + '/ipoptInformation.csv'

csvFile = open(OUTPUT_FILE_NAME, 'a') # Appending as an insurance
csvFile.write(
    'i,cost,numIterations,'
    + 'objectiveScaled,objectiveUnscaled,'
    + 'dualInfeasibilityScaled,dualInfeasibilityUnscaled,'
    + 'constraintViolationScaled,constraintViolationUnscaled,'
    + 'variableBoundViolationScaled,variableBoundViolationUnscaled,'
    + 'complementarityScaled,complementarityUnscaled,'
    + 'overallNLPErrorScaled,overallNLPErrorUnscaled,'
    + 'numObjectiveFunctionEvals,numObjectiveGradientEvals,'
    + 'numEqualityConstraintEvals,numInequalityConstraintEvals,'
    + 'numEqualityConstraintJacobianEvals,numInequalityConstraintJacobianEvals,'
    + 'numLagrangianHessianEvals,'
    + 'totalSeconds,convergedToOptimalFlag\n'
)
csvFile.flush()

def getNumbersInString(str):
    return re.findall(r"[-+]?(?:\d*\.*\d+)", str)


# Importing the data from CSV file
df = pandas.read_csv('data/data.csv')

for index in tqdm(range(NUM_PROBLEMS)):
    N = df.iloc[index].N
    alpha = df.iloc[index].alpha
    lb_y = df.iloc[index].lb_y
    ub_y = df.iloc[index].ub_y
    lb_u = df.iloc[index].lb_u
    ub_u = df.iloc[index].ub_u
    d_const = df.iloc[index].d_const
    target_profile_equation = df.iloc[index].target_profile_equation
    costIPOPT = df.iloc[index].cost
    problemType = 'diri'

    (y_desired_ipopt, y_solution_ipopt, terminalOutput) = utils.generateBoundaryDataFromIPOPT(
        problemType, N, alpha, lb_y, ub_y, lb_u, ub_u, d_const, target_profile_equation)

    cost = utils.costFunction2(
        y_desired_ipopt, y_solution_ipopt,
        lb_y, ub_y, lb_u, ub_u,
        alpha=alpha
    ).mean()

    # Extract info from 'terminalOutput':

    splits = terminalOutput.splitlines()

    if splits[-1].__contains__('EXIT: Optimal Solution Found.'):
        convergedToOptimalFlag = True
    elif splits[-1].__contains__('EXIT: Converged to a point of local infeasibility. Problem may be infeasible.'):
        convergedToOptimalFlag = False
    elif splits[-1].__contains__('EXIT: Maximum Number of Iterations Exceeded.'):
        convergedToOptimalFlag = False
    else:
        raise Exception('Something went wrong')

    splits = splits[-21:-1]

    numIterations = int(getNumbersInString(splits[0])[0])

    vals = getNumbersInString(splits[3])
    objectiveScaled = float(vals[0] + 'e' + vals[1])
    objectiveUnscaled = float(vals[2] + 'e' + vals[3])

    vals = getNumbersInString(splits[4])
    dualInfeasibilityScaled = float(vals[0] + 'e' + vals[1])
    dualInfeasibilityUnscaled = float(vals[2] + 'e' + vals[3])

    vals = getNumbersInString(splits[5])
    constraintViolationScaled = float(vals[0] + 'e' + vals[1])
    constraintViolationUnscaled = float(vals[2] + 'e' + vals[3])

    vals = getNumbersInString(splits[6])
    variableBoundViolationScaled = float(vals[0] + 'e' + vals[1])
    variableBoundViolationUnscaled = float(vals[2] + 'e' + vals[3])

    vals = getNumbersInString(splits[7])
    complementarityScaled = float(vals[0] + 'e' + vals[1])
    complementarityUnscaled = float(vals[2] + 'e' + vals[3])

    vals = getNumbersInString(splits[8])
    overallNLPErrorScaled = float(vals[0] + 'e' + vals[1])
    overallNLPErrorUnscaled = float(vals[2] + 'e' + vals[3])

    numObjectiveFunctionEvals = int(getNumbersInString(splits[11])[0])
    numObjectiveGradientEvals = int(getNumbersInString(splits[12])[0])
    numEqualityConstraintEvals = int(getNumbersInString(splits[13])[0])
    numInequalityConstraintEvals = int(getNumbersInString(splits[14])[0])
    numEqualityConstraintJacobianEvals = int(getNumbersInString(splits[15])[0])
    numInequalityConstraintJacobianEvals = int(getNumbersInString(splits[16])[0])
    numLagrangianHessianEvals = int(getNumbersInString(splits[17])[0])

    totalSeconds = float(getNumbersInString(splits[18])[0])

    # Write it all to the CSV file
    csvFile.write(
        f'{index},{cost},{numIterations},'
        + f'{objectiveScaled},{objectiveUnscaled},'
        + f'{dualInfeasibilityScaled},{dualInfeasibilityUnscaled},'
        + f'{constraintViolationScaled},{constraintViolationUnscaled},'
        + f'{variableBoundViolationScaled},{variableBoundViolationUnscaled},'
        + f'{complementarityScaled},{complementarityUnscaled},'
        + f'{overallNLPErrorScaled},{overallNLPErrorUnscaled},'
        + f'{numObjectiveFunctionEvals},{numObjectiveGradientEvals},'
        + f'{numEqualityConstraintEvals},{numInequalityConstraintEvals},'
        + f'{numEqualityConstraintJacobianEvals},{numInequalityConstraintJacobianEvals},'
        + f'{numLagrangianHessianEvals},'
        + f'{totalSeconds},{int(convergedToOptimalFlag)}\n'
    )
    csvFile.flush()
