# Any method aimed at coming up with a good initial guess should aim to beat
# the baseline set by:
# 1. Zero boundary conditions
# 2. Desired array boundary conditions.
# Let us save the baseline values for each problems so that experiments that we
# conduct in the future can easily verify how good they are.

import torch
import random
import pandas
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
OUTPUT_FILE_NAME = FOLDER_NAME + '/simpleStatisticalBaselines.csv'

csvFile = open(OUTPUT_FILE_NAME, 'a') # Appending as an insurance
csvFile.write(
    'i,costMean,costMedian\n'
)
csvFile.flush()

# Importing the data from CSV file
df = pandas.read_csv('data/data.csv')

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

    return N, array, bounds, d_const, u_init, alpha, costIPOPT


def getDConstMatrix(d_const, N):
    # Just memorizing to avoid recalculation everytime
    if not hasattr(getDConstMatrix, 'dConstMatrixMap'):
        getDConstMatrix.dConstMatrixMap = dict()
    ret = getDConstMatrix.dConstMatrixMap.get(N)
    if ret is None:
        ret = utils.solvePDE(
            torch.zeros((1, 4, N)),
            torch.tensor(-10),
            0
        )[:, 1:-1, 1:-1]
        getDConstMatrix.dConstMatrixMap[N] = ret
    return ret * (d_const.item() / -10)

for index in tqdm(range(NUM_PROBLEMS)):
    # Generate a problem
    N, array, bounds, d_const, u_init, alpha, costIPOPT = getEquationAt(index)

    # Define the loss function
    loss_fn = utils.CostFunctionLoss()

    output = array.clone()[:, 1:-1, 1:-1] - getDConstMatrix(d_const, N)
    output_mean = torch.zeros((1,4,N)) + output.mean()
    output_median = torch.zeros((1,4,N)) + output.median()

    loss_mean = loss_fn(
        output_mean,
        array, bounds, d_const, u_init, alpha
    )
    loss_median = loss_fn(
        output_median,
        array, bounds, d_const, u_init, alpha
    )

    # Log information
    csvFile.write(
        f'{index},{loss_mean.item()},{loss_median.item()}\n'
    )
    csvFile.flush()
