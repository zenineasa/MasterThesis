import torch
import random
import pandas
import copy
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
NUM_PROBLEMS_TO_TEST = 400

FOLDER_NAME = 'experiment'
OUTPUT_FILE_NAME = FOLDER_NAME + '/boundaryFromArray.csv'

csvFile = open(OUTPUT_FILE_NAME, 'a') # Appending as an insurance
csvFile.write(
    'i,cost\n'
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

    d_const = torch.tensor([[d_const]], dtype=torch.float)
    bounds = torch.tensor([[lb_y, ub_y, lb_u, ub_u]])
    array = utils.generateProfile(target_profile_equation, N)
    u_init = 0

    return N, array, bounds, d_const, u_init, alpha

# Just get the boundary values from the 'array' (which is 'x' here) and clamp
# the values to the upper and lower limits of the boundaries.
def getClampedBoundaryValues(x, bounds):
    x = torch.cat((
        x[:, 0, 1:-1], x[:, 1:-1, -1], x[:, -1, 1:-1], x[:, 1:-1, 0]
    )).unsqueeze(0)
    x = x.clamp(min=bounds[0][0], max=bounds[0][1])
    x = x.clamp(min=bounds[0][2], max=bounds[0][3])
    return x

for index in tqdm(range(NUM_PROBLEMS_TO_TEST)):
    # Generate a problem
    N, array, bounds, d_const, u_init, alpha = getEquationAt(index)

    # Define the loss function
    loss_fn = utils.CostFunctionLoss()

    # Solve with boundary clamping
    outputs = getClampedBoundaryValues(array.clone(), bounds)
    loss = loss_fn(outputs, array, bounds, d_const, u_init, alpha)

    csvFile.write(
        f'{index},{loss.item()}\n'
    )
    csvFile.flush()
