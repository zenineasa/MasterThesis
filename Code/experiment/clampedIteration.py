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
EARLY_CONVERGENCE_CHECK_STEPS = 100
MAX_REINFORCE_STEPS = 100
LEARNING_RATE_PATIENCE = 5
LEARNING_RATE_INIT = 1
LEARNING_RATE_MIN = 1e-3

FOLDER_NAME = 'experiment'
OUTPUT_FILE_NAME = FOLDER_NAME + '/clampedIteration.csv'

csvFile = open(OUTPUT_FILE_NAME, 'a') # Appending as an insurance
csvFile.write(
    'i,minNetCost,minNetCostIndex,firstIndexThatBeatIPOPT\n'
)
csvFile.flush()

# Importing the data from CSV file
df = pandas.read_csv('data/data.csv')

# Define neural network
class Net(torch.nn.Module):
    def __init__(self, N):
        super(Net, self).__init__()

        self.N = N

        # Defining the layers
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(0, 4*N),
        )

    def forward(self):
        # Pure reinforcement without any input; this would be fun!
        return self.seq(torch.empty(0)).reshape((1, 4, self.N))

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

# Clamped iterative solver
def clampedIterativeSolver(N, x, bounds, d_const):
    # TODO: We can improve the performance by incorporating information from 'sourcingTerm.py'
    # to reduce the number of times 'd_constTimesOneBySquaredN' is subtracted.
    d_constTimesOneBySquaredN = d_const / (N ** 2)

    # Iterate
    xPrev = x.clone()
    for iter in range(10000):
        x[:, 1:-1, 1:-1] = 0.25 * (
            (x[:, 2:, 1:-1] + x[:, :-2, 1:-1] + x[:, 1:-1, 2:] + x[:, 1:-1, :-2])
            - d_constTimesOneBySquaredN
        )

        # Clamp the values based on domain limits
        x = x.clamp(min=bounds[0][0], max=bounds[0][1])

        if iter % EARLY_CONVERGENCE_CHECK_STEPS == 0:
            if torch.allclose(x, xPrev, rtol=1e-04):
                break
            xPrev = x.clone()

    # The values at the boundary are not modified, but the values slightly
    # within are, in the iterations. Let's use these as a starting point.
    x = torch.cat((
        x[:, 1, 1:-1], x[:, 1:-1, -2], x[:, -2, 1:-1], x[:, 1:-1, 1]
    )).unsqueeze(0)
    x = x.clamp(min=bounds[0][2], max=bounds[0][3])
    return x

def reinforcementWithAdam(initBVal, N, array, bounds, d_const, u_init, alpha):
    # Create a new model of size N
    model = Net(N)

    # Define the loss function
    loss_fn = utils.CostFunctionLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_INIT, weight_decay=0.0001)

    # NOTE: We are not training here. We are evaluating and providing the
    # cost as a reward for reinforcement.
    networkCostArr = []
    model.train()
    for epoch in range(MAX_REINFORCE_STEPS):
        # Evaluate the network
        outputs = initBVal + model()

        # Calculate the loss
        loss = loss_fn(outputs, array, bounds, d_const, u_init, alpha)
        loss.backward()

        networkCostArr.append(loss.item())
        #print(loss.item())

        # Store the best model
        if min(networkCostArr) == loss.item():
            best_model = copy.deepcopy(model)

        # Now, step the optimizer
        optimizer.step()
        optimizer.zero_grad()

        if epoch > LEARNING_RATE_PATIENCE:
            if networkCostArr[-LEARNING_RATE_PATIENCE] <= min(networkCostArr[-LEARNING_RATE_PATIENCE + 1:]):
                #print('Changing learning rate and continuing from the best model...')
                # Use the best model till this point for further training
                model = copy.deepcopy(best_model)

                # Calculate new learning rate
                newLR = optimizer.param_groups[0]['lr'] * 0.5
                if newLR < LEARNING_RATE_MIN:
                    break

                # Reinitialize the optimizer with a different learning rate
                optimizer = torch.optim.Adam(model.parameters(), lr=newLR, weight_decay=0.0001)

    return (initBVal + best_model().detach(), networkCostArr)

for index in tqdm(range(NUM_PROBLEMS_TO_TEST)):
    # Generate a problem
    N, array, bounds, d_const, u_init, alpha, costIPOPT = getEquationAt(index)

    # Solve with clamped iterative solver
    outputs1 = clampedIterativeSolver(N, array.clone(), bounds, d_const)

    # Now, use this for reinforcement with Adam
    (outputs2, networkCostArr) = reinforcementWithAdam(outputs1, N, array, bounds, d_const, u_init, alpha)

    # Log information
    minNetCost = min(networkCostArr)
    minNetCostIndex = networkCostArr.index(minNetCost)

    resultsBetterThanIPOPT = [x for x in enumerate(networkCostArr) if x[1] < costIPOPT]
    if resultsBetterThanIPOPT:
        firstIndexThatBeatIPOPT = resultsBetterThanIPOPT[0][0]
    else:
        firstIndexThatBeatIPOPT = -1

    csvFile.write(
        f'{index},{minNetCost},{minNetCostIndex},{firstIndexThatBeatIPOPT}\n'
    )
    csvFile.flush()
