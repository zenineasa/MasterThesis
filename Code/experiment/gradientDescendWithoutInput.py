import torch
import random
import pandas
import copy

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
MAX_REINFORCE_STEPS = 100
LEARNING_RATE_PATIENCE = 5
LEARNING_RATE_INIT = 1
#LEARNING_RATE_LATER = 0.001

FOLDER_NAME = 'experiment'
OUTPUT_FILE_NAME = FOLDER_NAME + '/gradientDescendWithoutInput.csv'

csvFile = open(OUTPUT_FILE_NAME, 'a') # Appending as an insurance
csvFile.write(
    'i,costIPOPT,minNetCost,minNetCostIndex,maxNetCost,maxNetCostIndex\n'
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
            #torch.nn.ReLU(),
            #torch.nn.Linear(4*N, 4*N),
            #torch.nn.ReLU(),
            #torch.nn.Linear(4*N, 4*N),
        )

    def forward(self):
        # Pure reinforcement without any input; this would be fun!
        return self.seq(torch.empty(0)).reshape((1, 4, self.N))

    def getDConstMatrix(self, d_const, N):
        zeroPDEVal = self.dConstMatrixMap.get(frozenset({d_const.item(), N}))

        if zeroPDEVal is None:
            boundaryValues = torch.zeros((1, 4, N))
            zeroPDEVal = utils.solvePDE(boundaryValues, d_const, u_init)[:, 1:-1, 1:-1]
            self.dConstMatrixMap[frozenset({d_const.item(), N})] = zeroPDEVal

        return zeroPDEVal

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


for index in range(NUM_PROBLEMS_TO_TEST):
    # Generate a problem
    N, array, bounds, d_const, u_init, alpha, costIPOPT = getEquationAt(index)

    # Create a new model of size N
    model = Net(N)

    # Define the loss function
    loss_fn = utils.CostFunctionLoss()

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE_INIT, momentum=0.9, weight_decay=0.0001)
    learningRateChangedFlag = False

    print(f'\n\nNew problem\nLeast cost by IPOPT: {costIPOPT}\n')

    networkCostArr = []

    # NOTE: We are not training here. We are evaluating and providing the
    # cost as a reward for reinforcement.
    model.train()
    for epoch in range(MAX_REINFORCE_STEPS):
        # Evaluate the network
        outputs = model()

        # Calculate the loss
        loss = loss_fn(outputs, array, bounds, d_const, u_init, alpha)
        loss.backward()

        networkCostArr.append(loss.item())
        print(loss.item())

        # Store the best model
        if min(networkCostArr) == loss.item():
            best_model = copy.deepcopy(model)

        # Now, step the optimizer
        optimizer.step()
        optimizer.zero_grad()

        if epoch > LEARNING_RATE_PATIENCE and not learningRateChangedFlag:
            if networkCostArr[-LEARNING_RATE_PATIENCE] <= min(networkCostArr[-LEARNING_RATE_PATIENCE + 1:]):
                print('Changing learning rate and continuing from the best model...')
                # Use the best model till this point for further training
                model = copy.deepcopy(best_model)

                # Reinitialize the optimizer with a different learning rate
                newLR = optimizer.param_groups[0]['lr'] * 0.5
                optimizer = torch.optim.SGD(model.parameters(), lr=newLR, momentum=0.9, weight_decay=0.0001)
                learningRateChangedFlag = True

    # Log the information
    minNetCost = min(networkCostArr)
    minNetCostIndex = networkCostArr.index(minNetCost)
    maxNetCost = max(networkCostArr)
    maxNetCostIndex = networkCostArr.index(maxNetCost)

    str = (f'Train || Min. network cost: {minNetCost}, IPOPT Cost: {costIPOPT}\n')
    print(str)

    csvFile.write(
        f'{index},{costIPOPT},{minNetCost},{minNetCostIndex},{maxNetCost},{maxNetCostIndex}\n'
    )
    csvFile.flush()
