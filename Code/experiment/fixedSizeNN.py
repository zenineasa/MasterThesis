import torch
import torch.nn as nn
import torch.optim as optim
import pandas

# importing the utilities module from the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utilities as utils

# Seeds for reproducibility
#random.seed(0)
#np.random.seed(0)
torch.manual_seed(0)

# Hyperparams
ARRAY_SIZE = 64
NUM_EPOCHS = 100
NUM_GRADIENT_ACCUMMULATION_STEPS = 32
NUM_STEPS_TO_RUN_VALIDATION = 4

# Importing the data from CSV file
df = pandas.read_csv('data/data.csv')

# Divide this into train and validation sets
trainingData = df.sample(frac=0.8,random_state=100)
validationData = df.drop(trainingData.index)

# Define neural network
class Net(nn.Module):
    def __init__(self, array_size):
        super(Net, self).__init__()
        self.array_size = array_size
        self.bounds_size = 4
        self.const_term_size = 1 # just to incorporate d_const for now

        self.conv_out_size = 1 * self.array_size ** 2

        # Convolutional layers for array
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(self.conv_out_size, 64)

        # Fully connected layers for bounds
        self.fc3 = nn.Linear(self.bounds_size, 32)
        self.fc4 = nn.Linear(32, 64)

        # Full connected layers for d_const
        self.fc5 = nn.Linear(self.const_term_size, 32)
        self.fc6 = nn.Linear(32, 64)

        # Final fully connected layer for output
        self.fc7 = nn.Linear(64 + 64 + 64, 64 + 64 + 64)
        self.fc8 = nn.Linear(64 + 64 + 64, 4 * self.array_size)

    def forward(self, x, bounds, d_const):
        # Process array
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, self.conv_out_size)
        x = torch.relu(self.fc1(x))

        # Process bounds
        bounds = torch.clamp(bounds, min=-1e3, max=1e3)
        y = torch.relu(self.fc3(bounds))
        y = torch.relu(self.fc4(y))

        # Process d_const
        z = torch.relu(self.fc5(d_const))
        z = torch.relu(self.fc6(z))

        # Concatenate processed array and bounds
        a = torch.cat((x, y, z), dim=1)

        # Output layer
        a = torch.relu(self.fc7(a))
        a = self.fc8(a)
        a = a.view(-1, 4, self.array_size)
        return a

# Input generator
def generateInputToNetwork(trainMode=True):
    # Sample a problem from equationsCSV
    if trainMode:
        equation = trainingData.sample(n=1)
    else: # Validation mode
        equation = validationData.sample(n=1)

    N = ARRAY_SIZE # overriding
    alpha = equation.alpha.item()
    lb_y = equation.lb_y.item()
    ub_y = equation.ub_y.item()
    lb_u = equation.lb_u.item()
    ub_u = equation.ub_u.item()
    d_const = equation.d_const.item()
    target_profile_equation = equation.target_profile_equation.item()
    costIOPT = equation.cost.item()

    d_const = torch.tensor([[d_const]], dtype=torch.float)
    bounds = torch.tensor([[lb_y, ub_y, lb_u, ub_u]])
    array = utils.generateProfile(target_profile_equation, N)
    u_init = 0

    return array, bounds, d_const, u_init, alpha, costIOPT

model = Net(array_size=ARRAY_SIZE)

# Define the loss function and optimizer
loss_fn = utils.CostFunctionLoss()
optimizer = optim.Adam(model.parameters())

# Loss arrays
trainLossArr = []
validLossArr = []

trainIPOPTCostArr = []
validIPOPTCostArr = []

# Training
for epoch in range(NUM_EPOCHS):
    # Accumulate gradients and step the optimizer only after some number of steps
    networkCostArr = []
    ipoptCostArr = []
    model.train()
    optimizer.zero_grad()
    for _ in range(NUM_GRADIENT_ACCUMMULATION_STEPS):
        # Get a random problem
        array, bounds, d_const, u_init, alpha, costIOPT = generateInputToNetwork(trainMode=True)

        # Evaluate the network
        outputs = model(array[:, 1:-1, 1:-1], bounds, d_const)

        # Onnx export
        '''
        torch.onnx.export(model,
            args=(array[:, 1:-1, 1:-1], bounds, d_const),
            f="fixedSizeNN.onnx",
            input_names=["Desired domain values", "Bounds", "Sourcing term"],
            output_names=["Boundary values"])
        '''
        # Open this in Netron app to visualize the network:
        # https://netron.app/


        # Calculate the loss
        loss = loss_fn(outputs, array, bounds, d_const, u_init, alpha)
        loss.backward()

        networkCostArr.append(loss.item())
        ipoptCostArr.append(costIOPT)

    # Step the optimizer
    optimizer.step()

    # Report average loss
    trainLossArr.append(sum(networkCostArr)/NUM_GRADIENT_ACCUMMULATION_STEPS)
    trainIPOPTCostArr.append(sum(ipoptCostArr)/NUM_GRADIENT_ACCUMMULATION_STEPS)
    print(f'Batch training loss: {trainLossArr[-1]}')
    print(f'Batch training IPOPT loss: {trainIPOPTCostArr[-1]}')

    # Validation
    if epoch % NUM_STEPS_TO_RUN_VALIDATION == 0:
        networkCostArr = []
        ipoptCostArr = []

        model.eval()
        with torch.no_grad(): # just to reduce memory consumption
            for _ in range(NUM_GRADIENT_ACCUMMULATION_STEPS):
                # Get a random problem
                array, bounds, d_const, u_init, alpha, costIOPT = generateInputToNetwork(trainMode=False)

                # Evaluate the network
                outputs = model(array[:, 1:-1, 1:-1], bounds, d_const)

                # Calculate the loss
                loss = loss_fn(outputs, array, bounds, d_const, u_init, alpha)

                networkCostArr.append(loss.item())
                ipoptCostArr.append(costIOPT)

            # Report average loss
            validLossArr.append(sum(networkCostArr)/NUM_GRADIENT_ACCUMMULATION_STEPS)
            validIPOPTCostArr.append(sum(ipoptCostArr)/NUM_GRADIENT_ACCUMMULATION_STEPS)
            print(f'Batch validation loss: {validLossArr[-1]}')
            print(f'Batch training IPOPT loss: {validIPOPTCostArr[-1]}')

# Summarize training and validation loss
print('Training loss array:')
print(trainLossArr)
print('Training IPOPT cost array:')
print(trainIPOPTCostArr)
print('Validation loss array:')
print(validLossArr)
print('Validation IPOPT cost array:')
print(validIPOPTCostArr)
