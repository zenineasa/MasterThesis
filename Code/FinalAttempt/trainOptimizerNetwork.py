import torch
import random
from tqdm import tqdm
from networks import OptimizerNetwork, InitGuessNetwork
import extraUtilities as eUtils

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
NETWORK_NAME = 'optimizerNetwork'
TRAINING_MODE = False # False means testing mode

NUM_EPOCHS = 160
NUM_GRADIENT_ACCUMULATION_STEPS = 8
NUM_STEPS_TO_RUN_VALIDATION = 8
NUM_PROBLEMS_TO_VALIDATE_WITH = 8
REINFORCE_STEPS = 8

FOLDER_NAME = 'FinalAttempt'
MODELS_FOLDER_NAME = FOLDER_NAME + '/models'
CSV_FOLDER_NAME = FOLDER_NAME + '/csv'
INIT_GUESS_MODEL_PATH = MODELS_FOLDER_NAME + '/initGuessNetwork.pt'
BEST_MODEL_PATH = MODELS_FOLDER_NAME + '/' + NETWORK_NAME + '.pt'
OUTPUT_FILE_NAME_TRAIN = CSV_FOLDER_NAME + '/' + NETWORK_NAME + '_train.csv'
OUTPUT_FILE_NAME_VALID = CSV_FOLDER_NAME + '/' + NETWORK_NAME + '_valid.csv'
OUTPUT_FILE_NAME_TEST = CSV_FOLDER_NAME + '/' + NETWORK_NAME + '_test.csv'

# Define the loss
loss_fn = utils.CostFunctionLoss()

# Load initial guess model
initGuessModel = torch.load(INIT_GUESS_MODEL_PATH)
initGuessModel.eval()

if __name__ == "__main__":
    if TRAINING_MODE:
        # Opening the files to log to
        os.system(f'mkdir -p {MODELS_FOLDER_NAME}')
        os.system(f'mkdir -p {CSV_FOLDER_NAME}')
        csvFileTrain = open(OUTPUT_FILE_NAME_TRAIN, 'a')
        csvFileTrain.write('epoch,trainingLoss\n')
        csvFileTrain.flush()
        csvFileValid = open(OUTPUT_FILE_NAME_VALID, 'a')
        csvFileValid.write('epoch,validationLoss\n')
        csvFileValid.flush()

        model = OptimizerNetwork()
        optimizer = torch.optim.Adam(model.parameters())

        # Variable to store the best loss encountered; helps in saving the best network
        lowestValidationLossDetected = float('inf')

        for epoch in tqdm(range(NUM_EPOCHS)):
            networkCostArr = []

            # Accumulate gradients and step the optimizer only after some number of steps
            model.train()
            optimizer.zero_grad()
            for _ in range(NUM_GRADIENT_ACCUMULATION_STEPS):
                # Get a random problem
                index, N, array, bounds, d_const, u_init, alpha = eUtils.generateInputToNetwork(
                    mode='training', useIPOPTConvergedProblems=True
                )

                # Get the initial guess values
                guessBVal = initGuessModel(array[:, 1:-1, 1:-1], bounds, d_const)

                # Now, run it over the optimizer network
                x, loss_val = eUtils.solveWithOptimizerModel(
                    N, model, guessBVal, array, bounds, d_const, u_init, alpha, loss_fn,
                    REINFORCE_STEPS
                )

                # Calculate the loss
                loss = loss_fn(guessBVal + x, array, bounds, d_const, u_init, alpha)
                loss.backward()

                networkCostArr.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()

            # Mean cost
            meanNetworkCost = sum(networkCostArr)/NUM_GRADIENT_ACCUMULATION_STEPS

            csvFileTrain.write(f'{epoch},{meanNetworkCost}\n')
            csvFileTrain.flush()

            # Validation
            if epoch % NUM_STEPS_TO_RUN_VALIDATION == 0:
                networkCostArr = []

                model.eval()
                #with torch.no_grad():
                for _ in range(NUM_PROBLEMS_TO_VALIDATE_WITH):
                    # Get a random problem
                    index, N, array, bounds, d_const, u_init, alpha = eUtils.generateInputToNetwork(
                        mode='validation', useIPOPTConvergedProblems=True
                    )

                    # Get the initial guess values
                    guessBVal = initGuessModel(array[:, 1:-1, 1:-1], bounds, d_const)

                    # Now, run it over the optimizer network
                    x, loss_val = eUtils.solveWithOptimizerModel(
                        N, model, guessBVal, array, bounds, d_const, u_init, alpha, loss_fn,
                        REINFORCE_STEPS
                    )

                    # Calculate the loss
                    loss = loss_fn(guessBVal + x, array, bounds, d_const, u_init, alpha)

                    networkCostArr.append(loss.item())

                # Mean cost
                meanNetworkCost = sum(networkCostArr)/NUM_PROBLEMS_TO_VALIDATE_WITH

                csvFileValid.write(f'{epoch},{meanNetworkCost}\n')
                csvFileValid.flush()

                # Save the network
                if meanNetworkCost < lowestValidationLossDetected:
                    print(f'Lowest validation loss detected after epoch {epoch}; saving the model\n')
                    lowestValidationLossDetected = meanNetworkCost
                    torch.save(model, BEST_MODEL_PATH)

    else: # Testing
        # Opening the files to log to
        os.system(f'mkdir -p {CSV_FOLDER_NAME}')
        csvFileTest = open(OUTPUT_FILE_NAME_TEST, 'a')
        csvFileTest.write('i,networkCost\n')
        csvFileTest.flush()

        model = torch.load(BEST_MODEL_PATH)

        model.eval()
        #with torch.no_grad():
        for index in tqdm(range(len(eUtils.df))):
            # Generate a problem
            N, array, bounds, d_const, u_init, alpha = eUtils.getEquationAt(index)

            # Get the initial guess values
            guessBVal = initGuessModel(array[:, 1:-1, 1:-1], bounds, d_const)

            # Now, run it over the optimizer network
            x, loss_val = eUtils.solveWithOptimizerModel(
                N, model, guessBVal, array, bounds, d_const, u_init, alpha, loss_fn,
                REINFORCE_STEPS
            )

            # Log information
            csvFileTest.write(
                f'{index},{loss_val}\n'
            )
            csvFileTest.flush()
