# Before you run this, uncomment the commented out code under
# 'OptimizerNetwork' in 'Network.py' file.

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

NUM_PROBLEMS_TO_TEST = 100
REINFORCE_STEPS = 32

FOLDER_NAME = 'FinalAttempt'
MODELS_FOLDER_NAME = FOLDER_NAME + '/models'
CSV_FOLDER_NAME = FOLDER_NAME + '/csv'
INIT_GUESS_MODEL_PATH = MODELS_FOLDER_NAME + '/initGuessNetwork.pt'
OPTIMIZER_MODEL_PATH = MODELS_FOLDER_NAME + '/optimizerNetwork.pt'
OUTPUT_FILE_NAME = CSV_FOLDER_NAME + '/' + NETWORK_NAME + '_Contributions.csv'

# Define the loss
loss_fn = utils.CostFunctionLoss()

# Load initial models
initGuessModel = torch.load(INIT_GUESS_MODEL_PATH)
initGuessModel.eval()
optimizerModel = torch.load(OPTIMIZER_MODEL_PATH)
optimizerModel.eval()

if __name__ == "__main__":
        # Opening the files to log to
        os.system(f'mkdir -p {CSV_FOLDER_NAME}')
        csvFile = open(OUTPUT_FILE_NAME, 'a')
        csvFile.write('iter,N,adam,rms,net,diffNet\n')

        optimizerModel.csvFile = csvFile

        for _ in tqdm(range(NUM_PROBLEMS_TO_TEST)):
            # Get a random problem
            index, N, array, bounds, d_const, u_init, alpha = eUtils.generateInputToNetwork(
                mode='training', useIPOPTConvergedProblems=True
            )

            # Run the profiler on initial guess network
            guessBVal = initGuessModel(array[:, 1:-1, 1:-1], bounds, d_const)

            # Run the profiler on optimizer network
            x, loss_val = eUtils.solveWithOptimizerModel(
                N, optimizerModel, guessBVal, array, bounds, d_const, u_init, alpha, loss_fn,
                REINFORCE_STEPS
            )
