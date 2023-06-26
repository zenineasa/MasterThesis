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
REINFORCE_STEPS = 32

FOLDER_NAME = 'FinalAttempt'
MODELS_FOLDER_NAME = FOLDER_NAME + '/models'
CSV_FOLDER_NAME = FOLDER_NAME + '/csv'
INIT_GUESS_MODEL_PATH = MODELS_FOLDER_NAME + '/initGuessNetwork.pt'
OPTIMIZER_MODEL_PATH = MODELS_FOLDER_NAME + '/optimizerNetwork.pt'
OUTPUT_FILE_NAME = CSV_FOLDER_NAME + '/' + NETWORK_NAME + '_extended_test.csv'

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
    csvFileTest = open(OUTPUT_FILE_NAME, 'a')
    csvFileTest.write(
        'i,minNetCost,minNetCostIter,minNetMaxAbsCV,minNetMeanSquaredCV,'
        + 'minCostWithoutCV,minCostWithoutCVIter,'
        + 'beatIPOPTIter,beatIPOPTWithZeroViolationIter\n'
    )
    csvFileTest.flush()

    for index in tqdm(range(len(eUtils.df))):
        # Generate a problem
        N, array, bounds, d_const, u_init, alpha = eUtils.getEquationAt(index)

        # Get the initial guess values
        guessBVal = initGuessModel(array[:, 1:-1, 1:-1], bounds, d_const)

        # Get IPOPT cost
        costIPOPT = eUtils.ipoptInfoDf.iloc[index].cost

        # Solve and collect statistical information
        (
            minNetCost, minNetCostIter, minNetMaxAbsCV, minNetMeanSquaredCV,
            minCostWithoutCV, minCostWithoutCVIter,
            beatIPOPTIter, beatIPOPTWithZeroViolationIter
        ) = eUtils.solveWithOptimizerModelAndCollectStatistics(
            N, optimizerModel, guessBVal, array, bounds, d_const, u_init, alpha, loss_fn,
            REINFORCE_STEPS, costIPOPT
        )

        # Log information
        str = (
            f'{index},{minNetCost},{minNetCostIter},{minNetMaxAbsCV},{minNetMeanSquaredCV},'
            + f'{minCostWithoutCV},{minCostWithoutCVIter},'
            + f'{beatIPOPTIter},{beatIPOPTWithZeroViolationIter}\n'
        )
        csvFileTest.write(str)
        csvFileTest.flush()
