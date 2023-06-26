# "TIME" part of the experiment was conducted on:
# Ubuntu 22.04.02 LTS
# AMD Ryzen 5 2500U with Radeon Vega Mobile Gfx

import torch
import random
import torch.profiler.profiler as profiler
from networks import OptimizerNetwork, InitGuessNetwork
import extraUtilities as eUtils

import time

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
NETWORK_NAME_INIT_GUESS = 'initGuessNetwork'
NETWORK_NAME_OPTIMIZER = 'optimizerNetwork'

FOLDER_NAME = 'FinalAttempt'
MODELS_FOLDER_NAME = FOLDER_NAME + '/models'
CSV_FOLDER_NAME = FOLDER_NAME + '/csv'
INIT_GUESS_MODEL_PATH = MODELS_FOLDER_NAME + '/initGuessNetwork.pt'
OPTIMIZER_MODEL_PATH = MODELS_FOLDER_NAME + '/optimizerNetwork.pt'

OPTION = "LARGEN" # "FLOPS", "TIME", "LARGEN"

# Define the loss
loss_fn = utils.CostFunctionLoss()

# Input generator: Just return the same problem with different domain
# sizes based on the input parameter. Both of the following functions
# returns information for the same problem.
def getEquation(N, index=0):
    #N = df.iloc[index].N
    alpha = eUtils.df.iloc[index].alpha
    lb_y = eUtils.df.iloc[index].lb_y
    ub_y = eUtils.df.iloc[index].ub_y
    lb_u = eUtils.df.iloc[index].lb_u
    ub_u = eUtils.df.iloc[index].ub_u
    d_const = eUtils.df.iloc[index].d_const
    target_profile_equation = eUtils.df.iloc[index].target_profile_equation

    d_const = torch.tensor([[d_const]], dtype=torch.float)
    bounds = torch.tensor([[lb_y, ub_y, lb_u, ub_u]])
    array = utils.generateProfile(target_profile_equation, N)
    u_init = 0

    return array, bounds, d_const, u_init, alpha
def getEquationForIPOPT(N, index=0):
    #N = df.iloc[index].N
    problemType = 'diri'
    alpha = eUtils.df.iloc[index].alpha
    lb_y = eUtils.df.iloc[index].lb_y
    ub_y = eUtils.df.iloc[index].ub_y
    lb_u = eUtils.df.iloc[index].lb_u
    ub_u = eUtils.df.iloc[index].ub_u
    d_const = eUtils.df.iloc[index].d_const
    target_profile_equation = eUtils.df.iloc[index].target_profile_equation

    return problemType, N, alpha, lb_y, ub_y, lb_u, ub_u, d_const, target_profile_equation

# Load initial models
initGuessModel = torch.load(INIT_GUESS_MODEL_PATH)
initGuessModel.eval()
optimizerModel = torch.load(OPTIMIZER_MODEL_PATH)
optimizerModel.eval()

if __name__ == "__main__":
    if(OPTION == "FLOPS"):
        OUTPUT_FILE_NAME_INIT_GUESS = CSV_FOLDER_NAME + '/' + NETWORK_NAME_INIT_GUESS + '_FLOP.log'
        OUTPUT_FILE_NAME_OPTIMIZER = CSV_FOLDER_NAME + '/' + NETWORK_NAME_OPTIMIZER + '_FLOP.log'

        # Opening the files to log to
        os.system(f'mkdir -p {CSV_FOLDER_NAME}')
        logFileInitGuess = open(OUTPUT_FILE_NAME_INIT_GUESS, 'a')
        logFileOptimizer = open(OUTPUT_FILE_NAME_OPTIMIZER, 'a')

        for num_steps in range(1, 5):
            for N in range(10, 101, 10):
                # The problem that we are solving does not matter;
                # we just need to try it for different values of N
                array, bounds, d_const, u_init, alpha = getEquation(N)

                # Run the profiler on initial guess network
                with profiler.profile(with_flops=True) as prof:
                    guessBVal = initGuessModel(array[:, 1:-1, 1:-1], bounds, d_const)
                # Log information
                info = "For N = " + str(N) + " | Num optimizer steps = " + str(num_steps) + "\n\n"
                info += prof.key_averages().table()
                info += "\n\n\n"
                logFileInitGuess.write(info)
                logFileInitGuess.flush()

                # Run the profiler on optimizer network
                with profiler.profile(with_flops=True) as prof:
                    x, loss_val = eUtils.solveWithOptimizerModel(
                        N, optimizerModel, guessBVal, array, bounds, d_const, u_init, alpha, loss_fn,
                        num_steps
                    )
                # Log information
                info = "For N = " + str(N) + " | Num optimizer steps = " + str(num_steps) + "\n\n"
                info += prof.key_averages().table()
                info += "\n\n\n"
                logFileOptimizer.write(info)
                logFileOptimizer.flush()

    elif(OPTION == "TIME"):
        NUM_STEPS = 32
        OUTPUT_FILE_NAME = CSV_FOLDER_NAME + '/' + NETWORK_NAME_OPTIMIZER + '_TIME.csv'

        # Opening the files to log to
        os.system(f'mkdir -p {CSV_FOLDER_NAME}')
        csvFile = open(OUTPUT_FILE_NAME, 'a')
        csvFile.write('N,minInitGuessTime,maxInitGuessTime,meanInitGuessTime,minOptimizerTime,maxOptimizerTime,meanOptimizerTime\n')
        csvFile.flush()

        for N in range(10, 101, 5):
            initGuessTimes = []
            optimizerTimes = []
            for _ in range(5):
                # The problem that we are solving does not matter;
                # we just need to try it for different values of N
                array, bounds, d_const, u_init, alpha = getEquation(N)

                # Check the time taken to run the initial guess network
                start_time = time.time()
                guessBVal = initGuessModel(array[:, 1:-1, 1:-1], bounds, d_const)
                initGuessTime = time.time() - start_time

                # Check the time taken to run the optimizer network
                start_time = time.time()
                x, loss_val = eUtils.solveWithOptimizerModel(
                    N, optimizerModel, guessBVal, array, bounds, d_const, u_init, alpha, loss_fn,
                    NUM_STEPS
                )
                optimizerTime = time.time() - start_time

                initGuessTimes.append(initGuessTime)
                optimizerTimes.append(optimizerTime)

            # Log information
            meanInitGuessTime = sum(initGuessTimes) / len(initGuessTimes)
            meanOptimizerTime = sum(optimizerTimes) / len(optimizerTimes)
            csvFile.write(f'{N},{min(initGuessTimes)},{max(initGuessTimes)},{meanInitGuessTime},{min(optimizerTimes)},{max(optimizerTimes)},{meanOptimizerTime}\n')
            csvFile.flush()

    elif(OPTION == "LARGEN"):
        # Report accuracy for larger values of N
        NUM_STEPS = 32
        OUTPUT_FILE_NAME = CSV_FOLDER_NAME + '/' + NETWORK_NAME_OPTIMIZER + '_LARGEN.csv'

        # Opening the files to log to
        os.system(f'mkdir -p {CSV_FOLDER_NAME}')
        csvFile = open(OUTPUT_FILE_NAME, 'a')
        csvFile.write('i,N,costMethod,costIPOPT\n')
        csvFile.flush()

        for index in [0, 17, 54, 70]:
            for N in list(range(100, 201, 10)) + list(range(300, 1001, 100)):
                # Getting a problem for which we had found feasible solution earlier
                array, bounds, d_const, u_init, alpha = getEquation(N, index)

                # Get the initial guess values
                guessBVal = initGuessModel(array[:, 1:-1, 1:-1], bounds, d_const)

                # Solve with optimizer network
                x, loss_val = eUtils.solveWithOptimizerModel(
                    N, optimizerModel, guessBVal, array, bounds, d_const, u_init, alpha, loss_fn,
                    NUM_STEPS
                )

                # Solve with IPOPT and calculate the cost
                problemType, N, alpha, lb_y, ub_y, lb_u, ub_u, d_const, target_profile_equation = getEquationForIPOPT(N, index)
                (y_desired_ipopt, y_solution_ipopt, terminalOutput) = utils.generateBoundaryDataFromIPOPT(
                    problemType, N, alpha, lb_y, ub_y, lb_u, ub_u, d_const, target_profile_equation
                )
                costIPOPT = utils.costFunction2(
                    y_desired_ipopt, y_solution_ipopt,
                    lb_y, ub_y, lb_u, ub_u,
                    alpha=alpha
                ).mean()

                # Report the cost values from both the methods
                csvFile.write(f'{index},{N},{loss_val},{costIPOPT}\n')
                csvFile.flush()
