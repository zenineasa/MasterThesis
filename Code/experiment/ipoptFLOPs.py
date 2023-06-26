# This experiment was conducted on:
# Ubuntu 22.04.02 LTS
# AMD Ryzen 5 2500U with Radeon Vega Mobile Gfx

# We need to somehow get the values for around 32 iterations
# Just have N between 10 and 100 with a difference of 5.

import os
import random
import torch
import pandas
import re
from tqdm import tqdm

# importing the utilities module from the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utilities as utils

# Seeds for reproducibility
random.seed(0)
#np.random.seed(0)
torch.manual_seed(0)

# Hyperparams
NUM_PROBLEMS = 10000

FOLDER_NAME = 'experiment'
OUTPUT_FILE_NAME = FOLDER_NAME + '/ipoptFLOPs.csv'

csvFile = open(OUTPUT_FILE_NAME, 'a') # Appending as an insurance
csvFile.write(
    'N,totalSeconds,numIterations,'
    + 'dp_add_sub_flops,dp_div_flops,dp_mult_add_flops,dp_mult_flops,'
    + 'sp_add_sub_flops,sp_div_flops,sp_mult_add_flops,sp_mult_flops\n'
)
csvFile.flush()

def getNumbersInString(str):
    return re.findall(r"[-+]?(?:\d*\.*\d+)", str)

# Importing the data from CSV file
df = pandas.read_csv('data/data.csv')

for N in tqdm(range(10, 101, 5)):
    for index in range(NUM_PROBLEMS):
        #N = df.iloc[index].N
        alpha = df.iloc[index].alpha
        lb_y = df.iloc[index].lb_y
        ub_y = df.iloc[index].ub_y
        lb_u = df.iloc[index].lb_u
        ub_u = df.iloc[index].ub_u
        d_const = df.iloc[index].d_const
        target_profile_equation = df.iloc[index].target_profile_equation
        costIPOPT = df.iloc[index].cost
        problemType = 'diri'

        (y_desired_ipopt, y_solution_ipopt, terminalOutput, perfOutput) = utils.generateBoundaryDataFromIPOPT(
            problemType, N, alpha, lb_y, ub_y, lb_u, ub_u, d_const, target_profile_equation, usePerf=True)

        cost = utils.costFunction2(
            y_desired_ipopt, y_solution_ipopt,
            lb_y, ub_y, lb_u, ub_u,
            alpha=alpha
        ).mean()

        # Extract info from 'terminalOutput':
        splits = terminalOutput.splitlines()
        splits = splits[-21:-1]

        numIterations = int(getNumbersInString(splits[0])[0])
        totalSeconds = float(getNumbersInString(splits[18])[0])

        # Extract info from 'perfOutput':
        splits = perfOutput.splitlines() # splits[5:13] contains the info we need

        dp_add_sub_flops = ''.join(getNumbersInString(splits[5])[:-1])
        dp_div_flops = ''.join(getNumbersInString(splits[6])[:-1])
        dp_mult_add_flops = ''.join(getNumbersInString(splits[7])[:-1])
        dp_mult_flops = ''.join(getNumbersInString(splits[8])[:-1])

        sp_add_sub_flops = ''.join(getNumbersInString(splits[9])[:-1])
        sp_div_flops = ''.join(getNumbersInString(splits[10])[:-1])
        sp_mult_add_flops = ''.join(getNumbersInString(splits[11])[:-1])
        sp_mult_flops = ''.join(getNumbersInString(splits[12])[:-1])


        if numIterations == 32:
            # Write it all to the CSV file
            csvFile.write(
                f'{N},{totalSeconds},{numIterations},' +
                f'{dp_add_sub_flops},{dp_div_flops},{dp_mult_add_flops},{dp_mult_flops},' +
                f'{sp_add_sub_flops},{sp_div_flops},{sp_mult_add_flops},{sp_mult_flops}\n'
            )
            csvFile.flush()

            # Break the loop; let's do it for other values of N now
            break
