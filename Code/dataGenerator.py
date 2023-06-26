# This script shall invoke the random equation generator defined in
# 'utilities.py', feed it as an input to IPOPT and store it all into a database
# or database-like thing. This data shall be used to train test and validate
# the models that we are coming up with.
# Even for unsupervised methods, this helps in comparing the solution created
# against IPOPT; helping in reporting how good our method/model is.
import os
import random
import torch
import utilities as utils

# Seeds for reproducibility
random.seed(0)
#np.random.seed(0)
torch.manual_seed(0)

# Number of equations to be generated
NUM_EQUATIONS = 10000
MINIMUM_REQUIRED_DIFFERENCE = 0.3
MAX_ALLOWED_COST = 0.2

# Creating folders and files
folderName = 'data'
csvFileName = folderName + '/data.csv'
os.system(f'rm -rf {folderName}')
os.system(f'mkdir {folderName}')

csvFile = open(csvFileName, 'a')
#csvFile.write('--- CSVFILE ---\n\n')
#csvFile.flush()
csvFile.write(
    'i,N,alpha,lb_y,ub_y,lb_u,ub_u,d_const,target_profile_equation,cost\n'
)
csvFile.flush()

for i in range(NUM_EQUATIONS):
    # To ensure that the equations generated has a good difference between
    # the maximum and minimum values, and, the cost predicted by IPOPT is low
    # enough, we have this loop.
    do = True
    while do:
        # Generate random problem
        (
            problemType, N, alpha,
            lb_y, ub_y, lb_u, ub_u,
            d_const, target_profile_equation, array
        ) = utils.generateRandomBoundaryProblem()

        if array.max() - array.min() < MINIMUM_REQUIRED_DIFFERENCE:
            continue

        # Solve the problem in IPOPT
        (y_desired_ipopt, y_solution_ipopt) = utils.generateBoundaryDataFromIPOPT(
            problemType, N, alpha,
            lb_y, ub_y, lb_u, ub_u,
            d_const, target_profile_equation
        )

        # Let's just make sure that our way of calculating the domain values is
        # consistent with the manner in which IPOPT does it
        #assert torch.allclose(y_desired_ipopt, array)

        # Calculate cost
        cost = utils.costFunction2(
            y_desired_ipopt, y_solution_ipopt, lb_y, ub_y, lb_u, ub_u
        ).mean()

        do = cost > MAX_ALLOWED_COST


    # Write the data into the CSV file
    csvFile.write(
        f'{i},{N},{alpha},{lb_y},{ub_y},{lb_u},{ub_u},{d_const},"{target_profile_equation}",{cost}\n'
    )
    csvFile.flush()

    # Save 'y_desired_ipopt' and 'y_solution_ipopt'
    #torch.save(y_desired_ipopt, f'{folderName}/{i}_desired_IPOPT.pt')
    #torch.save(y_solution_ipopt, f'{folderName}/{i}_solution_IPOPT.pt')

    print(f'Finished processing equation number {i}...\n')

# TODO: Ensure that the equations generated has a good difference between
# the maximum and minimum values, and, the cost predicted by IPOPT is low
# enough.
# Define a threshold for the same based on the examples in the original paper.
