# Application of Deep and Reinforcement Learning to Boundary Control Problems

Zenin Easa Panthakkalakath, Juraj Kardoš and Olaf Schenk

## Abstract:

The goal of boundary control problem is, in essence, to find the optimal values for the boundaries such that the values for the enclosed domain are as close as possible to desired values while adhering to the predetermined limits for the values of the domain and boundaries. Many scientific problems, such as fluid dynamics problems involving drag reduction, temperature control with some desired flow pattern, etc., rely on optimal boundary control algorithms to maintain the values at a desired range. These forward solves are performed for multiple simulation timesteps; thus is a time-critical component of the application toolchain. Having a method that can solve the boundary control problem with fewer computational resources would improve the performance of the overall simulations.

Traditionally, the solution is obtained using nonlinear optimization methods, such as interior point, wherein the computational bottleneck is introduced by the large linear systems. Interior point methods use information from the Hessian matrix of the logarithmic barrier function and use Newton's method to iteratively arrive at a solution; wherein it requires calculating the inverse of the Hessian matrix, constraint matrix and Jacobian of constraint matrix to be found, which is often done using the conjugate gradient method. The computational complexity of this is quite high and there seems to be room for improvement here.

The objectives of this project are to explore the possibilities of using deep learning and reinforcement learning methods to solve boundary control problems, and, design experiments wherein such methods are implemented and evaluated in an attempt to see if these can rival existing solvers in terms of speed and accuracy. One such category of approaches arrived at is along the lines of policy gradient reinforcement learning method. This method utilizes the idea behind iterative optimization by treating the iterative optimization algorithm as a policy, and learning or improving this policy using policy gradient method. A method has been arrived at using this strategy that demonstrated slightly better accuracy than traditional interior point method based solvers, while the performance is calculated to be slightly worse. Another such category of approaches is along the lines of agent-based modeling with reinforcement learning, wherein the values at the boundaries and/or domain are controlled by agents. However, no method along this line has been able to produce any results that can remotely compare to the the traditional interior point method based solvers.

Overall, using deep learning and reinforcement learning to arrive at methods to solve boundary control problems have a lot of promise.

## About repository

Before you decide to run any code, ensure that the instructions in [Code/setup.ipynb](Code/setup.ipynb) is followed. This helps in setting up the Anaconda environment and IPOPT for running the problems generated.

The code related to applying deep learning and reinforcement learning methods to solve boundary control problems is made available in the folder [Code/FinalAttempt](Code/FinalAttempt).

The code related to different baselines generated and a few initial experiments can be seen in the folder [Code/experiment](Code/experiment).

The information on synthetic data generation is made available in the file named [Code/dataGenerator.py](Code/dataGenerator.py).

## Citation

A lot of time and effort has been put into working on this project, writing the report and ensuring that the code is readable enough. Therefore, it would be lovely if you would cite us using the following.

```
@misc{panthakkalakath2023application,
      title={Application of deep and reinforcement learning to boundary control problems},
      author={Zenin Easa Panthakkalakath and Juraj Kardoš and Olaf Schenk},
      year={2023},
      eprint={2310.15191},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```