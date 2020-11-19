# Minimal example to test MPI parallelization of code on a very basic level.
# For each particle in the swarm, start a separate HyppoPy optimization run on one processor.
# The individual HyppoPy optimization runs mutually communicate their intermediate results
# (fitnesses) to congruently propagate/update particles for the next generation.
# To run this script with MPI parallelization, OpenMPI needs to be installed:
# mpirun -np <number of processors> python dynPSO_mpi.py
# Workers not implemented yet!

import hyppopy.HyppopyProject
import hyppopy.solvers.DynamicPSOSolver
import numpy
from mpi4py import MPI

def updateParam(pop_history, num_params_obj): 
	return numpy.ones(num_params_obj)

def combineObj(args, params): 
	return sum([a*p for a, p in zip(args, params)])
	
def f(x,y):
	return [(x-5)**2, y**2]
	
# MPI-related stuff
size = MPI.COMM_WORLD.Get_size()	# number of overall processors
rank = MPI.COMM_WORLD.Get_rank()

# Parameters to set by user depending on optimization problem and (HPC) system:
block_size           = 1            # block size (block = 1 dyn. PSO sim. node + workers)
num_particles        = 1            # number of (local) particles per block
num_generations      = 500          # number of generations, i.e. iterations per (local) particle
num_particles_global = size // block_size
# overall number of iterations = block_size * num_generations

# intra-block communicator (divide different blocks from one another)
block_comm_color = rank // block_size
block_comm_key   = rank % block_size
comm_block       = MPI.COMM_WORLD.Split(color=block_comm_color, key=block_comm_key)

# inter-block communicator (divide simulators from bitches)
pso_comm_color = int(rank % block_size == 0)
pso_comm_key  = rank // block_size
comm_pso       = MPI.COMM_WORLD.Split(color=pso_comm_color, key=pso_comm_key) 
	
if rank % block_size == 0:
	project = hyppopy.HyppopyProject.HyppopyProject()
	project.add_hyperparameter(name="x", domain="uniform", data=[-10, 10], type=float)
	project.add_hyperparameter(name="y", domain="uniform", data=[-10, 10], type=float)
	project.add_setting(name="num_particles", value=num_particles)
	project.add_setting(name="num_generations", value=num_generations)
	project.add_setting(name="num_particles_global", value=num_particles_global)
	project.add_setting(name="num_params_obj", value=2) 
	project.add_setting(name="num_args_obj", value=2) 
	project.add_setting(name="combine_obj", value=combineObj)  
	project.add_setting(name="update_param", value=updateParam) 
	project.add_setting(name="comm_inter", value=comm_pso)
	project.add_setting(name="comm_intra", value=comm_block)
	
	solver=hyppopy.solvers.DynamicPSOSolver.DynamicPSOSolver(project)
	solver.blackbox=f
	solver.run()

else: pass
