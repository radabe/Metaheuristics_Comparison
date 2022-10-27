# Benchmarking Metaheuristics for Effectivity Date Optimization
This repository contains simple python codes for a Genetic Algorithm, Ant Colony Optimization, Particle Swarm Optimization, Tabu Search and Simmulated Annealing. and is Supplementary material for the publication 'Deciding on when to change - A Benchmark of Metaheuristic Algorithms for Timing Engineering Changes' by Peter Burggräf, Fabian Steinberg, Tim Weißer and Ognjen Radisic-Aberger. 

The repository contains scripts for the Algorithms themselves, and running the benchmark with a test set-up. 

## Reference
If you use code from this repository, please cite the original paper:


## Requirements
The code is written in Python 3.6. The necessary libraries vary from algorithm to algorithm, and are all included in the main Input.py file. 

## Usage

### Data format

The scripts assume that input is given in the form of an .xlsx file, as seen in the the Example input file.
The script loads all Engineering Batches by the name of the respective table in the .xlsx file, and extracts the relevant data into a python dictonary, which is than called upon by the algorithms. 
