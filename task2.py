#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from number_game import pick_numbers, eval_tree, display_tree
from genetic_algorithm import evolve_pop
import signal
from contextlib import contextmanager
import csv
'''
Custom exception to be thrown when function exceed time limit
'''
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    """
    Method time limiter function, will use signal package included in python.
    Throw an exception when the time limit is exceeded.

    Reference: Please note this block of code was taken from a public forum and
    provided an excellent solution to adding a time limit to a function.  CC BY-SA 3.0
    https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call

    Parameters
    ----------
    seconds: number of seconds to run function for
    """
    def signal_handler(_, __):
        """
        Handle the 

        Parameters
        ----------
        Params are unused and therefore       
        """
        raise TimeoutException("Timed out!")
    # Setup the signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    # Provide number of seconds to limit the signal to
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
    

def do_gen_alg(Q, target, pop_size=500, max_iters=200):
    """
    Complete the genetic algorithm from Q values and a target value.
    Try to solve the numbers game.

    Parameters
    ----------
    Q : list of integers
        Integers that were drawn by the game host
    target: integer
           target value of the game
    pop_size: population size of the gentical alg
    max_iters: number of max iterations to limit the genetic algorithm to 

    Returns
    -------
    v, T: the best expression tree found and its value
    """
    v, T = evolve_pop(Q, target, 
                    max_num_iteration = max_iters,
                    population_size = pop_size,
                    parents_portion = 0.5)
    return v, T

def print_intro(Q, target):
    """
    Print Q and target to the console
    
    Parameters
    ----------
    target : target value for number game to achieve
    Q : list of numbers initially available in the game
    """
    print('List of drawn numbers is ',Q)
    print(f'Target: {target}')

def print_results(v, T, target):
    """
    Print results of the genetic algorithm on the number game
    
    Parameters
    ----------
    v: resulting cost of completing the genetic algorithm 
    T : expression tree
    target : target value for number game to achieve
    """
    print('----------------------------')
    if v==0:
        print("\n***** Perfect Score!! *****")
    print(f'\ntarget {target} , tree value {eval_tree(T)}\n')
    display_tree(T)

def setup(print_log=True):
    """
    Setup the game by picking number and getting a target value for the game.
    These picked numbers are then sorted, intro is printed to the console.
    
    Returns
    -------
    Q : list of integers
            Integers that were drawn by the game host
    
    target: integer
           target value of the game
    """
    Q = pick_numbers()
    target = np.random.randint(1,1000)
    Q.sort()
    if print_log == True:
        print_intro(Q, target)
    return Q, target

def find_max_iters(min_pop_bound=100, max_pop_bound=2100, pop_step_size=100, 
                min_iters_bound=20, max_iters_bound=420, iters_step_size=20, number_games=5):
    """
    Numbers and Game gentic algorithm hyper parameter grid search between bounds.  Hyper-
    params being tuned is population size and max number of iterations.

    Find the maximum number of iterations that can be completed in the time constraint of
    2 seconds at different population sizes.  
    
    Notes: 
    ----------
    Possible success_or_fail values:
        0: Genetic Algorithm did not get to target value of game
        1: Genetic Algorithm did get to target value of game
        2: Gentic Algorithm exceeded 2 seconds constraint

    Parameters
    ----------
    min_pop_bound: minimum population size bound
    max_pop_bound: maximum population size bound
    pop_step_size: step size to increment the population
    min_iters_bound: minimum max iterations size bound
    max_iters_bound: minimum max iterations size bound
    iters_step_size: step size to increment the max iterations
    number_games: number of games to play per population per max iterations

    Returns
    ----------
    test_results: Results of all the games played while looping through bounds of pop and max iters

    """
    # Variable to hold test results in
    test_results = []
    # Perform gridsearch on hyper params population size and max iterations
    for p in range(min_pop_bound, max_pop_bound, pop_step_size):
        for i in range(min_iters_bound, max_iters_bound, iters_step_size):
            success_or_fail = [] # Array to hold values for test
            for _ in range(number_games):
                try:
                    # Setting up the game
                    Q, target = setup(print_log=False)
                    # Executing the gentic alg with a time limit of 2 seconds as per task 2 constraints
                    with time_limit(2):
                        # Perform genetic alg
                        v, T = do_gen_alg(Q, target, pop_size=p,  max_iters=i)
                        # Find out whether test was successfull or failure in finishing the numbers game
                        if v == 0:
                            success_or_fail.append(1)
                        else:
                            success_or_fail.append(0)
                # Exception if the time limit is exceeded
                except TimeoutException as e:
                    #If the method timed out add a 2 to the result array
                    success_or_fail.append(2)
                    print("Timed out!")
            # Compile test results in dict to make output easily convertable to csv
            test = {}
            test['pop'] = p
            test['iters'] = i
            test['res'] = success_or_fail
            #Append test results
            test_results.append(test)
    #Return results
    return test_results



def complete_test_output_csv(file_name, test_results):
    """
    Test results calculated by the find_max_iters function are outputted to a
    csv type file

    Parameters
    ----------
    file_name: File name the csv will be saved as
    test_results: Results from the find_max_iters function to be exported to csv
    """
    keys = test_results[0].keys()
    with open(file_name, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(test_results)


test_results = find_max_iters(min_pop_bound=100, max_pop_bound=2100, 
                            pop_step_size=100, min_iters_bound=20, 
                            max_iters_bound=240, iters_step_size=20)
# complete_test_output_csv('tests.csv')  

# test_results = find_max_iters(min_pop_bound=100, max_pop_bound=200, 
#                             pop_step_size=100, min_iters_bound=20, 
#                             max_iters_bound=240, iters_step_size=20, number_games=30)
# complete_test_output_csv('pop_100.csv', test_results)  

# test_results = find_max_iters(min_pop_bound=200, max_pop_bound=300, 
#                             pop_step_size=100, min_iters_bound=20, 
#                             max_iters_bound=120, iters_step_size=20, number_games=30)
# complete_test_output_csv('pop_200.csv', test_results)  

# test_results = find_max_iters(min_pop_bound=300, max_pop_bound=400, 
#                             pop_step_size=100, min_iters_bound=20, 
#                             max_iters_bound=80, iters_step_size=20, number_games=30)
# complete_test_output_csv('pop_300.csv', test_results)  

# test_results = find_max_iters(min_pop_bound=400, max_pop_bound=500, 
#                             pop_step_size=100, min_iters_bound=20, 
#                             max_iters_bound=60, iters_step_size=20, number_games=30)
# complete_test_output_csv('pop_400.csv', test_results)  

# test_results = find_max_iters(min_pop_bound=500, max_pop_bound=600, 
#                             pop_step_size=100, min_iters_bound=20, 
#                             max_iters_bound=60, iters_step_size=20, number_games=30)
# complete_test_output_csv('pop_500.csv', test_results)  


