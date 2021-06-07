'''

In the Letters and Numbers (L&N) game,
One contestant chooses how many "small" and "large" numbers they would like 
to make up six randomly chosen numbers. Small numbers are between 
1 and 10 inclusive, and large numbers are 25, 50, 75, or 100. 
All large numbers will be different, 
so at most four large numbers may be chosen. 


How to represent a computation?

Let Q = [q0, q1, q2, q3, q4, q5] be the list of drawn numbers

The building blocks of the expression trees are
 the arithmetic operators  +,-,*
 the numbers  q0, q1, q2, q3, q4, q5

We can encode arithmetic expressions with Polish notation
    op arg1 arg2
where op is one of the operators  +,-,*

or with expression trees:
    (op, left_tree, right_tree)
    
Recursive definition of an Expression Tree:
 an expression tree is either a 
 - a scalar   or
 - a binary tree (op, left_tree, right_tree)
   where op is in  {+,-,*}  and  
   the two subtrees left_tree, right_tree are expressions trees.

When an expression tree is reduced to a scalar, we call it trivial.


Author: f.maire@qut.edu.au

Created on April 1 , 2021
    

This module contains functions to manipulate expression trees occuring in the
L&N game.

'''



import numpy as np
import random
import ast

import copy # for deepcopy

import collections
from collections import Counter
import signal
from contextlib import contextmanager
import csv

SMALL_NUMBERS = tuple(range(1,11))
LARGE_NUMBERS = (25, 50, 75, 100)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(8799997, 'James', 'Tonkin'), (10512977, 'Lydia', 'Chen'), (10098101, 'Linh', 'Vu')]
# ----------------------------------------------------------------------------

def pick_numbers():
    '''    
    Create a random list of numbers according to the L&N game rules.
    
    Returns
    -------
    Q : int list
        list of numbers drawn randomly for one round of the game
    '''
    LN = set(LARGE_NUMBERS)
    Q = []
    for i in range(6):
        x = random.choice(list(SMALL_NUMBERS)+list(LN))
        Q.append(x)
        if x in LN:
            LN.remove(x)
    return Q


# ----------------------------------------------------------------------------

def bottom_up_creator(Q):
    '''
    Create a random algebraic expression tree
    that respects the L&N rules.
    
    Warning: Q is shuffled during the process

    Parameters
    ----------
    Q : non empty list of available numbers
        

    Returns  T, U
    -------
    T : expression tree 
    U : values used in the tree

    '''
    n = random.randint(1,6) # number of values we are going to use
    
    random.shuffle(Q)
    # Q[:n]  # list of the numbers we should use
    U = Q[:n].copy()
    
    if n==1:
        # return [U[0], None, None], [U[0]] # T, U
        return U[0], [U[0]] # T, U
        
    F = [u for u in U]  # F is initially a forest of values
    # we start with at least two trees in the forest
    while len(F)>1:
        # pick two trees and connect then with an arithmetic operator
        random.shuffle(F)
        op = random.choice(['-','+','*'])
        T = [op,F[-2],F[-1]]  # combine the last two trees
        F[-2:] = [] # remove the last two trees from the forest
        # insert the new tree in the forest
        F.append(T)
    # assert len(F)==1
    return F[0], U
  
# ---------------------------------------------------------------------------- 

def display_tree(T, indent=0):
    '''
    
    Eval the algebraic expression represented by T
    
    Parameters
    ----------
    T : Expression Tree
    indent: indentation for the recursive call

    Returns None

    '''
    # if T is a scalar, then we return it directly
    if isinstance(T, int):
        print('|'*indent,T, sep='')
        return
    # T is non trivial
    root_item = T[0]
    print('|'*indent, root_item, sep='')
    display_tree(T[1], indent+1)
    print('|'*indent)
    display_tree(T[2], indent+1)
   
# ---------------------------------------------------------------------------- 

def eval_tree(T):
    '''
    
    Eval the algebraic expression represented by T
    
    Parameters
    ----------
    T : Expression Tree

    Returns
    -------
    value of the algebraic expression represented by the T

    '''
    # if T is a scalar, then we return it directly
    if isinstance(T, int):
        return T
    # T is non trivial
    root_item = T[0]
    # assert root_item in ('-','+','*')
    left_value = eval_tree(T[1])
    right_value = eval_tree(T[2])
    return eval( str(left_value) +root_item + str(right_value) )
    # return eval(root_item.join([str(left_value), str(right_value)]))
   
     
# ---------------------------------------------------------------------------- 

def expr_tree_2_polish_str(T):
    '''
    Convert the Expression Tree into Polish notation

    Parameters
    ----------
    T : expression tree

    Returns
    -------
    string in Polish notation represention the expression tree T

    '''
    if isinstance(T, int):
        return str(T)
    root_item = T[0]
    # assert root_item in ('-','+','*')
    left_str = expr_tree_2_polish_str(T[1])
    right_str = expr_tree_2_polish_str(T[2])
    return '[' + ','.join([root_item,left_str,right_str]) + ']'
    

# ----------------------------------------------------------------------------

def polish_str_2_expr_tree(pn_str):
    '''
    Convert a polish notation string of an expression tree
    into an expression tree T.
    Parameters
    ----------
    pn_str : string representing an L&N algebraic expression
    Returns
    -------
    T
    '''
    pn_str = str(pn_str) # Ensure passed in is string

    # Method unused --> still implemented might be useful
    def find_match(i):
        '''
        Starting at position i where pn_str[i] == '['
        Return the index j of the matching ']'
        That is, pn_str[j] == ']' and the substring pn_str[i:j+1]
        is balanced
        '''
        # Use stack to track opening and closing of brackets
        stack = []
        for j,val in enumerate(pn_str):
            #If opening append to the stack
            if val == '[':
                stack.append('[')
            #If closing bracket and 1 opening bracket left thus must be the closing bracket
            if val == ']' and len(stack) == 1:
                return j, pn_str[i:j+1] # return index of the closing bracket
            if val == ']':
            # Pop as we are closing brackets
                stack.pop()
        return len(pn_str), pn_str[i:len(pn_str)+1]
            
     # .................................................................  

    # Add quotations marks to all elements so literal eval can be used
    # to convert the string to arrays
    new_pn_str = ""
    # Loop through each character and check for op or num
    for i in pn_str:
        if i == '[' or i == ']' or i == ',':
        # If bracket or comma, append same character
            new_pn_str+=i
        else:
        # If a op or num add quotation marks
            new_pn_str+='"'
            new_pn_str+=i
            new_pn_str+='"'


    def iter_convert(arr):
        '''
        Recursively traverse array and swap any string literal 
        integers to integers
        Parameters
        ----------
        arr : an n dimensional array to recursively loop through

        Returns
        -------
        Array such that all string literals of integers are swapped 
        out for integers
        '''

        for i, x in enumerate(arr):
            # If a list then recursively traverse the 
            if isinstance(x, list):
                iter_convert(x)
            else:
                # Else swap the the string literal to a string
                # using ignore exception method
                arr[i] = to_int_ignore_exception(x)
    # Convert the string representation of the expression tree (polish notation)
    # to literal
    arr = ast.literal_eval(new_pn_str)
    #Convert all the number string literals to literals
    iter_convert(arr)
    return arr

 
def to_int_ignore_exception(element):
    '''
    Attempt to convert element to an integer.  If a ValueError exception is raised
    return the element unchanged.  IF successful return the integer.
    Parameters
    ----------
    element : element to try to convert to int

    Returns
    -------
    Element as an int if successful, otherwise return the element unchanged
    '''
    try:
        return int(element)
    except ValueError:
        return element
# ----------------------------------------------------------------------------

def op_address_list(T, prefix = None):
    '''
    Return the address list L of the internal nodes of the expresssion tree T
    
    If T is a scalar, then L = []

    Note that the function 'decompose' is more general.

    Parameters
    ----------
    T : expression tree
    prefix: prefix to prepend to the addresses returned in L

    Returns
    -------
    L
    '''
    if isinstance(T, int):
        return []
    
    if prefix is None:
        prefix = []
        
    L = [prefix.copy() + [0]] # first adddress is the op of the root of T
    left_al = op_address_list(T[1], prefix.copy() + [1])
    L.extend(left_al)
    right_al = op_address_list(T[2], prefix.copy() + [2])
    L.extend(right_al)
    
    return L


# ----------------------------------------------------------------------------

def decompose(T, prefix = None):
    '''
    Compute
        Aop : address list of the operators
        Lop : list of the operators
        Anum : address of the numbers
        Lnum : list of the numbers
    
    For example, if 
    
    T =  ['-', ['+', ['-', 75, ['-', 10, 3]], ['-', 100, 50]], 3]
    
    then, 
    
     Aop is  [[0], [1, 0], [1, 1, 0], [1, 1, 2, 0], [1, 2, 0]] 
    
     Lop is ['-', '+', '-', '-', '-'] 
    
     Anum is [[1, 1, 1], [1, 1, 2, 1], [1, 1, 2, 2], [1, 2, 1], [1, 2, 2], [2]] 
    
     Lnum is [75, 10, 3, 100, 50, 3]    
        
    
    Parameters
    ----------
    T : expression tree 
    
    prefix : address to preprend 

    Returns
    -------
    Aop, Lop, Anum, Lnum

    '''
    #Ensure prefix is a workable list on init
    if prefix is None:
        prefix = []

    # If T == int we have reached a specific number in notation
    if isinstance(T, int):
        # Return data, no need to continue to break down notation
        Aop = []
        Lop = [] 
        Anum = [prefix]
        Lnum = [T]
        return Aop, Lop, Anum, Lnum
    # Make sure T is a list
    assert isinstance(T, list)
    
    Aop = [prefix.copy() + [0]] # first adddress is the op of the root of T
    Lop = [T[0]] # get the first operation
    # Lists to store the Address number and list of number
    Anum = []
    Lnum = []

    # Recursive call to get data for left side of polish notation
    left_op_add, left_op, left_num_add, left_num = decompose(T[1], prefix.copy() + [1])
    #Extend left operators
    Aop.extend(left_op_add)
    Lop.extend(left_op)
    Anum.extend(left_num_add)
    Lnum.extend(left_num)

    # Recursive call to get data for right side of polish notation
    right_op_add, right_op, right_num_add, right_num = decompose(T[2], prefix.copy() + [2])
    #Extend right operators
    Aop.extend(right_op_add)
    Lop.extend(right_op)
    Anum.extend(right_num_add)
    Lnum.extend(right_num)
    
    # Return data
    return Aop, Lop, Anum, Lnum


# ----------------------------------------------------------------------------

def get_item(T, a):
    '''
    Get the item at address a in the expression tree T

    Parameters
    ----------
    T : expression tree
    a : valid address of an item in the tree

    Returns
    -------
    the item at address a

    '''
    if len(a)==0:
        return T
    # else
    return get_item(T[a[0]], a[1:])
        
# ----------------------------------------------------------------------------

def replace_subtree(T, a, S):
    '''
    Replace the subtree at address a
    with the subtree S in the expression tree T
    
    The address a is a sequence of integers in {0,1,2}.
    
    If a == [] , then we return S
    If a == [1], we replace the left subtree of T with S
    If a == [2], we replace the right subtree of T with S

    Returns
    ------- 
    The modified tree

    Warning: the original tree T is modified. 
             Use copy.deepcopy()  if you want to preserve the original tree.
    '''    
    
    # base case, address empty
    if len(a)==0:
        return S
    
    # recursive case
    T[a[0]] = replace_subtree(T[a[0]], a[1:], S)
    return T


# ----------------------------------------------------------------------------

def mutate_num(T, Q):
    '''
    Mutate one of the numbers of the expression tree T
    
    Parameters
    ----------
    T : expression tree
    Q : list of numbers initially available in the game

    Returns
    -------
    A mutated copy of T

    '''
    # Decompose the tree
    Aop, Lop, Anum, Lnum = decompose(T)    

    # Make a copy of T
    T_copy = copy.deepcopy(T)

    # Check if T is a scalar, return T
    if isinstance(T, int):
        return T
    
    # Check if all the numbers in Q are used for T
    # If correct, T is not mutated, return the main T
    if len(Q) == len(Lnum):
        mutant_T = T_copy
    
    # If not correct, T is mutated
    else:
        # Choose a number randomly in the tree to mutate
        num_address = random.choice(Anum)
        num_new = get_item(T_copy, num_address)

        # Choosing the set number for mutation
        option_num = list((Counter(Q) - Counter(Lnum)).elements())

        # Check for diffence in the option since it has repeated value
        if len(option_num) != 0:
            option_random = []
            
            # Check if the number is the same with the option
            for i in range(len(option_num)):
                if option_num[i] != num_new:
                    option_random.append(option_num[i])
            
            #Check to make sure there is a value to select
            if len(option_random) != 0:
                temp_num = random.choice(option_random)
                # Replace the new number into the tree
                mutant_T = replace_subtree(T_copy, num_address, temp_num)
            else:
                # This result in no difference in the option
                # The tree is not mutated
                mutant_T = T_copy
        else:
            # This result in no difference in the option
            # The tree is not mutated
            mutant_T = T_copy

    return mutant_T    
    

# ----------------------------------------------------------------------------

def mutate_op(T):
    '''
    Mutate an operator of the expression tree T
    If T is a scalar, return T

    Parameters
    ----------
    T : non trivial expression tree

    Returns
    -------
    A mutated copy of T

    '''
    if isinstance(T, int):
        return T
    
    # Make a deep copy of T
    T_copy = copy.deepcopy(T)

    # Get the address of operator in T
    La = op_address_list(T_copy)

    # random address of an op in T
    a = random.choice(La)  

    # the char of the op
    op_c = get_item(T_copy, a)       
    
    # Option for mutation
    option1 = ['-', '+']
    option2 = ['+', '*']
    option3 = ['-', '*']

    if op_c == '-':
        temp_op = random.choice(option2)
    elif op_c == '+':
        temp_op = random.choice(option3)
    elif op_c == '*':
        temp_op = random.choice(option1)

    # Replace the new operator in the tree
    mutant_c = replace_subtree(T_copy,a,temp_op)
    return mutant_c
    

# ----------------------------------------------------------------------------

def cross_over(P1, P2, Q):    
    '''
    Perform crossover on two non trivial parents
    
    Parameters
    ----------
    P1 : parent 1, non trivial expression tree  (root is an op)
    P2 : parent 2, non trivial expression tree  (root is an op)
        
    Q : list of the available numbers
        Q may contain repeated small numbers    
        

    Returns
    -------
    C1, C2 : two children obtained by crossover
    '''
    
    def get_num_ind(aop, Anum):
        '''
        Return the indices [a,b) of the range of numbers
        in Anum and Lum that are in the sub-tree 
        rooted at address aop

        Parameters
        ----------
        aop : address of an operator (considered as the root of a subtree).
              The address aop is an element of Aop
        Anum : the list of addresses of the numbers

        Returns
        -------
        a, b : endpoints of the semi-open interval
        
        '''
        d = len(aop)-1  # depth of the operator. 
                        # Root of the expression tree is a depth 0
        # K: list of the indices of the numbers in the subtrees
        # These numbers must have the same address prefix as aop
        p = aop[:d] # prefix common to the elements of the subtrees
        K = [k for k in range(len(Anum)) if Anum[k][:d]==p ]
        return K[0], K[-1]+1
        # .........................................................
        
    Aop_1, Lop_1, Anum_1, Lnum_1 = decompose(P1)
    Aop_2, Lop_2, Anum_2, Lnum_2 = decompose(P2)

    C1 = copy.deepcopy(P1)
    C2 = copy.deepcopy(P2)
    
    i1 = np.random.randint(0,len(Lop_1)) # pick a subtree in C1 by selecting the index
                                         # of an op
    i2 = np.random.randint(0,len(Lop_2)) # Select a subtree in C2 in a similar way
 
    # i1, i2 = 4, 0 # DEBUG    
 
    # Try to swap in C1 and C2 the sub-trees S1 and S2 
    # at addresses Lop_1[i1] and Lop_2[i2].
    # That's our crossover operation!
    
    # Compute some auxiliary number lists
    
    # Endpoints of the intervals of the subtrees
    a1, b1 = get_num_ind(Aop_1[i1], Anum_1)     # indices of the numbers in S1 
                                                # wrt C1 number list Lnum_1
    a2, b2 = get_num_ind(Aop_2[i2], Anum_2)   # same for S2 wrt C2
    
    # Lnum_1[a1:b1] is the list of numbers in S1
    # Lnum_2[a2:b2] is the list of numbers in S2
    
    # numbers is C1 not used in S1
    nums_C1mS1 = Lnum_1[:a1]+Lnum_1[b1:]
    # numbers is C2-S2
    nums_C2mS2 = Lnum_2[:a2]+Lnum_2[b2:]
    
    # S2 is a fine replacement of S1 in C1
    # if nums_S2 + nums_C1mS1 is contained in Q
    # if not we can bottom up a subtree with  Q-nums_C1mS1

    counter_Q = collections.Counter(Q) # some small numbers can be repeated
    
    d1 = len(Aop_1[i1])-1
    aS1 = Aop_1[i1][:d1] # address of the subtree S1 
    S1 = get_item(C1, aS1)

    d2 = len(Aop_2[i2])-1
    aS2 = Aop_2[i2][:d2] # address of the subtree S1 
    S2 = get_item(C2, aS2)

    # print('\n DEBUG -------- S1 and S2 ----------') # DEBUG
    # print(S1)
    # print(S2)


    # count the numbers (their occurences) in the candidate child C1
    counter_1 = collections.Counter(Lnum_2[a2:b2]+nums_C1mS1)
    
    # Test whether child C1 is ok
    if all(counter_Q[v]>=counter_1[v]  for v in counter_Q):
        # candidate is fine!  :-)
        C1 = replace_subtree(C1, aS1, S2)
    else:
        available_nums = counter_Q.copy()
        available_nums.subtract(
            collections.Counter(nums_C1mS1)
            )
        R1, _ = bottom_up_creator(list(available_nums.elements()))
        C1 = replace_subtree(C1, aS1, R1)
        
    # count the numbers (their occurences) in the candidate child C2
    counter_2 = collections.Counter(Lnum_1[a1:b1]+nums_C2mS2)
    
    # Test whether child C2 is ok
    if all(counter_Q[v]>=counter_2[v]  for v in counter_Q):
        # candidate is fine!  :-)
        C2 = replace_subtree(C2, aS2, S1)
    else:
        available_nums = counter_Q.copy()
        available_nums.subtract(
            collections.Counter(nums_C2mS2)
            )
        R2, _ = bottom_up_creator(list(available_nums.elements()))
        C2 = replace_subtree(C2, aS2, R2)
    
    
    return C1, C2

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


def setup():
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
                    Q, target = setup()
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


def complete_genetic_algorithm_tests():
    """
    Tests that were completed to find the max number of iterations for different numbers of 
    population sizes.  Note, to find these bound 5 separete runs of the letters and numbers game
    was run and solved with gentic algorithm.  After bounds where found for each pop size that could
    successfully be run without exceeding 2 second time limit and corrersponding max iters, 30 iterations
    of the letters and numbers game was completed using the gentic alg.

    After results were populated, these results were then exported into csv files and then analysed
    using excel, jupyter notebooks, python, and the pandas package.  All the was completed was building
    graphs for the data and is not included in this solution.  The report covers the graph and the analysis.
    """

    # RUN 5 GAMES PER POP AND ITER SIZE
    # Initial tests to find the upper bound of max iters by find at which point
    # time outs start to become an issue for 2 second time constraint
    test_results = find_max_iters(min_pop_bound=100, max_pop_bound=200, 
                            pop_step_size=100, min_iters_bound=20, 
                            max_iters_bound=500, iters_step_size=20)
    complete_test_output_csv('tests.csv')  

    # Found the upper bound for max iters is 220 --> run coarse grid search within bounds
    test_results = find_max_iters(min_pop_bound=100, max_pop_bound=2100, 
                            pop_step_size=100, min_iters_bound=20, 
                            max_iters_bound=240, iters_step_size=20)
    complete_test_output_csv('tests.csv')  

    # Max iteration bounds found to not exceed time limit of 2 seconds
    # Population Size 100:
    # Max iters: 240
    # Population Size 200:
    # Max iters: 100
    # Population Size 300:
    # Max iters: 60
    # Population Size 400:
    # Max iters: 40
    # Population Size 500:
    # Max iters: 40

    # RUN 30 GAMES PER POP AND ITER SIZE   

    # Test to find to find success and fail rate for population 100
    test_results = find_max_iters(min_pop_bound=100, max_pop_bound=200, 
                                pop_step_size=100, min_iters_bound=20, 
                                max_iters_bound=240, iters_step_size=20, number_games=30)
    complete_test_output_csv('pop_100.csv', test_results)  

    # Test to find to find success and fail rate for population 200
    test_results = find_max_iters(min_pop_bound=200, max_pop_bound=300, 
                                pop_step_size=100, min_iters_bound=20, 
                                max_iters_bound=120, iters_step_size=20, number_games=30)
    complete_test_output_csv('pop_200.csv', test_results)  

    # Test to find to find success and fail rate for population 300
    test_results = find_max_iters(min_pop_bound=300, max_pop_bound=400, 
                                pop_step_size=100, min_iters_bound=20, 
                                max_iters_bound=80, iters_step_size=20, number_games=30)
    complete_test_output_csv('pop_300.csv', test_results)  

    # Test to find to find success and fail rate for population 400
    test_results = find_max_iters(min_pop_bound=400, max_pop_bound=500, 
                                pop_step_size=100, min_iters_bound=20, 
                                max_iters_bound=60, iters_step_size=20, number_games=30)
    complete_test_output_csv('pop_400.csv', test_results)  

    # Test to find to find success and fail rate for population 500
    test_results = find_max_iters(min_pop_bound=500, max_pop_bound=600, 
                                pop_step_size=100, min_iters_bound=20, 
                                max_iters_bound=60, iters_step_size=20, number_games=30)
    complete_test_output_csv('pop_500.csv', test_results)  


## To use comment out import in genetic_algorithm.py --> wont work if both files have imports: circular imports
# from genetic_algorithm import evolve_pop
# if __name__ == "__main__":
#     complete_genetic_algorithm_tests()