#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:28:42 2021

"""


import numpy as np
import random
from collections import Counter

from number_game import expr_tree_2_polish_str, polish_str_2_expr_tree

# Q = [2, 3, 4, 4, 25, 50]
# Lnum = [2, 3, 4]



# print(list(set(Q) - set(Lnum)))

# print()

# print(list((Counter(Q) - Counter(Lnum)).elements()))


P1 = ['+', 3, ['-', ['+', 6, ['+', 9, 5]], ['+', 8, 10]]]

s = expr_tree_2_polish_str(P1)

print(f"expr_tree_2_polish_str: {s}   {type(s)}")

t = polish_str_2_expr_tree(s)
print(f"polish_str_2_expr_tree: {t}   {type(t)}")
# print(q_val)
# print(q_counts)

# print()


# print(l_val)
# print(l_counts)


# Q = nb.pick_numbers()
# T, U = nb.bottom_up_creator(Q)

# num_mutated = nb.mutate_num(T, Q)
# op_mutated = nb.mutate_op(T)

# print('----Numbers-----')
# print(Q)
# print('-------Tree--------')
# print(T)
# print('-----MutateNum------')
# print(num_mutated)
# print('--------MutateOp--------')
# print(op_mutated)
