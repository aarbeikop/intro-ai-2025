#! /usr/bin/env python3

import argparse
import random
import sys

import instance
from utils import print_values, print_policy, wait_for_input


"""
Initialize state-values to 0 for all states for simplicity. (State-
values for non-goal states could be initialized arbitrarily.)
"""
def get_initial_values(inst):
    values = { s: 0.0 for s in inst.states }
    return values


"""
Compute the Q-value for state s and action under the given state-values.

returns:
    float: Q-value
"""
def compute_q_value(inst, s, action, values):
    # TODO: add your code here.
    # The goal state has Q-value of 0.
    # Return a float.
    raise NotImplementedError


"""
Compute the greedy action in state s under the given state-values (None
if s is a goal state) and also return the resulting Q-value of that best
action in s.

returns:
    tuple (str, float): greedy action, max Q-value
"""
def compute_greedy_action_and_q_value(inst, s, values):
    if s == inst.goal:
        return None, 0.0
    # TODO: add your code here.
    # Make use of compute_q_value to compute Q-values.
    # Return a pair of best action and its Q-value.
    raise NotImplementedError


"""
Update (in-place) the state-value of a random state according to the
Bellman equation for the given state-values (with discounted reward).

returns:
    None
"""
def bellman_update_in_place(values):
    # TODO: add your code here.
    # Make use of Python's random.choice to choose a random state.
    # Make use of compute_greedy_action_and_q_value to update
    # state-values with discount factor 0.9.
    raise NotImplementedError


"""
Compute a mapping from states to actions that represents the greedy
policy.
"""
def get_greedy_policy(inst, values):
    greedy_policy = {}
    for s in inst.states:
        best_a, _ = compute_greedy_action_and_q_value(inst, s, values)
        if best_a is None:
            assert s == inst.goal
            greedy_policy[s] = ' '
        else:
            assert inst.action_is_applicable(s, best_a)
            greedy_policy[s] = best_a
    return greedy_policy


"""
Run asynchronous value iteration for num_iterations many iterations.
In each iteration, perform a Bellman update for a single random state.
Return the final state-values and the greedy policy.

returns:
    tuple (dict, dict): values, greedy policy
"""
def asynchronous_value_iteration(inst, num_iterations):
    # TODO: add your code here.
    # Implement the algorithm. Initialize state-values using
    # get_initial_values(inst). In the loop of the algorithm, make use
    # of bellman_update_in_place(...).
    # For debugging, you can print state-values using
    # print_values(inst, values)
    # In each iteration, print the number of the current iteration
    # and the current state-values (again using print_values(...)).
    # Return the final state-values and a greedy policy computed
    # using the provided get_greedy_policy(inst, values).
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-iterations', type=int,
        help="number of iterations that asynchronous value iteration should run", default=300)
    args = parser.parse_args()

    inst = instance.get_example_instance()
    print(inst)

    values, policy = asynchronous_value_iteration(inst, args.num_iterations)
    print("")

    print("Final state-values:")
    print_values(inst, values)

    print("Final policy:")
    print_policy(inst, policy)
