# robot.py
# Coded by George H. Chen (georgehc@csail.mit.edu)
from __future__ import division
import numpy as np
from utils import Distribution


#-----------------------------------------------------------------------------
# Some constants
#

GRID_WIDTH  = 20
GRID_HEIGHT = 6


#-----------------------------------------------------------------------------
# Please read!
#
# Convention: (x, y) = (0, 0) is the top left of the grid
#
# Each hidden state is encoded as (x, y, action)
# where: 0 <= x <= GRID_WIDTH - 1,
#        0 <= y <= GRID_HEIGHT - 1,
#        action is one of
#        {'left', 'right', 'up', 'down', 'stay'}.
# Note that <action> refers to the *previous* action taken so that we arrived
# at (x, y). In particular, it is NOT guaranteed to be the next action to be
# taken by the robot.
#
# Each observed state is encoded as (x, y)
# where: 0 <= x <= GRID_WIDTH - 1,
#        0 <= y <= GRID_HEIGHT - 1.
#


#-----------------------------------------------------------------------------
# Functions specifying the robot model (e.g., listing all possible hidden and
# observed states, initial distribution, transition model, observation model)
#

def get_all_hidden_states():
    # lists all possible hidden states
    all_states = []
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            possible_prev_actions = ['left', 'right', 'up', 'down', 'stay']

            if x == 0: # previous action could not have been to go right
                possible_prev_actions.remove('right')
            if x == GRID_WIDTH - 1: # could not have gone left
                possible_prev_actions.remove('left')
            if y == 0: # could not have gone down
                possible_prev_actions.remove('down')
            if y == GRID_HEIGHT - 1: # could not have gone up
                possible_prev_actions.remove('up')

            for action in possible_prev_actions:
                all_states.append( (x, y, action) )
    return all_states

def get_all_observed_states():
    # lists all possible observed states
    all_observed_states = []
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            all_observed_states.append( (x, y) )
    return all_observed_states

def initial_distribution():
    # returns a Distribution for the initial hidden state
    prior = Distribution()
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            prior[(x, y, 'stay')] = 1./(GRID_WIDTH*GRID_HEIGHT)
    return prior

def make_dict_based_model(model_dict):
    # Takes a 2-layer dictionary representing a conditional distribution, and
    # returns a function.
    #   That function takes the conditioning as argument and returns
    #   a Distribution.
    def dict_based_model(state):
        return model_dict[state]
    return dict_based_model

make_observation_model = make_dict_based_model
make_transition_model  = make_dict_based_model

def get_valid_next_states(state):
    x, y, action = state

    valid_next_states = []

    # we can always stay where we are
    valid_next_states.append((x, y, 'stay'))

    if y > 0 and action in ('stay', 'up'): # we can go up
        valid_next_states.append((x, y-1, 'up'))

    if y < GRID_HEIGHT - 1 and action in ('stay', 'down'): # we can go down
        valid_next_states.append((x, y+1, 'down'))

    if x > 0 and action in ('stay', 'left'): # we can go left
        valid_next_states.append((x-1, y, 'left'))

    if x < GRID_WIDTH - 1 and action in ('stay', 'right'): # we can go right
        valid_next_states.append((x+1, y, 'right'))

    return valid_next_states

def uniform_transition_model(state):
    next_state_distribution = Distribution()
    valid_next_states = get_valid_next_states(state)
    for next_state in valid_next_states:
        next_state_distribution[next_state] += 1
    next_state_distribution.renormalize()

    return next_state_distribution

def transition_model(state):
    # given a hidden state, return the Distribution for the next hidden state
    x, y, action = state
    next_states  = Distribution()

    # we can always stay where we are
    if action == 'stay':
        next_states[(x, y, 'stay')] = .2
    else:
        next_states[(x, y, 'stay')] = .1

    if y > 0: # we can go up
        if action == 'stay':
            next_states[(x, y-1, 'up')] = .2
        if action == 'up':
            next_states[(x, y-1, 'up')] = .9
    if y < GRID_HEIGHT - 1: # we can go down
        if action == 'stay':
            next_states[(x, y+1, 'down')] = .2
        if action == 'down':
            next_states[(x, y+1, 'down')] = .9
    if x > 0: # we can go left
        if action == 'stay':
            next_states[(x-1, y, 'left')] = .2
        if action == 'left':
            next_states[(x-1, y, 'left')] = .9
    if x < GRID_WIDTH - 1: # we can go right
        if action == 'stay':
            next_states[(x+1, y, 'right')] = .2
        if action == 'right':
            next_states[(x+1, y, 'right')] = .9

    next_states.renormalize()
    return next_states

def get_valid_observations(state):
    return spread_observation_model(state).keys()

def observation_model(state, radius=1):
    # given a hidden state, return the Distribution for its observation
    x, y, action    = state
    observed_states = Distribution()

    for x_new in range(x - radius, x + radius + 1):
        for y_new in range(y - radius, y + radius + 1):
            if x_new >= 0 and x_new <= GRID_WIDTH - 1 and \
               y_new >= 0 and y_new <= GRID_HEIGHT - 1:
                if (x_new - x)**2 + (y_new - y)**2 <= radius**2:
                    observed_states[(x_new, y_new)] = 1.

    observed_states.renormalize()
    return observed_states

def discretized_gaussian_observation_model(state, sigma=1):
    # given a hidden state, return the Distribution for its observation
    x, y, action    = state
    observed_states = Distribution()

    # x_new, y_new = np.meshgrid(range(GRID_WIDTH), range(GRID_HEIGHT))
    # values = np.exp( -( (x_new - x)**2 + (y_new -y)**2 )/(2.*sigma) )

    for x_new in range(GRID_WIDTH):
        for y_new in range(GRID_HEIGHT):
            observed_states[(x_new, y_new)] = \
                np.exp(-( (x_new - x)**2 + (y_new - y)**2 )/(2.*sigma))

    observed_states.renormalize()
    return observed_states

def spread_observation_model(state, radius=1):
    # given a hidden state, return the Distribution for its observation
    x, y, action    = state
    observed_states = Distribution()

    for x_new in range(x - radius, x + radius + 1):
        for y_new in range(y - radius, y + radius + 1):
            if x_new >= 0 and x_new <= GRID_WIDTH - 1 and \
               y_new >= 0 and y_new <= GRID_HEIGHT - 1:
                observed_states[(x_new, y_new)] = 1.

    observed_states.renormalize()
    return observed_states


#-----------------------------------------------------------------------------
# Comparison/evaluation
#

def compute_number_of_differences(state_seq1, state_seq2):
    # for two sequences of states of the same length, returns the number of
    # differences between the two sequences

    num_time_steps = len(state_seq1)
    assert len(state_seq2) == num_time_steps, \
           'The two lists of states to compare must have the same length.'

    difference = 0
    for n in range(num_time_steps):
        if state_seq1[n] != state_seq2[n]:
            difference += 1

    return difference

def compute_average_distance_between_states(state_seq1, state_seq2):
    distance = 0
    for (state1, state2) in zip(state_seq1, state_seq2):
        (x1, y1, direction) = state1
        (x2, y2, direction) = state2
        distance += np.sqrt((x1-x2)**2 + (y1-y2)**2)

    return distance / len(state_seq1)


#-----------------------------------------------------------------------------
# Saving and loading lists of hidden states and observations
#

def save_data(filename, hidden_states, observations):
    # saves a list of hidden states and observations to a text file where each
    # line says:
    # <hidden x> <hidden y> <hidden action> <observed x> <observed y>
    # OR
    # <hidden x> <hidden y> <hidden action> missing
    # with the latter happening for a missing observation
    f = open(filename, 'w')
    assert len(hidden_states) == len(observations)

    # start by writing grid size
    f.write("%d %d\n" % (GRID_WIDTH, GRID_HEIGHT))

    for time_step in range(len(hidden_states)):
        hidden_x, hidden_y, hidden_action = hidden_states[time_step]
        if observations[time_step] is not None:
            observed_x, observed_y = observations[time_step]
            f.write("%d %d %s %d %d\n" \
                    % (hidden_x, hidden_y, hidden_action,
                       observed_x, observed_y))
        else:
            f.write("%d %d %s missing\n" \
                    % (hidden_x, hidden_y, hidden_action))

    f.close()

def load_data(filename):
    # loads a list of hidden states and observations saved by save_data()
    f = open(filename, 'r')

    first_line = True

    hidden_states = []
    observations  = []
    for line in f.readlines():
        line = line.strip()
        if first_line:
            parts = line.split()

            grid_width  = int(parts[0])
            grid_height = int(parts[1])

            first_line = False
        else:
          if len(line) >= 4:
              parts = line.split()

              hidden_x      = int(parts[0])
              hidden_y      = int(parts[1])
              hidden_action = parts[2]
              hidden_states.append( (hidden_x, hidden_y, hidden_action) )

              if parts[3] == 'missing':
                  observations.append(None)
              elif len(parts) == 5:
                  observed_x = int(parts[3])
                  observed_y = int(parts[4])
                  observations.append( (observed_x, observed_y) )

    return grid_width, grid_height, hidden_states, observations

