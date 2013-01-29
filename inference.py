#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/14/2012 12:17am
# Modified by: George H. Chen (georgehc@mit.edu)
import sys
import numpy as np
import robot
import graphics
from utils import Distribution


#-----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states

    all_possible_observed_states: a list of possible observed states

    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state

    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state

    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)

    #-------------------------------------------------------------------------
    # Fold observations into singleton potentials
    #
    phis = [] # phis[n] is the singleton potential for node n
    for n in range(num_time_steps):
        potential      = Distribution()
        observed_state = observations[n]
        if n == 0:
            for hidden_state in prior_distribution:
                value = prior_distribution[hidden_state]
                if observed_state is not None:
                    value *= observation_model(hidden_state)[observed_state]
                if value > 0: # only store entries with nonzero prob.
                    potential[hidden_state] = value
        else:
            for hidden_state in all_possible_hidden_states:
                if observed_state is None:
                    # singleton potential should be identically 1
                    potential[hidden_state] = 1.
                else:
                    value = observation_model(hidden_state)[observed_state]
                    if value > 0: # only store entries with nonzero prob.
                        potential[hidden_state] = value
        assert len(potential.keys()) > 0 , \
                "Invalid observation at time %d. Maybe you \
                forgot the --use-spread-output argument?"%n
        phis.append(potential)


    # we need not recompute edge potentials since they're given by the
    # transition model: phi(x_i, x_j) = transition_model[x_i](x_j),
    # where j = i+1


    #-------------------------------------------------------------------------
    # Forward pass
    #
    forward_messages = []

    # compute message from non-existent node -1 to node 0
    message = Distribution()
    for hidden_state in all_possible_hidden_states:
        message[hidden_state] = 1.
    message.renormalize()
    forward_messages.append(message)

    for n in range(num_time_steps - 1):
        # compute message from node n to node n+1
        message = Distribution()

        ## the commented block below is easier to understand but is slow;
        ## a faster version is below that switches the order of the for loops
        ## and reduces the number of states that we iterate over

        #for next_hidden_state in all_possible_hidden_states:
        #    value = 0.
        #    # only loop over hidden states with nonzero singleton potential!
        #    for hidden_state in phis[n]:
        #        value += phis[n][hidden_state] * \
        #                 transition_model(hidden_state)[next_hidden_state] * \
        #                 forward_messages[-1][hidden_state]
        #    if value > 0: # only store entries with nonzero prob.
        #        message[next_hidden_state] = value

        ## faster version of the commented block above
        # 1. only loop over hidden states with nonzero singleton potential!
        for hidden_state in phis[n]:
            # 2. only loop over possible next hidden states given current
            #    hidden state
            for next_hidden_state in transition_model(hidden_state):
                factor = phis[n][hidden_state] * \
                         transition_model(hidden_state)[next_hidden_state] * \
                         forward_messages[-1][hidden_state]
                if factor > 0: # only store entries with nonzero prob.
                    if next_hidden_state in message:
                        message[next_hidden_state] += factor
                    else:
                        message[next_hidden_state] = factor

        message.renormalize()
        forward_messages.append(message)

    #-------------------------------------------------------------------------
    # Pre-processing to speed up the backward pass: cache for each hidden
    # state what the possible previous hidden states are
    #
    possible_prev_hidden_states = {}
    for hidden_state in all_possible_hidden_states:
        for next_hidden_state in transition_model(hidden_state):
            if next_hidden_state in possible_prev_hidden_states:
                possible_prev_hidden_states[next_hidden_state].add( \
                    hidden_state)
            else:
                possible_prev_hidden_states[next_hidden_state] = \
                    set([hidden_state])

    #-------------------------------------------------------------------------
    # Backward pass
    #
    backward_messages = []

    # compute message from non-existent node <num_time_steps> to node
    # <num_time_steps>-1
    message = Distribution()
    for hidden_state in all_possible_hidden_states:
        message[hidden_state] = 1.
    message.renormalize()
    backward_messages.append(message)

    for n in range(num_time_steps - 2, -1, -1):
        # compute message from node n+1 to n
        message = Distribution()

        ## again, I've commented out a block that's easier to understand but
        ## slow; the faster version is below

        #for hidden_state in all_possible_hidden_states:
        #    value = 0.
        #    for next_hidden_state in transition_model(hidden_state):
        #        value += phis[n+1][next_hidden_state] * \
        #                 transition_model(hidden_state)[next_hidden_state] * \
        #                 backward_messages[0][next_hidden_state]
        #    if value > 0: # only store entries with nonzero prob.
        #        message[hidden_state] = value

        ## faster version
        # 1. only loop over next hidden states with nonzero potential!
        for next_hidden_state in phis[n+1]:
            # 2. only loop over possible previous hidden states
            for hidden_state in possible_prev_hidden_states[next_hidden_state]:
                factor = phis[n+1][next_hidden_state] * \
                         transition_model(hidden_state)[next_hidden_state] * \
                         backward_messages[0][next_hidden_state]
                if factor > 0: # only store entries with nonzero prob.
                    if hidden_state in message:
                        message[hidden_state] += factor
                    else:
                        message[hidden_state] = factor

        message.renormalize()
        backward_messages.insert(0, message)

    #-------------------------------------------------------------------------
    # Compute marginals
    #
    marginals = []
    for n in range(num_time_steps):
        marginal = Distribution()
        for hidden_state in all_possible_hidden_states:
            if hidden_state in forward_messages[n] and \
               hidden_state in backward_messages[n] and \
               hidden_state in phis[n]:
                value = forward_messages[n][hidden_state] * \
                        backward_messages[n][hidden_state] * \
                        phis[n][hidden_state]
                if value > 0: # only store entries with nonzero prob.
                    marginal[hidden_state] = value
        marginal.renormalize()
        marginals.append(marginal)

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    ### YOUR CODE HERE: Estimate marginals & pairwise marginals
    pairwise_marginals = [None] * (num_time_steps - 1)

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return (marginals, pairwise_marginals)

def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0.:
        return float('-inf')
    else:
        return np.log(x)

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list of inputs for the function forward_backward() above.

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    num_time_steps = len(observations)

    # Below is an implementation of the Min-Sum algorithm presented in class
    # specialized to the HMM case

    messages      = [] # best values so far
    back_pointers = [] # back-pointers for best values so far

    #-------------------------------------------------------------------------
    # Fold observations into singleton potentials
    #
    phis = [] # phis[n] is the singleton potential for node n
    for n in range(num_time_steps):
        potential      = Distribution()
        observed_state = observations[n]
        if n == 0:
            for hidden_state in prior_distribution:
                value = prior_distribution[hidden_state]
                if observed_state is not None:
                    value *= observation_model(hidden_state)[observed_state]
                if value > 0: # only store entries with nonzero prob.
                    potential[hidden_state] = value
        else:
            for hidden_state in all_possible_hidden_states:
                if observed_state is None:
                    # singleton potential should be identically 1
                    potential[hidden_state] = 1.
                else:
                    value = observation_model(hidden_state)[observed_state]
                    if value > 0: # only store entries with nonzero prob.
                        potential[hidden_state] = value
        assert len(potential.keys()) > 0 , \
                "Invalid observation at time %d. Maybe you \
                forgot the --use-spread-output argument?"%n
        phis.append(potential)

    #-------------------------------------------------------------------------
    # Forward pass
    #

    # handle initial time step differently
    initial_message = {}
    for hidden_state in prior_distribution:
        value = -careful_log(phis[0][hidden_state])
        if value < float('inf'): # only store entries with nonzero prob.
            initial_message[hidden_state] = value
    messages.append(initial_message)

    # rest of the time steps
    for n in range(1, num_time_steps):
        prev_message     = messages[-1]
        new_message      = {}
        new_back_pointer = {}

        # only loop over hidden states with nonzero singleton potential!
        for hidden_state in phis[n]:
            values = []
            for prev_hidden_state in prev_message:
                value = prev_message[prev_hidden_state] \
                        - careful_log(transition_model(prev_hidden_state)[ \
                                          hidden_state]) \
                        - careful_log(phis[n][hidden_state])
                if value < float('inf'):
                    # only store entries with nonzero prob.
                    values.append((prev_hidden_state, value))

            if len(values) > 0:
                best_prev_hidden_state, best_value = \
                    min(values, key=lambda x: x[1])
                new_message[hidden_state]      = best_value
                new_back_pointer[hidden_state] = best_prev_hidden_state

        messages.append(new_message)
        back_pointers.append(new_back_pointer)

    #-------------------------------------------------------------------------
    # Backward pass (follow back-pointers)
    #
    estimated_hidden_states = []

    # handle last time step differently
    last_message = messages[-1]
    minimum      = np.inf
    arg_min      = None
    for hidden_state in last_message:
        if last_message[hidden_state] < minimum:
            minimum = last_message[hidden_state]
            arg_min = hidden_state
    estimated_hidden_states.append(arg_min)

    # rest of the time steps
    for n in range(num_time_steps - 2, -1, -1):
        next_back_pointers = back_pointers[n]
        best_hidden_state  = next_back_pointers[estimated_hidden_states[0]]
        estimated_hidden_states.insert(0, best_hidden_state)

    return estimated_hidden_states

def second_best(all_possible_hidden_states,
                all_possible_observed_states,
                prior_distribution,
                transition_model,
                observation_model,
                observations):
    """
    Inputs
    ------
    See the list of inputs for the function forward_backward() above.

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    num_time_steps = len(observations)

    # Basically for each (possible) hidden state at time step i, we need to
    # keep track of the best previous hidden state AND the second best
    # previous hidden state--where we need to keep track of TWO back pointers
    # per (possible) hidden state at each time step!

    messages       = [] # best values so far
    messages2      = [] # second-best values so far
    back_pointers  = [] # per time step per hidden state, we now need
                        # *two* back-pointers

    #-------------------------------------------------------------------------
    # Fold observations into singleton potentials
    #
    phis = [] # phis[n] is the singleton potential for node n
    for n in range(num_time_steps):
        potential      = Distribution()
        observed_state = observations[n]
        if n == 0:
            for hidden_state in prior_distribution:
                value = prior_distribution[hidden_state]
                if observed_state is not None:
                    value *= observation_model(hidden_state)[observed_state]
                if value > 0: # only store entries with nonzero prob.
                    potential[hidden_state] = value
        else:
            for hidden_state in all_possible_hidden_states:
                if observed_state is None:
                    # singleton potential should be identically 1
                    potential[hidden_state] = 1.
                else:
                    value = observation_model(hidden_state)[observed_state]
                    if value > 0: # only store entries with nonzero prob.
                        potential[hidden_state] = value
        phis.append(potential)

    #-------------------------------------------------------------------------
    # Forward pass
    #

    # handle initial time step differently
    initial_message = {}
    for hidden_state in prior_distribution:
        value = -careful_log(phis[0][hidden_state])
        if value < float('inf'): # only store entries with nonzero prob.
            initial_message[hidden_state] = value
    messages.append(initial_message)
    initial_message2 = {} # there is no second-best option
    messages2.append(initial_message2)

    # rest of the time steps
    for n in range(1, num_time_steps):
        prev_message      = messages[-1]
        prev_message2     = messages2[-1]
        new_message       = {}
        new_message2      = {}
        new_back_pointers = {} # need to store 2 per possible hidden state

        for hidden_state in phis[n]:
            # only look at possible hidden states given observation

            values = []
            # each entry in values will be a tuple of the form:
            # (<value>, <previous hidden state>,
            #  <which back pointer we followed>),
            # where <which back pointer we followed> is 0 (best back pointer)
            # or 1 (second-best back pointer)

            # iterate through best previous values
            for prev_hidden_state in prev_message:
                value = prev_message[prev_hidden_state] - \
                        careful_log(transition_model(prev_hidden_state)[ \
                                        hidden_state]) - \
                        careful_log(phis[n][hidden_state])
                if value < float('inf'):
                    # only store entries with nonzero prob.
                    values.append((value, prev_hidden_state, 0))

            # also iterate through second-best previous values
            for prev_hidden_state in prev_message2:
                value = prev_message2[prev_hidden_state] - \
                        careful_log(transition_model(prev_hidden_state)[ \
                                        hidden_state]) - \
                        careful_log(phis[n][hidden_state])
                if value < float('inf'):
                    # only store entries with nonzero prob.
                    values.append((value, prev_hidden_state, 1))

            if len(values) > 0:
                # this part could actually be sped up by not using a sorting
                # algorithm...
                sorted_values = sorted(values, key=lambda x: x[0])
                best_value, best_prev_hidden_state, which_back_pointer = \
                    sorted_values[0]

                # for the best value, the back pointer should *always* be 0,
                # meaning that we follow the best back pointer and not the
                # second best

                if len(values) > 1:
                    best_value2, best_prev_hidden_state2, which_back_pointer2\
                        = sorted_values[1]
                else:
                    best_value2             = float('inf')
                    best_prev_hidden_state2 = None
                    which_back_pointer2     = None

                new_message[hidden_state]       = best_value
                new_message2[hidden_state]      = best_value2
                new_back_pointers[hidden_state] = \
                    ( (best_prev_hidden_state, which_back_pointer),
                      (best_prev_hidden_state2, which_back_pointer2) )

        messages.append(new_message)
        messages2.append(new_message2)
        back_pointers.append(new_back_pointers)

    #-------------------------------------------------------------------------
    # Backward pass (follow back-pointers)
    #
    estimated_hidden_states = []

    # handle last time step differently
    values = []
    for hidden_state, value in messages[-1].iteritems():
        values.append( (value, hidden_state, 0) )
    for hidden_state, value in messages2[-1].iteritems():
        values.append( (value, hidden_state, 1) )

    if len(values) > 1:
        # this part could actually be sped up by not using a sorting
        # algorithm...
        sorted_values = sorted(values, key=lambda x: x[0])
        second_best_value, hidden_state, which_back_pointer = sorted_values[1]

        estimated_hidden_states.append(hidden_state)

        # rest of the time steps
        for n in range(num_time_steps - 2, -1, -1):
            next_back_pointers = back_pointers[n]
            hidden_state, which_back_pointer = \
                next_back_pointers[hidden_state][which_back_pointer]
            estimated_hidden_states.insert(0, hidden_state)
    else:
        # this happens if there isn't a second best option, which should mean
        # that the only possible option (the MAP estimate) is the only
        # solution with 0 error
        estimated_hidden_states = [None] * num_time_steps

    return estimated_hidden_states


#-----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(initial_distribution, transition_model, observation_model,
                  num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from a hidden Markov model given an initial
    # distribution, transition model, observation model, and number of time
    # steps, generate samples from the corresponding hidden Markov model
    hidden_states = []
    observations  = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state       = initial_distribution().sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state   = hidden_states[-1]
        new_state    = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1: # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


#-----------------------------------------------------------------------------
# Main
#

if __name__ == '__main__':
    # flags
    make_some_observations_missing = False
    use_graphics                   = True
    need_to_generate_data          = True
    use_spread_output              = False

    # parameters

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]

            grid_width, grid_height, hidden_states, observations \
                = robot.load_data(filename)

            robot.GRID_WIDTH      = grid_width
            robot.GRID_HEIGHT     = grid_height
            graphics.GRID_WIDTH   = grid_width
            graphics.GRID_HEIGHT  = grid_height
            need_to_generate_data = False
            num_time_steps        = len(hidden_states)
        elif arg == '--use-spread-output':
            use_spread_output = True
        else:
            raise ValueError("I don't know how to handle argument %s"%arg)


    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 10
        hidden_states, observations = \
            generate_data(robot.initial_distribution,
                          robot.transition_model,
                          robot.observation_model,
                          num_time_steps,
                          make_some_observations_missing)

    all_possible_hidden_states   = robot.get_all_hidden_states()
    all_possible_observed_states = robot.get_all_observed_states()
    prior_distribution           = robot.initial_distribution()

    if use_spread_output:
        observation_model = robot.spread_observation_model
    else:
        observation_model = robot.observation_model
    print 'Running forward-backward...'
    (marginals, pairwise_marginals) = forward_backward(
                                 all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 robot.transition_model,
                                 observation_model,
                                 observations)
    print

    print "Marginal at time %d:" % (num_time_steps - 1)
    if marginals[-1] is not None:
        print marginals[-1]
    else:
        print '*No marginal computed*'
    print

    print 'Running Viterbi...'
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               robot.transition_model,
                               observation_model,
                               observations)
    print

    print "Last 10 hidden states in the MAP estimate:"
    for time_step in range(num_time_steps - 10, num_time_steps):
        if estimated_states[time_step] is None:
            print 'Missing'
        else:
            print estimated_states[time_step]
    print


    difference = 0
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
    print "Number of differences between MAP estimate and true hidden " + \
          "states:", difference

    mse = robot.compute_average_distance_between(estimated_states,
                                                 hidden_states)

    print "Mean distance between MAP estimate and true hidden states:", mse

    # display
    if use_graphics:
        # Don't run the GUI for too long
        gui_limit_steps = 100
        app = graphics.playback_positions(hidden_states[:gui_limit_steps],
                                          observations[:gui_limit_steps],
                                          estimated_states[:gui_limit_steps],
                                          marginals[:gui_limit_steps])
        app.mainloop()

