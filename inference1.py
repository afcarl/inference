# inference.py
# Base code by George H. Chen (georgehc@csail.mit.edu)
# Modified by: <your name here!>
import sys
import numpy as np
import robot
import graphics


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

    # TODO: This is for you to implement

    num_time_steps = len(observations)
    marginals = [None] * num_time_steps # remove this

    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: This is for you to implement

    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

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
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: This is for you to implement

    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

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

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(robot.initial_distribution,
                          robot.transition_model,
                          robot.observation_model,
                          num_time_steps,
                          make_some_observations_missing)

    all_possible_hidden_states   = robot.get_all_hidden_states()
    all_possible_observed_states = robot.get_all_observed_states()
    prior_distribution           = robot.initial_distribution()

    print 'Running forward-backward...'
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 robot.transition_model,
                                 robot.observation_model,
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
                               robot.observation_model,
                               observations)
    print

    print "Last 10 hidden states in the MAP estimate:"
    for time_step in range(num_time_steps - 10, num_time_steps):
        if estimated_states[time_step] is None:
            print 'Missing'
        else:
            print estimated_states[time_step]
    print

    print 'Finding second-best MAP estimate...'
    estimated_states2 = second_best(all_possible_hidden_states,
                                    all_possible_observed_states,
                                    prior_distribution,
                                    robot.transition_model,
                                    robot.observation_model,
                                    observations)
    print

    print "Last 10 hidden states in the second-best MAP estimate:"
    for time_step in range(num_time_steps - 10, num_time_steps):
        if estimated_states2[time_step] is None:
            print 'Missing'
        else:
            print estimated_states2[time_step]
    print

    print "Number of differences between MAP estimate and true hidden " + \
          "states:", \
          robot.compute_number_of_differences(estimated_states,
                                              hidden_states)
    print "Number of differences between second-best MAP estimate and " + \
          "true hidden states:", \
          robot.compute_number_of_differences(estimated_states2,
                                              hidden_states)
    print "Number of differences between MAP and second-best MAP " + \
          "estimates:", \
          robot.compute_number_of_differences(estimated_states,
                                              estimated_states2)

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states)
        app.mainloop()

