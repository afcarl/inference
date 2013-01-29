# learning.py
# Base code by Ramesh Sridharan (rameshvs@mit.edu)
from __future__ import division
import sys

import robot
import graphics
import inference

def baum_welch(all_possible_hidden_states,
               all_possible_observed_states,
               observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states

    all_possible_observed_states: a list of possible observed states

    observations: a list of observations, one per hidden state
                  (in this problem, we don't have any missing observations)


    Output
    ------
    A transition model and an observation model
    """
    ### Initialize
    transition_model = robot.uniform_transition_model
    observation_model = robot.spread_observation_model
    initial_state_distribution = robot.initial_distribution()

    num_time_steps = len(observations)

    convergence_reached = False
    while not convergence_reached:
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        ### YOUR CODE HERE: Estimate marginals & pairwise marginals
        pairwise_marginals = [None] * num_time_steps
        marginals = [None] * num_time_steps
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        ### Learning: estimate model parameters
        # maps states to distributions over possible next states
        transition_model_dict = {}

        # maps states to distributions over possible observations
        observation_model_dict = {}

        # initial
        initial_state_distribution = robot.Distribution()

        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # YOUR CODE HERE: Use the estimated marginals & pairwise marginals to
        # estimate the parameters.

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        new_transition_model = robot.make_transition_model(transition_model_dict)
        new_obs_model = robot.make_observation_model(observation_model_dict)
        ### Check for convergence
        if check_convergence(all_possible_hidden_states,
                             transition_model,
                             new_transition_model):
            convergence_reached = True
        transition_model = new_transition_model
        observation_model = new_obs_model

    return (transition_model,
            observation_model,
            initial_state_distribution,
            marginals)


def check_convergence(all_possible_hidden_states, 
                      old_transition_model,
                      new_transition_model):

    sum_so_far = 0
    for state in all_possible_hidden_states:
        old = old_transition_model(state)
        new = new_transition_model(state)
        possible_next_states = set(old.keys()) | set(new.keys())
        for next_state in possible_next_states:
            sum_so_far += ( old[next_state] - new[next_state] )**2
    return sum_so_far < 0.1

if __name__ == '__main__':

    need_to_generate_data          = True
    use_graphics                   = True

    for arg in sys.argv[1:]:
        if arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 1000
        hidden_states, observations = \
            inference.generate_data(robot.initial_distribution,
                                    robot.transition_model,
                                    robot.observation_model,
                                    num_time_steps,
                                    make_some_observations_missing=False)

    all_possible_hidden_states   = robot.get_all_hidden_states()
    all_possible_observed_states = robot.get_all_observed_states()
    prior_distribution           = robot.initial_distribution()

    print 'Running Baum-Welch...'
    (transition_model,
     observation_model,
     prior_distribution,
     marginals)                = baum_welch(all_possible_hidden_states,
                                            all_possible_observed_states,
                                            observations)

    print 'Running Viterbi with estimated parameters...'
    estimated_states = inference.Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               transition_model,
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
