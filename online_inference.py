# online_inference.py
# Base code by George H. Chen (georgehc@mit.edu)
# Modified by: <your name here!>
import sys
import numpy as np
import robot
import graphics
from utils import Distribution


#-----------------------------------------------------------------------------
# Functions for you to implement
#

def compute_marginal(particles, weights):
    """
    Essentially computes an *empirical* distribution given particles and
    weights

    Inputs
    ------
    particles: a list where each element is a hidden state value

    weights: a list where element i is the weight for particle i

    Output
    ------
    A Distribution, where each hidden state has probability proportional to
    the total weight for that hidden state (which may be from multiple
    particles)
    """
    marginal = Distribution()

    # TODO: Your code here
    raise NotImplementedError

    return marginal

def resample(particles, weights):
    """
    Resample from the empirical distribution of weighted particles to obtain a
    new set of (unweighted, i.e., weights are all the same) particles

    Inputs
    ------
    Same inputs as function compute_marginal

    Output
    ------
    new_particles: new list of particles

    new_weights: new list of weights

    The new particles and new weights result from resampling from the marginal
    distribution
    """
    num_particles = len(particles)
    new_weights   = [1. for i in range(num_particles)]

    # TODO: Your code here
    raise NotImplementedError

    return new_particles, new_weights

def particle_filter(prior_distribution,
                    transition_model,
                    observation_model,
                    observations,
                    num_particles=500):
    """
    Inputs
    ------
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state

    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state

    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    This function is a Python generator! It calculates outputs "on demand";
    the i-th output should be the marginal distribution for time step i.

    Please see these two pages on Python generators for more information:
      http://wiki.python.org/moin/Generators
      http://docs.python.org/2/tutorial/classes.html#generators

    For an example of a Python generator function, check out the function
    forward (for the forward algorithm) below.

    Key point: The output is *not* a list, so you can't use brackets [] to
    access the i-th entry. We've provided skeletal code so that it should be
    clear what you need to fill in.
    """
    num_time_steps = len(observations)

    # time step 0
    initial_marginal = robot.Distribution()
    # TODO: Your code here
    raise NotImplementedError

    yield initial_marginal # do not modify this line

    # remaining time steps
    for n in range(1, num_time_steps):
        marginal = robot.Distribution()
        # TODO: Your code here
        raise NotImplementedError

        yield marginal # do not modify this line


#-----------------------------------------------------------------------------
# The forward algorithm
#

def forward(all_possible_hidden_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states

    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state

    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state

    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    This function is a Python generator! It calculates outputs "on demand";
    the i-th output should be the marginal distribution for time step i.
    """
    num_time_steps = len(observations)

    #-------------------------------------------------------------------------
    # Forward pass
    #
    def compute_marginal(message, node_potential):
        marginal = Distribution()
        for hidden_state in all_possible_hidden_states:
            if hidden_state in message and \
               hidden_state in node_potential:
                value = message[hidden_state] * node_potential[hidden_state]
                if value > 0: # only store entries with nonzero prob.
                    marginal[hidden_state] = value
        marginal.renormalize()
        return marginal

    # compute message from non-existent node -1 to node 0
    message = Distribution()
    for hidden_state in all_possible_hidden_states:
        message[hidden_state] = 1.
    message.renormalize()

    # compute node potential for time step 0
    node_potential = Distribution()
    observed_state = observations[0]
    for hidden_state in prior_distribution:
        value = prior_distribution[hidden_state]
        if observed_state is not None:
            value *= observation_model(hidden_state)[observed_state]
        if value > 0:
            node_potential[hidden_state] = value

    yield compute_marginal(message, node_potential)
    prev_message        = message
    prev_node_potential = node_potential

    for n in range(1, num_time_steps):
        message = Distribution()

        node_potential = Distribution()
        observed_state = observations[n]

        # compute message from node n-1 to n and fill in node potential for
        # time step n
        for hidden_state in all_possible_hidden_states:
            # only loop over possible next hidden states given current
            # hidden state
            for next_hidden_state in transition_model(hidden_state):
                factor = prev_node_potential[hidden_state] * \
                         transition_model(hidden_state)[next_hidden_state] * \
                         prev_message[hidden_state]
                if factor > 0: # only store entries with nonzero prob.
                    if next_hidden_state in message:
                        message[next_hidden_state] += factor
                    else:
                        message[next_hidden_state] = factor

            if observed_state is not None:
                value = observation_model(hidden_state)[observed_state]
                if value > 0:
                    node_potential[hidden_state] = value
            else:
                node_potential[hidden_state] = 1.

        message.renormalize()
        yield compute_marginal(message, node_potential)
        prev_message        = message
        prev_node_potential = node_potential


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
    use_forward_algorithm          = False

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg == '--forward':
            use_forward_algorithm = True
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
        else:
            raise ValueError("I don't know how to handle argument %s" % arg)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(robot.initial_distribution,
                          robot.transition_model,
                          robot.observation_model,
                          num_time_steps,
                          make_some_observations_missing)

    all_possible_hidden_states = robot.get_all_hidden_states()
    prior_distribution         = robot.initial_distribution()
    transition_model           = robot.transition_model
    observation_model          = robot.discretized_gaussian_observation_model

    if not use_forward_algorithm:
        marginals = particle_filter(prior_distribution,
                                    transition_model,
                                    observation_model,
                                    observations)
    else:
        marginals = forward(all_possible_hidden_states,
                            prior_distribution,
                            transition_model,
                            observation_model,
                            observations)

    # display
    if use_graphics:
        app = graphics.playback_positions_live(hidden_states,
                                               observations,
                                               marginals)
        app.mainloop()

