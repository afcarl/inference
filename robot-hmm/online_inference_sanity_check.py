# sanity_check.py
# Coded by George H. Chen (georgehc@mit.edu)
import sys
import robot
from utils import KL_divergence
from online_inference import particle_filter, forward, generate_data


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
            need_to_generate_data = False
            num_time_steps        = len(hidden_states)
        else:
            raise ValueError("I don't know how to handle argument %s"%arg)

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

    pf_marginals = particle_filter(prior_distribution,
                                   transition_model,
                                   observation_model,
                                   observations)

    marginals = forward(all_possible_hidden_states,
                        prior_distribution,
                        transition_model,
                        observation_model,
                        observations)

    for n in range(num_time_steps):
        pf_marginal   = pf_marginals.next()
        true_marginal = marginals.next()
        print "Time step %d: KL divergence D(particle filter||true)=%f" \
              % (n, KL_divergence(pf_marginal, true_marginal))

