from __future__ import division
import sys
import random

import util

USAGE = "Usage: python %s <data file> <samples> <iterations between samples>"

def pagerank(conditional_distribution, N_samples, jumps_between_samples):
    """
    Computes the steady-state distribution by simulating running the Markov
    chain. Collects samples at regular intervals and returns the empirical
    distribution of the samples.

    Inputs
    ------
    conditional_distribution : A dictionary in which each key is an state,
                               and each value is a Distribution over other
                               states.

    N_samples : the desired number of samples for the approximate empirical
                distribution
    jumps_between_samples : how many jumps to perform between each collected
                            sample

    Returns
    -------
    An empirical Distribution over the states that should approximate the
    steady-state distribution.
    """
    ### YOUR CODE HERE
    empirical_distribution = util.Distribution()
    return empirical_distribution

def compute_distributions(actor_to_movies, movie_to_actors):
    """
    Computes conditional distributions for transitioning
    between actors (states).

    Inputs
    ------
    actor_to_movies : a dictionary in which each key is an actor name and each
                      value is a list of movies that actor starred in

    movie_to_actors : a dictionary in which each key is a movie and each
                      value is a list of actors in that movie

    Returns
    -------
    A dictionary in which each key is an actor, and each value is a
    Distribution over other actors. The probability of transitioning
    from actor i to actor j should be proportional to the number of
    movies they starred in together.
    """
    ### YOUR CODE HERE
    return {}

def read_file(filename):
    """
    Reads in a file with actors and movies they starred in, and returns two
    dictionaries: one mapping actors to lists of movies they starred in, and one
    mapping movies to lists of actors that were in them.

    The file should have the following format:

    <Actor 1>
            <Movie 1 for actor 1>
            <Movie 2 for actor 1>
            ...
    <Actor 2>
            <Movie 1 for actor 2>
            ...
    ...

    Actor lines should have no whitespace at the front, and movie lines
    must have whitespace (of any kind) at the front.
    """
    actor_to_movies = util.DefaultDict(lambda : [])
    movie_to_actors = util.DefaultDict(lambda : [])
    with open(filename) as f:
        for line in f:
            if line[0] != ' ':
                actor = line.strip()
            else:
                movie = line.strip()
                actor_to_movies[actor].append(movie)
                movie_to_actors[movie].append(actor)
    return (actor_to_movies, movie_to_actors)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(USAGE % sys.argv[0])
        sys.exit(1)
    data_filename = sys.argv[1]
    N_samples = int(sys.argv[2])
    iterations_between_samples = int(sys.argv[3])

    (actor_to_movies, movie_to_actors) = read_file(data_filename)
    conditional_distribution = compute_distributions(actor_to_movies,
                                                     movie_to_actors)

    steady_state = pagerank(conditional_distribution,
                            N_samples,
                            iterations_between_samples)

    actors = actor_to_movies.keys()
    top = sorted( ( ((steady_state[actor]), actor) for actor in actors) )

    values_to_show = min(20, len(steady_state))
    print("Top %d values from empirical distribution:" % values_to_show)
    for i in xrange(1, values_to_show+1):
        print("%0.6f: %s" %top[-i])

