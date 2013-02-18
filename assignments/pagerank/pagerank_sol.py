from __future__ import division
import sys
import random

import util

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
    states = conditional_distribution.keys()
    state = random.choice(states)
    samples = []
    i = 0
    while True:
        i += 1
        if random.random() < .1:
            state = random.choice(states)
        else:
            state = conditional_distribution[state].sample()
        if (i % jumps_between_samples) == 0:
            samples.append(state)
            if len(samples) >= N_samples:
                break

    # Construct empirical distribution from samples
    empirical_distribution = util.Distribution()
    for s in samples:
        empirical_distribution[s] += 1
    empirical_distribution.renormalize()
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
    out = {}
    counts = []
    lengths = []
    for actor in actor_to_movies:
        conditional_distribution = util.Distribution()
        for movie in actor_to_movies[actor]:
            lengths.append(len(movie_to_actors[movie]))
            for co_star in movie_to_actors[movie]:
                conditional_distribution[co_star] += 1
        counts.extend(conditional_distribution.values())
        conditional_distribution.renormalize()
        out[actor] = conditional_distribution
    return out

def read_file(filename):
    """
    Reads in a file with actors and movies they starred in, and returns two
    dictionaries: one mapping actors to lists of movies they starred in, and one
    mapping movies to lists of actors that were in them.

    The file should have the following format:

    <Actor 1>
            <Movie 1 for actor 1>
            ...
    <Actor 2>
            <Movie 2 for actor 1>
            ...
    ...

    Actor lines should have no whitespace at the front, and movie lines
    must have whitespace at the front.
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

def top_actors(actor_to_movies):
    import operator
    pairs = [(actor, len(movies)) for (actor, movies) in actor_to_movies.items()]
    pairs.sort(key=operator.itemgetter(1), reverse=True)
    print('Top actors and # of movies they starred in:')
    for (actor, count) in pairs[:10]:
        print('%s: %d' % (actor, count))

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python pagerank.py <data file> <samples> <iterations between samples>")
        sys.exit(1)
    data_filename = sys.argv[1]
    N_samples = int(sys.argv[2])
    iterations_between_samples = int(sys.argv[3])

    (actor_to_movies, movie_to_actors) = read_file(data_filename)
    conditional_distribution = compute_distributions(actor_to_movies,
                                                     movie_to_actors)

    top_actors(actor_to_movies)
    steady_state = pagerank(conditional_distribution,
                            N_samples,
                            iterations_between_samples)

    actors = actor_to_movies.keys()
    top = sorted( ( ((steady_state[actor]), actor) for actor in actors) )

    values_to_show = min(20, len(steady_state))
    print("Top %d values from empirical distribution:" % values_to_show)

    for i in xrange(1, values_to_show+1):
        print("%0.6f: %s" %top[-i])


