from __future__ import division
import sys
import os.path
import numpy as np

import collections
import util

USAGE = "%s <test data folder> <spam folder> <ham folder>"

def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    counts = util.Counter()
    for f in file_list:
        words = util.get_words_in_file(f)
        for w in set(words):
            counts[w] += 1
    return counts

def get_log_probabilities(file_list):
    """
    Computes log-frequencies for each word that occurs in the files in 
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    counts = get_counts(file_list)
    N_files = len(file_list)
    N_categories = 2
    # log_prob = util.DefaultDict(lambda : -np.log(N_files + N_categories))
    log_prob = collections.defaultdict(lambda : -np.log(N_files + N_categories))
    for word in counts:
        log_prob[word] = np.log(counts[word] + 1) - np.log(N_files + N_categories)
        assert log_prob[word] < 0
    return log_prob


def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files, 
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    log_probs_by_category = []
    prior_by_category = []

    for file_list in file_lists_by_category:
        log_prob = get_log_probabilities(file_list)
        log_probs_by_category.append(log_prob)
        prior_by_category.append(len(file_list))

    log_prior_by_category = [np.log(p/sum(prior_by_category)) for p in prior_by_category]

    return (log_probs_by_category, log_prior_by_category)

def classify_message(message_filename,
                     log_probabilities_by_category,
                     log_prior_by_category,
                     names = ['spam', 'ham']):
    """
    Uses Naive Bayes classification to classify the message in the given file.

    Inputs
    ------
    message_filename : name of the file containing the message to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    names : labels for each class (for this problem set, will always be just 
            spam and ham).

    Output
    ------
    One of the labels in names.
    """

    message_words = util.get_words_in_file(message_filename)
    N_categories = len(log_probabilities_by_category)

    log_likelihoods = []
    for i in xrange(N_categories):
        total = 0
        all_word_log_probs = log_probabilities_by_category[i].copy()

        # for w in all_word_log_probs.keys():
        #     log_prob = all_word_log_probs[w]
        #     test = (w in message_words)
        #     old_total += test*log_prob + (1-test)*np.log(1-np.exp(log_prob))

        all_word_log_probs = log_probabilities_by_category[i].copy()
        for w in message_words:
            total += all_word_log_probs[w]
            all_word_log_probs.pop(w)

        total += np.sum(np.log(1-np.exp(all_word_log_probs.values())))


        log_likelihoods.append(total)

    posterior = np.array(log_likelihoods) + np.array(log_prior_by_category)
    winner = np.argmax(posterior)
    return names[winner]

if __name__ == '__main__':
    ### Read arguments
    if len(sys.argv) != 4:
        print USAGE % sys.argv[0]
    testing_folder = sys.argv[1]
    (spam_folder, ham_folder) = sys.argv[2:4]

    ### Learn the distributions
    print("Training...")
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists)

    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    print("Testing...")
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_message(filename,
                                 log_probabilities_by_category,
                                 log_priors_by_category,
                                 ['spam', 'ham'])
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[true_index, guessed_index] += 1


        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        #print("%s : %s" %(label, filename))

    template="""You correctly classified:
        * %d out of %d (%0.1f%%) spam messages
        * %d out of %d (%0.1f%%) ham messages"""
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[0] / totals[0] * 100,
                      correct[1],
                      totals[1],
                      correct[1] / totals[1] * 100))

