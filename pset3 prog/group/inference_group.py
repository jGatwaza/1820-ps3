# inference.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# This code has been modified and extended for CS 1820 at Harvard University, 
# with adjustments tailored to align with the course curriculum and objectives.


from typing import List, Dict 

import bayesNet as bn
from bayesNet import normalize

from bayesNet import Factor
import random

# Rejection Sampling

def priorSample(bayesNet):
    """
    Generate a single sample from the prior distribution defined by the Bayes Net.

    bayesNet: The Bayes Net to sample from.

    Returns a complete assignment dict {variable: value}.
    """
    "*** YOUR CODE HERE ***"


def approximateInferenceByRejectionSampling(bayesNet, queryVariables, evidenceDict, numSamples):
    """
    Perform approximate probabilistic inference using rejection sampling.

    bayesNet:       The Bayes Net on which we are making a query.
    queryVariables: A list of the variables which are unconditioned in the inference query.
    evidenceDict:   An assignment dict {variable: value} for the variables that are evidence.
    numSamples:     The number of samples to generate.

    Returns a factor with the probabilities of the query variables.
    """
    "*** YOUR CODE HERE ***"


# Likelihood Weighting

def weightedSample(bayesNet, evidenceDict):
    """
    Generate a single weighted sample from the Bayes Net given the evidence.

    bayesNet:    The Bayes Net to sample from.
    evidenceDict: A dict {variable: value} for the evidence.

    Returns a tuple (sample, weight):
        - sample: A complete assignment dict {variable: value} for all variables in the net.
        - weight: The likelihood weight of this sample.
    """
    "*** YOUR CODE HERE ***"


def approximateInferenceByLikelihoodWeighting(bayesNet, queryVariables, evidenceDict, numSamples):
    """
    Perform approximate probabilistic inference using likelihood weighting.

    bayesNet:       The Bayes Net on which we are making a query.
    queryVariables: A list of the variables which are unconditioned in the query.
    evidenceDict:   A dict {variable: value} for evidence variables.
    numSamples:     The number of weighted samples to generate.

    Returns a Factor over queryVariables representing P(queryVariables | evidence).
    """
    "*** YOUR CODE HERE ***"


