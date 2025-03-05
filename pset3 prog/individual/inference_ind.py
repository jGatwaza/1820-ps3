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


##################################### 
#                                   #
# Helper Methods - DO NOT MODIFY!!! #
#                                   #
##################################### 

def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
    """
    Input factors is a list of factors.
    Input joinVariable is the variable to join on.

    This function performs a check that the variable that is being joined on 
    appears as an unconditioned variable in only one of the input factors.

    Then, it calls your joinFactors on all of the factors in factors that 
    contain that variable.

    Returns a tuple of 
    (factors not joined, resulting factor from joinFactors)
    """

    currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
    currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

    # typecheck portion
    numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
    vars_on_left = [factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()]
    if numVariableOnLeft > 1:
        print("Factor failed joinFactorsByVariable typecheck: ", vars_on_left)
        raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                            "joinVariable: " + str(joinVariable) + "\n" +
                            ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
    
    joinedFactor = joinFactors(currentFactorsToJoin)
    return currentFactorsNotToJoin, joinedFactor


def inferenceByEnumeration(bayesNet: bn, queryVariables: List[str], evidenceDict: Dict):
    """
    An inference by enumeration implementation provided as reference.
    This function performs a probabilistic inference query that
    returns the factor:

    P(queryVariables | evidenceDict)

    bayesNet:       The Bayes Net on which we are making a query.
    queryVariables: A list of the variables which are unconditioned in
                    the inference query.
    evidenceDict:   An assignment dict {variable : value} for the
                    variables which are presented as evidence
                    (conditioned) in the inference query. 
    """
    # initialize return variables and the variables to eliminate
    evidenceVariablesSet = set(evidenceDict.keys())
    queryVariablesSet = set(queryVariables)
    eliminationVariables = (bayesNet.variablesSet() - evidenceVariablesSet) - queryVariablesSet

    # grab all factors where we know the evidence variables (to reduce the size of the tables)
    currentFactorsList = bayesNet.getAllCPTsWithEvidence(evidenceDict)

    # join all factors by variable
    for joinVariable in bayesNet.variablesSet():
        currentFactorsList, joinedFactor = joinFactorsByVariable(currentFactorsList, joinVariable)
        currentFactorsList.append(joinedFactor)

    # currentFactorsList should contain the connected components of the graph now as factors, must join the connected components
    fullJoint = joinFactors(currentFactorsList)

    # marginalize all variables that aren't query or evidence
    incrementallyMarginalizedJoint = fullJoint
    for eliminationVariable in eliminationVariables:
        incrementallyMarginalizedJoint = eliminate(incrementallyMarginalizedJoint, eliminationVariable)

    fullJointOverQueryAndEvidence = incrementallyMarginalizedJoint

    # normalize so that the probability sums to one
    queryConditionedOnEvidence = normalize(fullJointOverQueryAndEvidence)

    return queryConditionedOnEvidence

###################
#                 #
# YOUR CODE BELOW #
#                 #
###################

def joinFactors(factors: List[Factor]):
    """
    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    "*** YOUR CODE HERE ***"



def eliminate(factor: Factor, eliminationVariable: str):
    """
    Input factor is a single factor.
    Input eliminationVariable is the variable to eliminate from factor.
    eliminationVariable must be an unconditioned variable in factor.
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the factor obtained by eliminating the variable
    eliminationVariable.

    Return a new factor where all of the rows mentioning
    eliminationVariable are summed with rows that match
    assignments on the other variables.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    "*** YOUR CODE HERE ***"
    


def inferenceByVariableElimination(bayesNet: bn, queryVariables: List[str], evidenceDict: Dict):
    """
    This function should perform a probabilistic inference query that
    returns the factor:

    P(queryVariables | evidenceDict)

    It should perform inference by interleaving joining on a variable
    and eliminating that variable, in the order of variables according
    to eliminationOrder.  See inferenceByEnumeration for an example on
    how to use these functions.

    You need to use joinFactorsByVariable to join all of the factors 
    that contain a variable in order for the autograder to 
    recognize that you performed the correct interleaving of 
    joins and eliminates.

    If a factor that you are about to eliminate a variable from has 
    only one unconditioned variable, you should not eliminate it 
    and instead just discard the factor.  This is since the 
    result of the eliminate would be 1 (you marginalize 
    all of the unconditioned variables), but it is not a 
    valid factor.  So this simplifies using the result of eliminate.

    The sum of the probabilities should sum to one (so that it is a true 
    conditional probability, conditioned on the evidence).

    bayesNet:         The Bayes Net on which we are making a query.
    queryVariables:   A list of the variables which are unconditioned
                      in the inference query.
    evidenceDict:     An assignment dict {variable : value} for the
                      variables which are presented as evidence
                      (conditioned) in the inference query. 
    eliminationOrder: The order to eliminate the variables in.

    Hint: BayesNet.getAllCPTsWithEvidence will return all the Conditional 
    Probability Tables even if an empty dict (or None) is passed in for 
    evidenceDict. In this case it will not specialize any variable domains 
    in the CPTs.

    Useful functions:
    BayesNet.getAllCPTsWithEvidence
    normalize
    eliminate
    joinFactorsByVariable
    joinFactors
    """

    "*** YOUR CODE HERE ***"
