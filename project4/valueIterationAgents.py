# Kevin Ding
# kevin.ding2@emory.edu/ ydin226/ 2397009
# THIS CODE WAS MY OWN WORK , IT WAS WRITTEN WITHOUT CONSULTING ANY
# SOURCES OUTSIDE OF THOSE APPROVED BY THE INSTRUCTOR. Kevin Ding






# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """

          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        states=self.mdp.getStates()

        for i in range(0,self.iterations):
            updatedValues=self.values.copy()

            for state in states:
                actions=self.mdp.getPossibleActions(state)
                qValues=util.Counter()

                for action in actions:
                    qValue=self.computeQValueFromValues(state,action)
                    qValues[action]=qValue

                ## select the maximum q value to be V
                maxAction=qValues.argMax()
                maxQValue=qValues[maxAction]
                updatedValues[state]=maxQValue

            self.values=updatedValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        q=0

        ## loop over all possible s'
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state,action):
            currentReward=self.mdp.getReward(state,action,nextState)
            futureReward=self.discount*self.getValue(nextState)
            q+=prob*(currentReward+futureReward)

        return q




    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        if self.mdp.isTerminal(state):
            return None

        qValues=util.Counter()
        actions=self.mdp.getPossibleActions(state)
        for action in actions:
            qValue=self.computeQValueFromValues(state,action)
            qValues[action]=qValue

        maxPolicy=qValues.argMax()
        return maxPolicy









    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
