
# Kevin Ding
# kevin.ding2@emory.edu / ydin226/ 2397009
# THIS CODE WAS MY OWN WORK , IT WAS WRITTEN WITHOUT CONSULTING ANY
# SOURCES OUTSIDE OF THOSE APPROVED BY THE INSTRUCTOR. Kevin Ding





# multiAgents.py
# --------------
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
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    '''
    comment on problem 1
    
    My evaluation score is composed of 3 parts. The first part is the score of the current game state, which I
    treat it as a base score. Next is what I call penalty. If pacman is less than the safe distance (5 unit)
    from either of the ghost, then essentially I return negative infinity, meaning I would not consider
    this to be my move. The last feature considered is the distance from pacman's current position
    to the nearest food and I take the reciprocal of that distance because the shorter the distance is,
    the better the move will be.
    
    '''
    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition() # new position after moving
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor) 
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        # too dangerous to be within safe distance (5 unit) with either of the ghost
        # so avoid this option
        penalty=0
        for ghost in successorGameState.getGhostPositions():
            safe_distance=5
            if (manhattanDistance(newPos, ghost) < safe_distance):
                penalty=-float("inf")
                break


        # if there is food left, get the distance to the nearest food
        min_distance=float("inf")
        for food in newFood.asList():
            distance=manhattanDistance(newPos,food)
            if distance<min_distance:
                min_distance=distance

        return successorGameState.getScore()+penalty+1.0/min_distance




def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)






# question 2
class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        value,action=self.min_max_search(gameState,0,0)
        return action

    def min_max_search(self,gameState,depth,agent):
        is_terminal=(gameState.isWin() or gameState.isLose() or depth==self.depth)
        is_pacman=(agent==0)
        if is_terminal:
            return (self.evaluationFunction(gameState),None)
        if is_pacman:
            return self.max_pacman(gameState,depth,agent)
        return self.min_ghost(gameState,depth,agent)



    def max_pacman(self,gameState,depth,agent):
        moves=gameState.getLegalActions()
        options=[]

        for move in moves:
            s=gameState.generateSuccessor(agent,move)
            s_depth=depth
            s_index=agent+1
            pacman_move=(s_index==gameState.getNumAgents())

            if pacman_move:
                s_depth+=1
                s_index=0

            options.append((self.min_max_search(s,s_depth,s_index)[0],move))
        return max(options,key=lambda entry:entry[0])



    def min_ghost(self,gameState,depth,agent):
        moves=gameState.getLegalActions(agent)
        options=[]

        for move in moves:
            s=gameState.generateSuccessor(agent,move)
            s_depth=depth
            s_index=agent+1
            pacman_move=(s_index==gameState.getNumAgents())

            if pacman_move:
                s_depth+=1
                s_index=0

            options.append((self.min_max_search(s,s_depth,s_index)[0],move))
        return min(options,key=lambda entry:entry[0])


# question 3
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha=-float("inf")
        beta=float("inf")

        value, pair = self.min_max_search(gameState, 0, 0,alpha,beta)
        return pair[1]

    def min_max_search(self, gameState, depth, agent,alpha,beta):
        is_terminal = (gameState.isWin() or gameState.isLose() or depth == self.depth)
        is_pacman = (agent == 0)
        if is_terminal:
            return (self.evaluationFunction(gameState), None)
        if is_pacman:
            return self.max_pacman(gameState, depth, agent,alpha,beta)
        return self.min_ghost(gameState, depth, agent,alpha,beta)

    def max_pacman(self, gameState, depth, agent,alpha,beta):
        moves = gameState.getLegalActions()
        options = []
        v=-float("inf")

        for move in moves:
            s = gameState.generateSuccessor(agent, move)
            s_depth = depth
            s_index = agent + 1
            pacman_move = (s_index == gameState.getNumAgents())

            if pacman_move:
                s_depth += 1
                s_index = 0

            # prune part
            current=self.min_max_search(s, s_depth, s_index,alpha,beta)
            value,action=current
            options.append((value, move))
            v=max(v,value)
            if v>beta:
                return (v,move)
            alpha=max(alpha,v)

        return (v,max(options, key=lambda entry: entry[0]))


    def min_ghost(self, gameState, depth, agent,alpha,beta):
        moves = gameState.getLegalActions(agent)
        options = []
        v=float("inf")

        for move in moves:
            s = gameState.generateSuccessor(agent, move)
            s_depth = depth
            s_index = agent + 1
            pacman_move = (s_index == gameState.getNumAgents())

            if pacman_move:
                s_depth += 1
                s_index = 0

            # prune part
            current=self.min_max_search(s, s_depth, s_index,alpha,beta)
            value,action=current
            options.append((value, move))
            v=min(v,value)
            if v<alpha:
                return ((v,move))
            beta=min(beta,v)
        return (v,min(options, key=lambda entry: entry[0]))



# question 4
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        value, action = self.min_max_search(gameState, 0, 0)
        return action

    def min_max_search(self, gameState, depth, agent):
        is_terminal = (gameState.isWin() or gameState.isLose() or depth == self.depth)
        is_pacman = (agent == 0)
        if is_terminal:
            return (self.evaluationFunction(gameState), None)
        if is_pacman:
            return self.max_pacman(gameState, depth, agent)
        return self.min_ghost(gameState, depth, agent)

    def max_pacman(self, gameState, depth, agent):
        moves = gameState.getLegalActions()
        options = []

        for move in moves:
            s = gameState.generateSuccessor(agent, move)
            s_depth = depth
            s_index = agent + 1
            pacman_move = (s_index == gameState.getNumAgents())

            if pacman_move:
                s_depth += 1
                s_index = 0

            options.append((self.min_max_search(s, s_depth, s_index)[0], move))
        return max(options, key=lambda entry: entry[0])

    def min_ghost(self, gameState, depth, agent):
        moves = gameState.getLegalActions(agent)
        p=1.0/len(moves)
        exp_value=0

        for move in moves:
            s = gameState.generateSuccessor(agent, move)
            s_depth = depth
            s_index = agent + 1
            pacman_move = (s_index == gameState.getNumAgents())

            if pacman_move:
                s_depth += 1
                s_index = 0

            current=self.min_max_search(s, s_depth, s_index)
            value,action=current
            exp_value+=p*value
        return (exp_value, None)




# question 5
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).




      DESCRIPTION: <write something here so we know what you did>

      Features included:
      current state score,
      total scare time of all ghosts,
      distance between pacman and nearest food,
      distance between pacman and nearest capsule

      I first add the current state score, treating it as the base of the evaluation score.
      Then I add the total scare time of all ghosts.

      Next, I add the reciprocals of distance between pacman position and nearest food and
      capsules respectively. I use reciprocal because for the distance value itself,
      the greater the worse.

      For all my features, I didn't add any weight. In other words, the weight for all of them is
      1 because I think they are equally important.

    """

    # Useful information you can extract from a GameState (pacman.py)

    # first add the score of the current state
    score=0
    score+=currentGameState.getScore()

    # current state's scare time of all ghosts
    ghostStates = currentGameState.getGhostStates()
    time = sum([ghostState.scaredTimer for ghostState in ghostStates])
    score+=time

    # distance between pacman and the nearest food
    pos=currentGameState.getPacmanPosition()
    dist=float("inf")
    for food in currentGameState.getFood().asList():
        dist=min(dist,manhattanDistance(pos,food))
    score+=1.0/dist

    # distance between pacman and the nearest capsule
    dist=float("inf")
    for cap in currentGameState.getCapsules():
        dist=min(dist,manhattanDistance(pos,cap))
    score+=1.0/dist


    return score







# Abbreviation
better = betterEvaluationFunction







# extra credit
class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

