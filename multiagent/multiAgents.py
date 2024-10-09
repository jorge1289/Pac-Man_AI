# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        # consider food locations and and ghost locations also the scared times
        '''food = newFood.asList()

        distance_to_food = 10000
        for fod in food:
            dist = manhattanDistance(fod, newPos)
            distance_to_food = min(distance_to_food, dist)
            #print(dist)
        distance_to_ghost = 10000
        for ghost in newGhostStates:
            dist_ghost = manhattanDistance(ghost.getPosition(),newPos)
            distance_to_ghost = min(distance_to_ghost, dist_ghost)
            print(dist_ghost)
        "*** YOUR CODE HERE ***"
        return successorGameState.getScore() - distance_to_food + distance_to_ghost'''

        nearest = []
        Food_position = newFood.asList()
        for is_food in Food_position:
            nearest.append(manhattanDistance(newPos, is_food))

        if nearest:
            minFood = min(nearest)
            return successorGameState.getScore() + 1/(1 + minFood)

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        val = self.value(self.index, gameState, self.depth)
        # print(val)
        return val[1]

    def value(self, index, gameState: GameState, depth):
        numberOfAgent = gameState.getNumAgents()

        if depth == 0 or gameState.isWin() or gameState.isLose():
            score = self.evaluationFunction(gameState)
            return (score, "Stop")

        elif index == 0:
           return self.Pacman_value(index, gameState, depth)

        else:
            return self.Ghost_value(index, gameState, depth)



    def Pacman_value(self, index, gameState: GameState, depth):
        maximum = float("-inf")
        Action = "Stop"

        legalMovesPacman = gameState.getLegalActions(index)

        for action in legalMovesPacman:
            successorsGameState = gameState.generateSuccessor(index, action)
            val = self.value(index + 1, successorsGameState, depth)
            if maximum < val[0]:
                maximum = val[0]
                Action = action
        # index += 1
        return (maximum, Action)

    def Ghost_value(self, index, gameState: GameState, depth):
        minium = float("inf")
        Action = "Stop"
        legalMovesPacman = gameState.getLegalActions(index)

        for action in legalMovesPacman:
            successorsGameState = gameState.generateSuccessor(index, action)
            if index ==  gameState.getNumAgents() - 1:
                val = self.value(0, successorsGameState, depth-1)
            else:
                # index += 1
                val = self.value(index + 1, successorsGameState, depth)
            if minium > val[0]:
                minium = val[0]
                Action = action

        # index += 1
        return (minium, Action)

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        res = self.value(self.index, gameState, self.depth, float("-inf"), float("inf"))

        return res[1]
        #util.raiseNotDefined()

    def value(self, index, gameState, depth, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            score = self.evaluationFunction(gameState)
            return (score, "Stop")

        elif index == 0:
            return self.Pacman_value(index, gameState, depth, alpha, beta)

        else:
            return self.Ghost_value(index, gameState, depth, alpha, beta)

    def Pacman_value(self, index, gameState, depth, alpha, beta):
        maximum = float("-inf")
        Action = "Stop"

        legalMovesPacman = gameState.getLegalActions(index)

        for action in legalMovesPacman:
            successorGameState = gameState.generateSuccessor(index, action)
            val = self.value(index + 1, successorGameState, depth, alpha, beta)[0]
            if val > maximum:
                maximum = val
                Action = action
            if maximum > beta:
                return (maximum, Action)
            if alpha < maximum:
                alpha = maximum
        return (maximum, Action)

    def Ghost_value(self, index, gameState, depth, alpha, beta):
        minimum = float("inf")
        Action = "Stop"

        legalMovesPacman = gameState.getLegalActions(index)

        for action in legalMovesPacman:
            successorGameState = gameState.generateSuccessor(index, action)
            if index == gameState.getNumAgents() - 1:
                val = self.value(0, successorGameState, depth - 1, alpha, beta)[0]
            else:
                val = self.value(index + 1, successorGameState, depth, alpha, beta)[0]

            if val < minimum:
                minimum = val
                Action = action
            if minimum < alpha:
                return (minimum, Action)
            if beta > minimum:
                beta = minimum
        return (minimum, Action)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        val = self.value(self.index, gameState, self.depth)
        # print(val)
        return val[1]

    def value(self, index, gameState: GameState, depth):
        numberOfAgent = gameState.getNumAgents()

        if depth == 0 or gameState.isWin() or gameState.isLose():
            score = self.evaluationFunction(gameState)
            return (score, "Stop")

        elif index == 0:
            return self.Pacman_value(index, gameState, depth)

        else:
            return self.Ghost_value(index, gameState, depth)

    def Pacman_value(self, index, gameState: GameState, depth):
        maximum = float("-inf")
        Action = "Stop"

        legalMovesPacman = gameState.getLegalActions(index)

        for action in legalMovesPacman:
            successorsGameState = gameState.generateSuccessor(index, action)
            val = self.value(index + 1, successorsGameState, depth)
            if maximum < val[0]:
                maximum = val[0]
                Action = action
        return (maximum, Action)

    def Ghost_value(self, index, gameState: GameState, depth):
        average = 0
        Action = "Stop"
        legalMovesPacman = gameState.getLegalActions(index)

        for action in legalMovesPacman:
            successorsGameState = gameState.generateSuccessor(index, action)
            if index == gameState.getNumAgents() - 1:
                val = self.value(0, successorsGameState, depth - 1)
            else:
                # index += 1
                val = self.value(index + 1, successorsGameState, depth)
            average += val[0]
            # Action = "Stop"
        minium = average/len(legalMovesPacman)
        return (minium, Action)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    '''newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    nearest = []
    Food_position = newFood.asList()
    for is_food in Food_position:
        nearest.append(manhattanDistance(newPos, is_food))
    
    nearest_ghost = []
    for ghost in newGhostStates:
        nearest_ghost.append(manhattanDistance(ghost.getPosition(),newPos))
        

    return currentGameState.getScore()'''

    Pacman_pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghost_states = currentGameState.getGhostStates()
    scared_times = [ghostState.scaredTimer for ghostState in ghost_states]
    scared_score =0
    nearest = []
    Food_position = food.asList()
    for is_food in Food_position:
        nearest.append(manhattanDistance(Pacman_pos, is_food))

    for ghost_state in ghost_states:
        ghost_position = ghost_state.getPosition()
        if manhattanDistance(ghost_position, Pacman_pos) == 0:
            return float("-inf")
        if ghost_state.scaredTimer > 1:
            scared_score += 1

    if nearest:
        minFood = min(nearest)
        score = currentGameState.getScore() + 1/(1+minFood) - scared_score
        return score

    return currentGameState.getScore() - scared_score

# Abbreviation
better = betterEvaluationFunction
