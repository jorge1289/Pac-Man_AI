# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    stack = util.Stack()
    start = (problem.getStartState(), [], 0)
    stack.push(start)
    visited = []

    while not stack.isEmpty():
        current = stack.pop()
        if not problem.isGoalState(current[0]):
            if current[0] not in visited:
                visited.append(current[0])
                children = problem.getSuccessors(current[0])

                for child in children:
                    position, action, cost = child
                    new_action = current[1] + [action]
                    new_child = (position, new_action, cost)
                    stack.push(new_child)

        if problem.isGoalState(current[0]):
            print(current[1])
            return current[1]
        

    

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    visited = []
    start = (problem.getStartState(), [], 0)
    queue.push(start)

    while not queue.isEmpty():
        current = queue.pop()
        if not problem.isGoalState(current[0]):
            if current[0] not in visited:
                visited.append(current[0])
                children = problem.getSuccessors(current[0])

                for child in children:
                    position, action, cost = child
                    new_action = current[1] + [action]
                    new_child = (position, new_action, cost)
                    queue.push(new_child)

        if problem.isGoalState(current[0]):
            return current[1]

    #util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #til.raiseNotDefined()
    PQ = util.PriorityQueue()
    visited = []
    start = (problem.getStartState(), [], 0)
    PQ.push(start, start[2])

    while not PQ.isEmpty():
        current = PQ.pop()
        if not problem.isGoalState(current[0]):
            if current[0] not in visited:
                visited.append(current[0])
                children = problem.getSuccessors(current[0])

                for child in children:
                    position, action, cost = child
                    new_action = current[1] + [action]
                    new_cost = current[2] + cost
                    new_child = (position, new_action, new_cost)
                    PQ.update(new_child, new_child[2])

        if problem.isGoalState(current[0]):
            print(current[1])
            return current[1]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    PQ = util.PriorityQueue()
    visited = []
    start_cost = heuristic(problem.getStartState(), problem)
    start = (problem.getStartState(), [], start_cost)
    PQ.push(start, start[2])

    while not PQ.isEmpty():
        current = PQ.pop()
        if not problem.isGoalState(current[0]):
            if current[0] not in visited:
                visited.append(current[0])
                children = problem.getSuccessors(current[0])

                for child in children:
                    position, action, cost = child
                    new_action = current[1] + [action]
                    new_cost = current[2] + cost
                    new_child = (position, new_action, new_cost)
                    PQ.push(new_child, new_child[2] + heuristic(position, problem))

        if problem.isGoalState(current[0]):
            return current[1]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
