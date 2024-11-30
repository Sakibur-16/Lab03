import util
from util import *

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
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
    Returns a sequence of moves that solves tinyMaze. For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """
    stack = Stack()  # Initialize a stack
    start = problem.getStartState()
    stack.push((start, []))  # Push the start state with an empty path
    visited = set()

    while not stack.isEmpty():
        state, path = stack.pop()
        if state in visited:
            continue
        visited.add(state)

        if problem.isGoalState(state):
            return path

        for successor, action, _ in problem.getSuccessors(state):
            if successor not in visited:
                stack.push((successor, path + [action]))

    return []


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    from util import Queue  # Import Queue
    start = problem.getStartState()  # Start state
    if problem.isGoalState(start):  # Check if start is the goal
        return []

    frontier = Queue()
    frontier.push((start, []))  # Push the start state with an empty path
    explored = set()

    while not frontier.isEmpty():
        state, path = frontier.pop()

        if state in explored:
            continue

        explored.add(state)

        if problem.isGoalState(state):
            return path

        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in explored:
                new_path = path + [action]
                frontier.push((successor, new_path))

    return []


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    pq = PriorityQueue()  # Min-priority queue
    start = problem.getStartState()
    pq.push((start, []), 0)  # Push the start state with cost 0
    visited = set()
    cost_so_far = {start: 0}  # Store the cost to reach each state

    while not pq.isEmpty():
        state, path = pq.pop()
        if state in visited:
            continue
        visited.add(state)

        if problem.isGoalState(state):
            return path

        for successor, action, step_cost in problem.getSuccessors(state):
            new_cost = cost_so_far[state] + step_cost
            if successor not in visited or new_cost < cost_so_far.get(successor, float('inf')):
                cost_so_far[successor] = new_cost
                pq.push((successor, path + [action]), new_cost)

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem. This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    pq = PriorityQueue()  
    start = problem.getStartState()
    pq.push((start, []), 0)  
    visited = set()
    cost_so_far = {start: 0}  

    while not pq.isEmpty():
        state, path = pq.pop()
        if state in visited:
            continue
        visited.add(state)

        if problem.isGoalState(state):
            return path

        for successor, action, step_cost in problem.getSuccessors(state):
            new_cost = cost_so_far[state] + step_cost
            heuristic_cost = new_cost + heuristic(successor, problem)
            if successor not in visited or new_cost < cost_so_far.get(successor, float('inf')):
                cost_so_far[successor] = new_cost
                pq.push((successor, path + [action]), heuristic_cost)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
