import pancake_problem

from pancake_problem import PancakeProblem
from queue import PriorityQueue
from search import Search, SearchNode

class WeightedAStarSearchNode(SearchNode):
  def __lt__(self, other):
    return self.g < other.g

class WeightedAStarSearch(Search):
  name = "weighted-astar"

  def __init__(self, search_problem, weight, **kwargs):
    super().__init__(search_problem, **kwargs)
    self.w = weight
    if weight == 0:
      self.name = "uniform-cost"
    elif weight == 1:
      self.name = "astar"

  def search(self):
    # early goal test for initial state
    p = self.search_problem
    if p.is_goal(p.initial_state):
      return [p.initial_state], 0

    # Initialize frontier as a priority queue
    frontier = PriorityQueue()
    initial_node = WeightedAStarSearchNode(p.initial_state, None, 0)
    # f(n) = g(n) + w*h(n)
    f_value = initial_node.g + self.w * p.h(p.initial_state)
    # Use counter for tiebreaking in priority queue
    frontier.put((f_value, initial_node))
    self.generated += 1
    
    # Keep track of reached states and their best known costs
    reached = {p.initial_state: 0}

    while not frontier.empty():
      if self.time_limit_reached():
        print("Time limit reached, terminating search.")
        return None, float('inf')
      
      # Get node with lowest f-value
      _, node = frontier.get()
      self.expanded += 1

      # Generate and evaluate all successors
      for action in p.actions(node.state):
        succ, cost = p.result(node.state, action)
        new_g = node.g + cost
        
        # Skip if we've seen this state with a better g-value
        if succ in reached and reached[succ] <= new_g:
          continue
          
        # Create successor node
        succ_node = WeightedAStarSearchNode(succ, node, new_g)
        
        # Early goal test
        if p.is_goal(succ):
          return self.extract_path(succ_node), new_g
        
        # Update reached states
        reached[succ] = new_g
        
        # Add to frontier with f-value priority
        f_value = new_g + self.w * p.h(succ)
        frontier.put((f_value, succ_node))
        self.generated += 1
        
        if self.generated >= self.max_generations:
          print(f"Aborting search after generating {self.max_generations} states without finding a solution.")
          return None, float('inf')
    
    # No solution found
    print("Explored entire search space, no solution exists.")
    return None, float('inf')

if __name__ == "__main__":
  problem = pancake_problem.generate_random_problem(5)
  problem = PancakeProblem((1, 5, 6, 2, 4, 3))
  problem.dump()
  astar = WeightedAStarSearch(problem, 1, print_statistics=True)
  astar.run()



