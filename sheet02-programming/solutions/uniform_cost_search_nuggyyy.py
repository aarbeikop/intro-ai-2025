import search_problem
import queue
from search import Search, SearchNode


class UniformCostSearch(Search):
  name = "uniform-cost"

  def search(self):
    # Initialize the open list as a priority queue
    open_list = queue.PriorityQueue()
    
    # Create the initial node with the initial state, no parent, and cost 0
    initial_node = SearchNode(self.search_problem.initial_state, None, 0)
    
    # Insert the initial node into the open list, prioritized by g-value (path cost)
    open_list.put((initial_node.g, id(initial_node), initial_node))
    
    # Initialize the closed set to keep track of visited states
    closed = set()
    
    # Count the initial node as generated
    self.generated += 1
    
    while not open_list.empty():
      # Pop the node with the lowest cost from the open list
      current_cost, _, current_node = open_list.get()
      
      # If we've reached a goal state, return the path and its cost
      if self.search_problem.is_goal(current_node.state):
        path = self.extract_path(current_node)
        return path, current_node.g
      
      # Skip if the state has already been visited (already in closed set)
      if current_node.state not in closed:
        # Add the current state to the closed set
        closed.add(current_node.state)
        # Count the node as expanded
        self.expanded += 1
        
        # Generate all successor states
        for action in self.search_problem.actions(current_node.state):
          # Get the successor state and the cost of the action
          successor_state, action_cost = self.search_problem.result(current_node.state, action)
          
          # Calculate the new path cost to the successor
          new_cost = current_node.g + action_cost
          
          # Create a new node for the successor state with the updated cost
          successor_node = SearchNode(successor_state, current_node, new_cost)
          
          # Insert the successor node into the open list, prioritized by its g-value
          open_list.put((successor_node.g, id(successor_node), successor_node))
          
          # Count the successor node as generated
          self.generated += 1
    
    # If the open list is empty and no solution has been found, return None
    return None, 0


if __name__ == "__main__":
  problem = search_problem.generate_random_problem(8, 2, 3, max_cost=10)
  problem.dump()
  ucs = UniformCostSearch(problem, True)
  ucs.run()


