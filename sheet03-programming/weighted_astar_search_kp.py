import pancake_problem

from pancake_problem import PancakeProblem
from queue import PriorityQueue
from search import Search, SearchNode, WeightedAStarSearchNode


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
    p = self.search_problem
    if p.is_goal(p.initial_state):
      return [p.initial_state], 0

    frontier = PriorityQueue()

    init_f = self.w * p.h(p.initial_state)

    frontier.put((init_f, WeightedAStarSearchNode(p.initial_state, None, 0)))
    self.generated += 1
    reached = {p.initial_state}


    #nodes_g = {frontier[0] : 0}
    #nodes_f = {frontier[0] : 0}

    while frontier:
      priority, node = frontier.get()
      self.expanded += 1

      # mark reached to avoid cycles
      reached.add(node.state)

      for action in p.actions(node.state):
        succ, cost = p.result(node.state, action)
        if succ in reached:
          continue


        h_score = p.h(succ)
        f_score = node.g + cost + self.w * h_score

        new_g = node.g + cost

        succ_node = WeightedAStarSearchNode(succ, node, new_g)

        # early goal test
        if p.is_goal(succ):
          return self.extract_path(succ_node), new_g

        # enqueue successor
        frontier.put((f_score, succ_node))
        self.generated += 1

        if self.generated == self.max_generations:
          print("Aborting search after generating " +
            f"{self.max_generations} states without finding a solution.")
          return None, None

    # no solution found
    print("Explored entire search problem, no solution exists.")
    return None, None



if __name__ == "__main__":
  problem = pancake_problem.generate_random_problem(5)
  problem = PancakeProblem((1, 5, 6, 2, 4, 3))
  problem.dump()
  astar = WeightedAStarSearch(problem, 1, print_statistics=True)
  astar.run()



