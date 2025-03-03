import search_problem
from search import Search, SearchNode
from queue import PriorityQueue

class UniformCostSearch(Search):
    name = "uniform-cost"

    def search(self):
        p = self.search_problem
        if p.is_goal(p.initial_state):
            return [p.initial_state], 0
        
        # Use PriorityQueue for uniform-cost search
        # Add a counter to break ties and avoid comparing SearchNode objects
        counter = 0
        frontier = PriorityQueue()
        frontier.put((0, counter, SearchNode(p.initial_state, None, 0)))
        self.generated += 1
        
        # Track reached states and their costs
        reached = {p.initial_state: 0}

        while not frontier.empty():
            # Get node with lowest cost
            cost, _, node = frontier.get()
            state = node.state
            
            # Check if we've reached the goal
            if p.is_goal(state):
                return self.extract_path(node), cost
            
            self.expanded += 1
            
            # Generate successors using the correct API
            for action in p.actions(state):
                next_state, action_cost = p.result(state, action)
                new_cost = cost + action_cost
                
                # If state not reached before or we found a better path
                if next_state not in reached or new_cost < reached[next_state]:
                    reached[next_state] = new_cost
                    counter += 1
                    frontier.put((new_cost, counter, SearchNode(next_state, node, new_cost)))
                    self.generated += 1
        
        # No solution found
        return None, None


if __name__ == "__main__":
    problem = search_problem.generate_random_problem(8, 2, 3, max_cost=10)
    problem.dump()
    ucs = UniformCostSearch(problem, True)
    ucs.run()