


I want a tree in python, each node has [[total distance traveled]], [[UGV Energy]], [[UAV Energy]], [[Distant to next]]

Each nodes should have N number of child, N is defined by size of the [[radius set]].


The depth of the tree is defined by how many Interest point need to be visit, Note as D. 

 Add Radius which is defined by the  [[radius set]], Add [[total time]] [done]

All possible rotates, find min total time 

MCTS 
## Edge Case Add close points[working]

Cut 0 energy, Cut high total time

Under the TSP tour, lowest total time means best overall. 

**Dijkstra's Algorithm** For traverse the path after the minimal total time is found. 

## full tree first then add data with depth first search (bf)

## MCTS  find in real time (UCB) 


## Here is the tree:
``` python
class Node:
    def __init__(self, id, total_distance, ugv_energy, uav_energy, distance_to_next, radius=None, total_time=None, level=0):
        self.id = id
        self.total_distance = total_distance
        self.ugv_energy = ugv_energy
        self.uav_energy = uav_energy
        self.distance_to_next = distance_to_next
        self.radius = radius
        self.total_time = total_time  # Total time taken to reach this node
        self.level = level
        self.children = []

    def add_child(self, node):
        node.level = self.level + 1  # Set child's level based on parent's level
        self.children.append(node)
        
    def delete_child(self, node):
        if node in self.children:
            self.children.remove(node)

    def display(self):
        print(f"Node ID: {self.id}, Total Distance: {self.total_distance}, UGV Energy: {self.ugv_energy}, UAV Energy: {self.uav_energy}, Distance to Next: {self.distance_to_next}, Radius: {self.radius}, Total Time: {self.total_time}")

    def traverse(self):
        self.display()
        for child in self.children:
            child.traverse()

    def get_nodes(self):
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_nodes())
        return nodes

```

## How to draw the tree (One way):
![[Pasted image 20230712173041.png]]
```python
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def draw_tree(root):
    def add_edges(graph, node):
        for child in node.children:
            graph.add_edge(node.id, child.id)
            add_edges(graph, child)

    G = nx.DiGraph()
    add_edges(G, root)

    # Get all nodes and sort by level
    all_nodes = sorted(root.get_nodes(), key=lambda node: node.level)

    # Count nodes per level
    level_counts = defaultdict(int)

    # Assign positions
    pos = {}
    for node in all_nodes:
        pos[node.id] = (level_counts[node.level], -node.level)
        level_counts[node.level] += 1

    # Adjust x-positions to center nodes on each level
    for node in all_nodes:
        x_adjust = -0.5 * (level_counts[node.level] - 1)
        pos[node.id] = (pos[node.id][0] + x_adjust, pos[node.id][1])

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=500)

    # Edges
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, width=2)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

    # Node data annotations
    for node in all_nodes:
        plt.annotate(
            f'D:{node.total_distance}, U:{node.ugv_energy}, V:{node.uav_energy}, N:{node.distance_to_next}',
            xy=pos[node.id], textcoords='offset points', xytext=(-50,-10))
            
    plt.show()
```
## (Another way) in pdf:![[Screenshot from 2023-07-12 17-32-14.png]]
```python
from graphviz import Digraph

def draw_tree(root):
    dot = Digraph()
    def add_edges(node):
        for child in node.children:
            dot.edge(str(node.id), str(child.id), 
                     label=f'D:{child.total_distance}, U:{child.ugv_energy}, V:{child.uav_energy}, N:{child.distance_to_next}')
            add_edges(child)
    add_edges(root)
    dot.view()
```
## Test Data:
```python
root = Node(0, 0, 100, 100, 10)
child1 = Node(1, 10, 90, 90, 20)
child2 = Node(2, 30, 80, 80, 15)
child3 = Node(3, 45, 70, 70, 25)
child4 = Node(4, 40, 75, 75, 30)
child5 = Node(5, 30, 80, 80, 50)
# child6 = Node(6, 30, 80, 80, 50)
root.add_child(child1)
root.add_child(child2)
child1.add_child(child3)
child1.add_child(child4)
child2.add_child(child5)
# child2.add_child(child6)
# Draw the tree
draw_tree(root)
```
