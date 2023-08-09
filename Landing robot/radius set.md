The radius set is derived from variable energy percentage levels that a drone can employ during a single flight. For instance, with a set like 10, 50, 100, the drone can opt to expend 10%, 50%, or 100% of its energy. This directly impacts the branching factor of each node in a tree structure, with each distinct energy level potentially representing a unique child node. These values directly influence the [[drone's round-trip range]], thereby defining its operational radius.

the expend tree function can be as follow:
```python
def expand_tree(parent_node, radius_set, id_start=1):
    """
    Function to expand a tree based on a given radius set. Each element in the radius set represents a child node.

    Parameters:
    parent_node (Node): The parent node to which child nodes will be attached.
    radius_set (list): The set of radius values, each representing a different child node.
    id_start (int): The starting value for child node IDs.

    Returns:
    int: The next ID value to be used for node creation.
    """
    
    for radius in radius_set:
        # Calculate the values for the new node
        total_distance = parent_node.total_distance + radius  # Example calculation
        ugv_energy = parent_node.ugv_energy - radius  # Example calculation
        uav_energy = parent_node.uav_energy - radius  # Example calculation
        distance_to_next = radius  # Example calculation

        # Ensure the energy levels do not drop below 0
        if ugv_energy >= 0 and uav_energy >= 0:
            # Create the new node and add it as a child of the parent node
            child_node = Node(id_start, total_distance, ugv_energy, uav_energy, distance_to_next)
            parent_node.add_child(child_node)

            # Increment the ID for the next node
            id_start += 1

    # Return the next ID to be used
    return id_start

# Example usage:
root = Node(0, 0, 100, 100, 0)  # Create a root node
radius_set = [10, 50, 100]  # Define the radius set
expand_tree(root, radius_set)  # Expand the tree based on the radius set

# Display the tree
root.traverse()

```