import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import product
import networkx as nx
from collections import defaultdict

def draw_circle(ax, center, radius):
    circle = plt.Circle(center, radius, fill=False, edgecolor='b', linewidth=1.5)
    ax.add_patch(circle)

def angle_between_points(p1, p2):
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

def point_on_circle(center, angle, radius):
    return (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))


def compute_distance(point1, point2):
    try:
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    except TypeError:
        print(f"Error with points: {point1} and {point2}")
        raise

def format_list_to_decimal(numbers_list):
    """Format each number in the list to two decimal places."""
    return ["{:.2f}".format(num) for num in numbers_list]


def find_meeting_point_with_survey_final(center,radius, ugv_start, ugv_end, speed_ugv, speed_uav, survey_time):
    """
    Final function to calculate the meeting point of UGV and UAV on the chord considering survey time.
    
    Parameters:
    - radius: Radius of the circle.
    - ugv_start: Starting point of UGV.
    - ugv_end: Ending point of UGV.
    - speed_ugv: Speed of UGV.
    - speed_uav: Speed of UAV.
    - survey_time: Time of survey the point
    
    Returns:
    - Meeting point of UGV and UAV.
    - Wait time for UGV (if any).
    """
    # print(f"In Find Meeting-----------------------{ugv_start,ugv_end}-------------------------------------------")
    # print(f"{ugv_end[0]} - {ugv_start[0]}")
    # Calculating the length of the chord
    chord_length = np.sqrt((ugv_end[0] - ugv_start[0])**2 + (ugv_end[1] - ugv_start[1])**2)
    
    # Distance traveled by UGV when UAV reaches the center and finishes survey
    distance_ugv_traveled = speed_ugv * ((radius / speed_uav) + survey_time)
    
    # Remaining distance on the chord for UGV
    remaining_distance = chord_length - distance_ugv_traveled
    
    # Time taken by UAV and UGV to meet on the chord after UAV finishes survey
    time_to_meet = remaining_distance / (speed_uav + speed_ugv)
    
    # Total distance traveled by UGV on the chord till they meet
    total_distance_ugv = distance_ugv_traveled + speed_ugv * time_to_meet
    
    # Coordinates of the meeting point on the chord
    x_meeting = ugv_start[0] + (total_distance_ugv / chord_length) * (ugv_end[0] - ugv_start[0])
    y_meeting = ugv_start[1] + (total_distance_ugv / chord_length) * (ugv_end[1] - ugv_start[1])
    
    # Check if meeting point is outside the circle
    distance_from_center = np.sqrt((x_meeting - center[0])**2 + (y_meeting - center[1])**2)
    
    wait_time = 0  # Default wait time
    
    if distance_from_center > radius:
        # print("yes",{distance_from_center})
        # Set the meeting point to be the UGV end location
        x_meeting, y_meeting = ugv_end[0], ugv_end[1]
        
        # Calculate distance UGV traveled to get to the end point
        distance_ugv_traveled = chord_length
        
        # Calculate the time taken by UGV to reach the end point
        time_ugv = distance_ugv_traveled / speed_ugv
        
        # Time taken by UAV to reach the UGV end point from the circle's center after survey
        distance_uav_traveled = radius*2
        time_uav = (distance_uav_traveled / speed_uav) + survey_time
        
        # Calculate the wait time for UGV
        wait_time = time_uav - time_ugv
    
    return (x_meeting, y_meeting), wait_time
def compute_ordered_points(plan_output, waypoints):
    ordered_points = [waypoints[int(node)] for node in plan_output.split() if node.isdigit()]
    ordered_points.append(ordered_points[0])
    return ordered_points

def initialize_plot(ordered_points):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(*zip(*ordered_points))
    return fig, ax

def compute_paths_for_radius(ordered_points, radius_combination, speed_ugv, speed_uav, survey_time):
    UGV_outer_path = [ordered_points[0]]
    UGV_path = [ordered_points[0]]
    UAV_path = []
    # UAV_distances = []
    UGV_inter_distances = []
    
    # Main logic to compute paths based on radius_combination
    # This is a placeholder for now. We'll populate this in the next steps.
    
    return UGV_path, UAV_path, UGV_outer_path, UGV_inter_distances

def calculate_UAV_inter_distances_inter(UAV_path):
    UAV_inter_distances = 0
    group_distance=0
    # print(f"what I got in function:{UAV_path}")
    for i in range(len(UAV_path) - 1):
        group_distance += compute_distance(UAV_path[i], UAV_path[i + 1])
    UAV_inter_distances=group_distance
    # print(f"UAV_path:{UAV_path},UAV_inter_distances{UAV_inter_distances}")
    return UAV_inter_distances

def compute_path_for_one_point(ordered_points, point_index, radius, speed_ugv, speed_uav, survey_time):
    if point_index <= 0 or point_index >= len(ordered_points) - 1:
        raise ValueError("Invalid point index. It should be between 1 and len(ordered_points) - 2.")
    
    UGV_outer_path = []
    UGV_path = []
    UAV_path = []
    UGVD_inter_without_drone = []
    UGVD_inter_with_drone = []
    chord_end = None
    prev_chord_end = None
    # final_wait_set = []
    radius_set = []
    final_wait_time = 0 
    point = ordered_points[point_index]
    
    # Assuming angle_between_points and point_on_circle functions are defined previously
    angle = angle_between_points(ordered_points[point_index - 1], ordered_points[point_index])
    nextangle = angle_between_points(ordered_points[point_index], ordered_points[point_index + 1])
    chord_start = point_on_circle(ordered_points[point_index], angle - np.pi, radius)
    chord_end = point_on_circle(ordered_points[point_index], nextangle, radius)
    prev_chord_end = point_on_circle(ordered_points[point_index - 1],angle,radius)
    radius_set.append(radius)
    
    if point_index > 1 and circles_overlap(ordered_points[point_index], radius, ordered_points[point_index - 1], radius):
        chord_start = prev_chord_end
 
    # Assuming find_meeting_point_with_survey_final function is defined previously
    final_meeting_point, final_wait_time = find_meeting_point_with_survey_final(
        ordered_points[point_index], radius, chord_start, chord_end, 
        speed_ugv, speed_uav, survey_time
    )
    final_wait_time = final_wait_time
    
    if chord_start:
        UAV_path_segment = [chord_start, point, final_meeting_point]
        UAV_path.append(UAV_path_segment)
        UGV_path.append(chord_start)

        # Assuming calculate_UAV_inter_distances_inter function is defined previously
        UAV_path_segment_distance = calculate_UAV_inter_distances_inter(UAV_path_segment)
        if point_index == 1:
            UGV_outer_path.append([ordered_points[0], chord_start])
        else:
            UGV_outer_path.append([prev_chord_end, chord_start])
        # print(f"UGV_outer_path:{UGV_outer_path}")
    if chord_end:
        UGV_path.append(chord_end)

    # Assuming compute_distance function is defined previously
    inter_distance_without_drone = compute_distance(chord_start, final_meeting_point)
    inter_distance_with_drone = compute_distance(final_meeting_point, chord_end)
    UGVD_inter_without_drone.append(inter_distance_without_drone)
    UGVD_inter_with_drone.append(inter_distance_with_drone)
    
    return radius_set, UGV_path, UAV_path, UAV_path_segment_distance,UGV_outer_path, UGVD_inter_without_drone, UGVD_inter_with_drone, final_wait_time

# Note: This function assumes the existence of functions like angle_between_points, point_on_circle, 
# find_meeting_point_with_survey_final, etc. which are not provided here. So, this function won't run as-is without those definitions.


def compute_non_overlapping_pairwise_distances(path):
    """Compute the distances between each non-overlapping pair of points in the given path."""
    distances = []
    for sub_path in path:
        for i in range(0, len(sub_path) - 1, 2):
            # Validate the points before computing the distance
            if isinstance(sub_path[i], (list, tuple)) and len(sub_path[i]) == 2 and \
               isinstance(sub_path[i+1], (list, tuple)) and len(sub_path[i+1]) == 2:
                distances.append(compute_distance(sub_path[i], sub_path[i+1]))
                # print(sub_path[i+1]) 
    return distances

    
def compute_tripwise_distances(path):
    """Compute the distances for each set of three points in the given path."""
    distances = []
    #print(path)
    for i in range(0, len(path) - 2, 3):
        
        distance_trip = compute_distance(path[i], path[i+1]) + compute_distance(path[i+1], path[i+2])
        distances.append(distance_trip)
    return distances

def circles_overlap(center1, radius1, center2, radius2):
    """Check if two circles overlap."""
    distance_between_centers = compute_distance(center1, center2)
    return distance_between_centers < (radius1 + radius2)
    
def remove_empty_lists(lst):
    return [sublist for sublist in lst if sublist]

def draw_circle(ax, center, radius, *args, **kwargs):
    circle = plt.Circle(center, radius, fill=False, *args, **kwargs)
    ax.add_patch(circle)

def calculate_UAV_distances(UAV_path, compute_distance):
    UAV_distances_set = []
    for group in UAV_path:
        distance = 0
        for i in range(len(group) - 1):
            distance += compute_distance(group[i], group[i+1])
        UAV_distances_set.append(distance)
    return UAV_distances_set


def calculate_energy_remaining(Total_E_UGV, Total_E_UAV, UGV_outer_path_distances, UGVD_inter_with_drone, UAV_distances_set, UAV_path, UAV_E_cost, UAV_E_s_cost, UGVD_inter_without_drone, UGV_E_cost_without_UAV, Charging_speed):
    UGV_energy_remaining = Total_E_UGV
    UAV_energy_remaining = Total_E_UAV

    for i in range(len(UGV_outer_path_distances)-1):
        charging_distance = UGVD_inter_with_drone[i] + UGV_outer_path_distances[i+1]
        UAV_distances = UAV_distances_set[i]
        
        UAV_Trip_Cost = UAV_distances * UAV_E_cost + UAV_E_s_cost * (len(UAV_path[i])-2)
        UGV_Trip_Cost = UGVD_inter_without_drone[i] * UGV_E_cost_without_UAV
        
        UAV_energy_remaining -= UAV_Trip_Cost 
        UGV_energy_remaining -= UGV_Trip_Cost
        if UGV_energy_remaining < 0 or UAV_energy_remaining < 0:
            return -1, -1 
            
        if UAV_energy_remaining < Total_E_UAV:
            charge_amount = min(Charging_speed * charging_distance , UAV_Trip_Cost, UGV_energy_remaining)
            UGV_energy_remaining -= charge_amount
            UAV_energy_remaining += charge_amount
        
        if UGV_energy_remaining < 0 or UAV_energy_remaining < 0:
            return -1, -1  # Indicating mission failed for both

    return UGV_energy_remaining, UAV_energy_remaining


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
    
    # Dynamic figure size
    num_nodes = len(all_nodes)
    num_levels = len(level_counts)
    fig_width = max(10, num_nodes * 0.5)
    fig_height = max(5, num_levels * 2)
    plt.figure(figsize=(fig_width, fig_height))

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=500)

    # Edges
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, width=2)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

    # Node data annotations
    for node in all_nodes:
        plt.annotate(
            f'T:{node.total_time:.2f}',
            xy=pos[node.id], textcoords='offset points', xytext=(-20,-30))

    plt.show()
    
def find_min_total_time_node_at_level(root, target_level):
    """
    Find the node with minimum node.total_time among all nodes at a specific node.level 
    and retrieve the path from that node back to the root.
    
    Parameters:
    - root: The root node of the tree.
    - target_level: The level at which we want to find the node with minimum total_time.
    
    Returns:
    - result: A dictionary containing the node's id, total_time, and the path from that node back to the root.
    """
    nodes_at_target_level = [node for node in root.get_nodes() if node.level == target_level]
    
    if not nodes_at_target_level:
        return None  # No nodes found at the target level
    
    # Find the node with the minimum total_time at the target level
    min_total_time_node = min(nodes_at_target_level, key=lambda node: node.total_time)
    
    # Retrieve the path from that node back to the root
    path = []
    current_node = min_total_time_node
    while current_node is not None:
        path.append(current_node.radius)
        current_node = current_node.parent
    path.reverse()  # Reverse the path to start from the root
    
    # Construct the result dictionary
    result = {
        "id": min_total_time_node.id,
        "value":min_total_time_node.value,
        "total_time": min_total_time_node.total_time,
        "radius comb": path
    }
    
    return result

def compute_optimized_paths_for_radius(ordered_points, radius_combination, speed_ugv, speed_uav, survey_time):
    UGV_outer_path = []
    UGV_path = [ordered_points[0]]
    UAV_path = []
    UGVD_inter_without_drone = []
    UGVD_inter_with_drone = []
    chord_end = None
    prev_chord_end = None
    final_wait_set = []
    radius_set = []
    for i, point in enumerate(ordered_points[:-1]):
        # radius_combination
        
        # print(radius_combination,i % len(radius_combination) )
        # print(i, point)
        
       
        # Exclude the start and end points from circles
        if 0 < i < len(ordered_points) - 2:
            current_radius = radius_combination[i-1]
            # print(i)
            angle = angle_between_points(ordered_points[i-1], ordered_points[i])
            nextangle = angle_between_points(ordered_points[i], ordered_points[i+1])
            chord_start = point_on_circle(ordered_points[i], angle - np.pi, current_radius)
            chord_end = point_on_circle(ordered_points[i], nextangle, current_radius)
            radius_set.append(current_radius)
            if prev_chord_end and i > 1 and circles_overlap(ordered_points[i], current_radius, ordered_points[i-1], radius_combination[(i-1) % len(radius_combination)]):
                chord_start = prev_chord_end

            final_meeting_point, final_wait_time = find_meeting_point_with_survey_final(
                ordered_points[i], current_radius, chord_start, chord_end, 
                speed_ugv, speed_uav, survey_time
            )
            final_wait_set.append(final_wait_time)
            # Update paths based on the provided structure
            if chord_start:
                UAV_path_segment = [chord_start, point, final_meeting_point]
                UAV_path.append(UAV_path_segment)
                UGV_path.append(chord_start)
                UAV_path_segment_distance = calculate_UAV_inter_distances_inter(UAV_path_segment)
                # Update UGV_outer_path as per the new structure
                if i == 1:
                    UGV_outer_path.append([ordered_points[0], chord_start])
                else:
                    UGV_outer_path.append([prev_chord_end, chord_start])
                
                    
                
            if chord_end:
                UGV_path.append(chord_end)
            inter_distance_without_drone = compute_distance(chord_start, final_meeting_point)
            inter_distance_with_drone = compute_distance(final_meeting_point, chord_end)
            UGVD_inter_without_drone.append(inter_distance_without_drone)
            UGVD_inter_with_drone.append(inter_distance_with_drone)
            prev_chord_end = chord_end

            
            # print(UAV_path_segment_distance)
            # print ( UGV_outer_path_distances, inter_distance_with_drone, UAV_distances_set, UAV_path_segment, UAV_E_cost, UAV_E_s_cost, UGVD_inter_without_drone, UGV_E_cost_without_UAV, Charging_speed)
            # UGV_energy_remaining, UAV_energy_remaining = calculate_energy_remaining_inter(Total_E_UGV, Total_E_UAV, UGV_outer_path_distances, UGVD_inter_with_drone, UAV_distances_set, UAV_path, UAV_E_cost, UAV_E_s_cost, UGVD_inter_without_drone, UGV_E_cost_without_UAV, Charging_speed)

    # Adding the last point to the paths
    UGV_path.append(ordered_points[-1])
    UGV_outer_path.append([chord_end,ordered_points[-1]])
    
    return radius_set ,UGV_path, UAV_path, UGV_outer_path, UGVD_inter_without_drone, UGVD_inter_with_drone,final_wait_set

from matplotlib.patches import FancyArrowPatch
def plot_updated_paths_with_circles(UAV_path, UGV_outer_path, ordered_points, result):
    fig, ax = plt.subplots(figsize=(10, 10))
    time= result[1]
    radius_combination = result[0]
    # # Plot UAV_path
    # for segment in UAV_path:
    #     # print(segment[0],segment[-1])
    #     ax.plot(*zip(segment[0],segment[-1]),'r--',label='UGV_alone')
        
    #     ax.plot(*zip(*segment), 'g--', label='UAV Path')
    #     # ax.scatter(*zip(*segment), color='g', marker='o')
    
    # # Plot UGV_outer_path without connecting segments to each other
    # for segment in UGV_outer_path:
    #     ax.plot(*zip(*segment), 'b-', label='UGV Outer Path')

        # ax.scatter(segment[0][0], segment[0][1], color='b', marker='x')  # Plotting start of the segment
        # ax.scatter(segment[1][0], segment[1][1], color='b', marker='x')  # Plotting end of the segment
    for index, segment in enumerate(UAV_path):
    # Plot for UAV_path
        ax.plot(*zip(segment[0], segment[-1]), 'r--', label='UGV_alone')
        ax.plot(*zip(*segment), 'g--', label='UAV Path')
        ax.scatter(*segment[-1], s=20, marker='o', facecolors='none', edgecolors='r', label='meeting_point')  # Plotting end of the segment
        ax.plot(*zip(segment[-1],UGV_outer_path[index+1][0]), 'm-', label='UGV_with_UGV')
        # Plot for UGV_outer_path
        # ax.plot(*zip(*UGV_outer_path[index]), 'b-', label='UGV Outer Path')
        line = ax.plot(*zip(*UGV_outer_path[index]), 'b-', label='UGV Outer Path')
        start_point = UGV_outer_path[index][0]
        end_point = UGV_outer_path[index][-1]
        arrow = FancyArrowPatch(posA=start_point, posB=end_point, mutation_scale=15, arrowstyle="-|>", color="b")
        ax.add_patch(arrow)
    # Since UGV_outer_path has one more segment, plot the last segment of UGV_outer_path separately
    ax.plot(*zip(*UGV_outer_path[-1]), 'b-', label='UGV Outer Path')
    # Draw circles
    for i, point in enumerate(ordered_points[:-1]):
        
        if 0 < i < len(ordered_points) - 2:
            
            current_radius = radius_combination[i-1]
            ax.annotate(f"R = {current_radius:.1f}", (point[0] + 0.2, point[1] + 0.2))
            draw_circle(ax, point, current_radius, color='b', linestyle='-')
        
    ax.scatter(*ordered_points[0], color='b', marker='x',label='Start')  # Plotting start of the segment 
    # Labels and Title
    ax.set_title(f"UAV and UGV Optomized Path, Total_T = {result[1]} h ")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    
    # Handling legends to avoid repetition
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))
    unique_handles = [handles[labels.index(lab)] for lab in unique_labels]
    ax.legend(unique_handles, unique_labels, loc="best")
    ax.set_aspect('equal', adjustable='box')
    plt.show()
