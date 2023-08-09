```python
def get_total_distance(old_distance_traveled, distance_to_circle, distance_in_circle):
    """
    Compute the total distance based on the old distance traveled, the distance to the range of the circle,
    and the distance inside the circle.

    Parameters:
    - old_distance_traveled: The distance already traveled before reaching the circle.
    - distance_to_circle: The distance from the current location to the range (edge) of the circle.
    - distance_in_circle: The distance traveled inside the circle.

    Returns:
    - total_distance: The total distance traveled.
    """

    # Compute the total distance
    total_distance = old_distance_traveled + distance_to_circle + distance_in_circle

    return total_distance

# Example usage:
old_distance_traveled = 100  # e.g. 100 km
distance_to_circle = 20  # e.g. 20 km
distance_in_circle = 30  # e.g. 30 km

total_distance = get_total_distance(old_distance_traveled, distance_to_circle, distance_in_circle)
print(f"Total distance: {total_distance} km")

```