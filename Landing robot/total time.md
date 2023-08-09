UGV only
Travel time + [[waiting time]]

**Using Iterative Methods Instead of Recursive:** Recursive methods can increase the complexity in terms of space and time. Where possible, use iterative methods.

```python
def calculate_total_ugv_time(accumulated_time, travel_time, wait_time):
    """
    Function to calculate the total time for a UGV.

    Parameters:
    accumulated_time (float): The accumulated time for the UGV.
    travel_time (float): The travel time for the UGV.
    wait_time (float): The wait time for the UGV.

    Returns:
    float: The total time for the UGV.
    """
    # Calculate the total time as the sum of the accumulated, travel, and wait times
    total_time = accumulated_time + travel_time + wait_time

    return total_time

# Test the function
accumulated_time = 10  # Accumulated time for UGV
travel_time = 5  # Travel time for UGV
wait_time = 2  # Wait time for UGV

print(calculate_total_ugv_time(accumulated_time, travel_time, wait_time))

```