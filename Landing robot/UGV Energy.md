This should base one [[Charging and discharge]]\, Waitting time does not cost energy.


```python
def get_ugv_energy(total_charge, travel_with_drone_mah, travel_without_drone_mah, drone_charge_mah):
    """
    Compute the remaining UGV energy based on the total charge and mAh used.

    Parameters:
    - total_charge: The total available charge in mAh.
    - travel_with_drone_mah: The mAh used for traveling with the drone.
    - travel_without_drone_mah: The mAh used for traveling without the drone.
    - drone_charge_mah: The mAh used to charge the drone.

    Returns:
    - remaining_energy: The remaining UGV energy in mAh.
    """

    used_charge = travel_with_drone_mah + travel_without_drone_mah + drone_charge_mah
    remaining_energy = total_charge - used_charge

    # Ensure that the remaining energy is not negative
    if remaining_energy < 0:
        print("Error: Used more energy than available!")
        return 0

    return remaining_energy

# Example usage:
total_charge = 10000  # 10000 mAh
travel_with_drone_mah = 3000
travel_without_drone_mah = 2000
drone_charge_mah = 2500

remaining_energy = get_ugv_energy(total_charge, travel_with_drone_mah, travel_without_drone_mah, drone_charge_mah)
print(f"Remaining UGV energy: {remaining_energy} mAh")


```