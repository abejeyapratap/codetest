This should base one [[Charging and discharge]]
```python
def get_uav_energy(total_charge, travel_mah, survey_mah, charge_gained_mah):
    """
    Compute the remaining UAV energy based on the total charge, mAh used for traveling, 
    mAh used for the site survey, and mAh gained from charging.

    Parameters:
    - total_charge: The total available charge in mAh.
    - travel_mah: The mAh used for traveling.
    - survey_mah: The mAh used for the site survey.
    - charge_gained_mah: The mAh gained from charging.

    Returns:
    - remaining_energy: The remaining UAV energy in mAh.
    """

    # Compute the total used charge
    total_used_charge = travel_mah + survey_mah - charge_gained_mah

    # Compute the remaining energy
    remaining_energy = total_charge - total_used_charge

    # Ensure that the remaining energy is not negative
    if remaining_energy < 0:
        print("Error: Used more energy than available!")
        return 0

    return remaining_energy

# Example usage:
total_charge = 5000  # 5000 mAh
travel_mah = 3000
survey_mah = 1000
charge_gained_mah = 2000

remaining_energy = get_uav_energy(total_charge, travel_mah, survey_mah, charge_gained_mah)
print(f"Remaining UAV energy: {remaining_energy} mAh")

```