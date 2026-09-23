import numpy as np
import pandas as pd

def calculate_full_trajectory(v0, theta_degrees, g=9.81, num_steps=200):
    """
    Calculates the entire trajectory for a given launch.
    """
    if v0 <= 0 or theta_degrees <= 5 or theta_degrees >= 85:
        return None, None
        
    theta_rad = np.deg2rad(theta_degrees)
    t_flight = (2 * v0 * np.sin(theta_rad)) / g
    
    # Ensure we have enough steps for a smooth curve
    if t_flight < 1:
        num_steps = 50
    
    t = np.linspace(0, t_flight, num_steps)
    x = v0 * np.cos(theta_rad) * t
    y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
    
    return x, y

# --- Data Generation Settings ---
NUM_TRAJECTORIES = 25000  # Daha az yörünge ama...
SAMPLES_PER_TRAJECTORY = 4 # ...her yörüngeden daha fazla nokta 
TOTAL_SAMPLES = NUM_TRAJECTORIES * SAMPLES_PER_TRAJECTORY

V0_RANGE = (20, 200)       # m/s
THETA_RANGE = (10, 80)     # degrees

print(f"Generating a smarter dataset with approx. {TOTAL_SAMPLES} samples...")
data = []
for _ in range(NUM_TRAJECTORIES):
    v0 = np.random.uniform(V0_RANGE[0], V0_RANGE[1])
    theta = np.random.uniform(THETA_RANGE[0], THETA_RANGE[1])
    
    x_coords, y_coords = calculate_full_trajectory(v0, theta)
    
    if x_coords is None:
        continue

    # From each trajectory, pick a few random points (not the very start or end)
    # This teaches the AI about intermediate points, which is the real problem.
    num_points = len(x_coords)
    if num_points > 20: # Ensure the trajectory is long enough to sample from
        for _ in range(SAMPLES_PER_TRAJECTORY):
            # Pick a random index from the middle 80% of the trajectory
            random_index = np.random.randint(int(num_points * 0.1), int(num_points * 0.9))
            
            target_x = x_coords[random_index]
            target_y = y_coords[random_index]
            
            data.append({
                'target_x': target_x,
                'target_y': target_y,
                'initial_velocity': v0,
                'launch_angle': theta
            })

# Create a DataFrame and save it
df = pd.DataFrame(data)
# Let's remove duplicates just in case
df.drop_duplicates(subset=['target_x', 'target_y'], inplace=True)
df.to_csv('projectile_dataset.csv', index=False)

print(f"Dataset with {len(df)} unique samples created successfully!")
print(df.head())