from scipy.stats import gamma  # Import gamma distribution
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

Runs = 1000
Nodes = 1012
start_seed = 1
end_seed = 2147483647

# Gamma parameters
shape = 10.0  # k (shape parameter)
scale = (50e-6) / shape  # θ (scale parameter) calculated to achieve mean = 50e-6

def function(run):
    parent_folder = str(run + 1)
    os.makedirs(parent_folder, exist_ok=True)

    seed = np.random.randint(start_seed, end_seed + 1)
    np.random.seed(seed)
    
    # Generate gamma distributed samples
    gamma_samples = gamma.rvs(a=shape, scale=scale, size=Nodes)
    
    # Sort the generated samples
    sort_r = np.sort(gamma_samples)
    r1 = sort_r[::-1].reshape(1, Nodes)  # Sorted in descending order

    r2 = gamma_samples.copy()
    np.random.shuffle(r2)

    # Prepare inverse radius values
    inv_r = 1 / r2

    # Save data in CSV file
    filename = f"Inv_Radius_{Nodes}_{shape:.1f}_{scale:.1e}GAMMA.csv"
    filepath = os.path.join(parent_folder, filename)
    
    with open(filepath, 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(inv_r)

    # Plot Gamma and Inverse Values
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(gamma_samples, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Gamma Samples (Run {run + 1})\nShape={shape}, Scale={scale:.3e}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(inv_r, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.title(f'Inverse Gamma Samples (Run {run + 1})')
    plt.xlabel('Inverse Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(parent_folder, f'Histogram_Gamma_Run_{run + 1}.png'))
    plt.close()
    
    # Calculate and return statistics for this run
    return {
        'Run': run + 1,
        'Original_Mean': np.mean(gamma_samples),
        'Original_Variance': np.var(gamma_samples)
    }

# Compute statistics
all_stats = []
for x in range(Runs):
    stats = function(x)
    all_stats.append(stats)

# Write statistics to a text file
with open('simulation_stats.txt', 'w') as stat_file:
    stat_file.write("\nStatistics for each run:\n")
    stat_file.write("-" * 80 + "\n")
    stat_file.write(f"{'Run':<6} {'Original Mean':<15} {'Original Var':<15}\n")
    stat_file.write("-" * 80 + "\n")
    for stat in all_stats:
        stat_file.write(f"{stat['Run']:<6} {stat['Original_Mean']:<15.6e} {stat['Original_Variance']:<15.6e}\n")
    stat_file.write("-" * 80 + "\n")

    stat_file.write("\nOverall Statistics:\n")
    stat_file.write("-" * 40 + "\n")
    stat_file.write(f"Average Original Mean: {np.mean([stat['Original_Mean'] for stat in all_stats]):.6e}\n")
    stat_file.write(f"Average Original Variance: {np.mean([stat['Original_Variance'] for stat in all_stats]):.6e}\n")

    stat_file.write("\nFirst, Last, Minimum, and Maximum Values from Each CSV File:\n")
    stat_file.write("-" * 140 + "\n")
    stat_file.write(f"{'Run':<6} {'First Value':<20} {'Last Value':<20} {'Minimum Value':<20} {'Maximum Value':<20}\n")
    stat_file.write("-" * 140 + "\n")
    
    for run in range(Runs):
        folder = str(run + 1)
        filename = f"Inv_Radius_{Nodes}_{shape:.1f}_{scale:.1e}GAMMA.csv"
        filepath = os.path.join(folder, filename)
        
        df = pd.read_csv(filepath, header=None)
        first_value = 1 / df.iloc[0, 0]
        last_value = 1 / df.iloc[0, -1]
        min_value = 1 / df.iloc[0].max()
        max_value = 1 / df.iloc[0].min()
        
        stat_file.write(f"{run+1:<6} {first_value:<20.6e} {last_value:<20.6e} {min_value:<20.6e} {max_value:<20.6e}\n")
    stat_file.write("-" * 140 + "\n")