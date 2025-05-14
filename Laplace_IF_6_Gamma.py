import os
import ast
from glob import glob
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from matplotlib.ticker import ScalarFormatter
import sys
from matplotlib.ticker import MaxNLocator
import numpy as np

# Folder numbers to process
num = 500
folder_numbers = [i+1 for i in range(num)]

all_mu = []
all_sigma = []
all_r_squared = []
folders_with_plots = []
all_min_len_j=[]

shape=10.0

for folder_number in folder_numbers:
    folder_path = fr"D:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\48_1012_nodes_var_{shape}_Sh_local_S6_Cb_0.002_Gamma_14\{folder_number}"
    
    # Process "Pr_cap" files
    pr_cap_pattern = os.path.join(folder_path, "*Pr_cap*")
    matching_files = glob(pr_cap_pattern)

    all_numbers, elements_count = [], []

    if matching_files:
        print(f"\nProcessing folder {folder_number}: Found files {matching_files}")
        
        for file_path in matching_files:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            for line_num, line in enumerate(lines):
                try:
                    numbers = ast.literal_eval(line.strip())
                    if isinstance(numbers, list):  # Validate the line contains a list
                        all_numbers.append(numbers)
                        elements_count.append(len(numbers))
                except Exception as e:
                    print(f"Error processing line {line_num} in {file_path}: {line.strip()}")
                    print("Error:", str(e))
        
        if all_numbers:
            mean_list = [np.mean(sublist) for sublist in all_numbers]
            min_list = [np.min(sublist) for sublist in all_numbers]
            max_list = [np.max(sublist) for sublist in all_numbers]
            
            max_elements = max(elements_count)
            last_max_index = next(i for i in reversed(range(len(elements_count))) if elements_count[i] == max_elements)
            last_max_sublist_mean = mean_list[last_max_index]

            print(f"Last occurrence of maximum elements ({max_elements}): Found in sublist index {last_max_index}")
            print(f"Mean of this sublist: {last_max_sublist_mean}")
        else:
            print("No valid data found in 'Pr_cap' files.")
    else:
        print(f"No files containing 'Pr_cap' found in folder {folder_path}.")
    
    # Check for "X_mean" files
    x_mean_files = [file for file in os.listdir(folder_path) if file.startswith('X_mean') and file.endswith('.txt')]
    if x_mean_files:
        print("\nFound 'X_mean' files:")
        for file in x_mean_files:
            print(file)
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                contents = f.read()
                X_mean_t = [float(num) for num in contents.split()]
                if mean_list:  # Ensure mean_list exists
                    Laplace_pr = [x - m for x, m in zip(X_mean_t, mean_list)]
                    print("Laplace_pr:", Laplace_pr)
    else:
        print("\nNo 'X_mean' files found.")

    # Check for "len_j" files
    len_j_files = [file for file in os.listdir(folder_path) if file.lower().startswith('len_j') and file.endswith('.txt')]
    
    if not len_j_files:
        print("\nNo 'len_j' files found. Moving to the next folder.")
        continue  # Skip to the next folder
    
    if len_j_files:
        len_j_file = len_j_files[0]
        len_j_file_path = os.path.join(folder_path, len_j_file)
        
        # Extract token for Pr_entry if possible
        tokens = len_j_file.split('_')
        if len(tokens) > 9:
            try:
                Pr_entry = [ast.literal_eval(tokens[9])]
                print(f"\nExtracted Pr_entry from filename: {Pr_entry}")
            except Exception as e:
                print(f"Error extracting Pr_entry from filename: {e}")
        
        # Process len_J from the file
        with open(len_j_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                print(f"The file '{len_j_file}' is empty. Moving to the next folder.")
                continue  # Skip to the next folder

            if lines:
                len_J = ast.literal_eval(lines[-1].strip())
                len_J = [1 - (i / 1012) for i in len_J]
        
        # Only remove the last element if it equals 1
                if len_J and len_J[-1] == 1:
                    len_J.pop()
            
                print("Processed len_J:", len_J)
            else:
                print(f"The file '{len_j_file}' is empty.")
    else:
        print("\nNo 'len_j' files found.")
        
    

    # Exit if necessary data is missing
    if not all([Laplace_pr, len_J, Pr_entry]):
        print("\nRequired data missing for fitting.")
        continue
    
    print("Laplce pr here =",Laplace_pr)
    print("Entry pr here =",Pr_entry)
    
    # Dividing each element of Laplace_pr by Pr_entry
    ratio = [Laplace_pr1 / Pr_entry1 for Laplace_pr1 in Laplace_pr for Pr_entry1 in Pr_entry]
    #print("ratio here =",ratio)
    
    
    #sys.exit()
    

    # Define the model function
    def model(x, mu, sigma):
        return (1 - len_J[0]) * stats.norm.cdf(x / Pr_entry[0], loc=mu, scale=sigma) + len_J[0]

    # Define the function to calculate negative R-squared
    def negative_r_squared(params, x, y):
        mu, sigma = params
        y_pred = model(x, mu, sigma)
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        return -r_squared

    # Convert data to NumPy arrays
    Laplace_pr = np.array(Laplace_pr)
    len_J = np.array(len_J)

   # Initial guesses for optimization
    initial_guess = [np.mean(Laplace_pr), np.std(Laplace_pr)]

# Attempt optimization for each folder
    try:
    # Optimize parameters
        result = minimize(negative_r_squared, initial_guess, args=(Laplace_pr, len_J), bounds=[(-2,2),(0,1)], method='L-BFGS-B')
    
    # Check if optimization was successful    bounds=[(ratio[0], ratio[-1]), (len_J[0], len_J[-1])]
        if result.success:
            optimized_mu, optimized_sigma = result.x
            r_squared_optimized = -result.fun

        # Store results
            mu_list = [round(optimized_mu, 4)]
            sigma_list = [round(optimized_sigma, 4)]
            r_squared_list = [round(r_squared_optimized, 4)]
        
            print("\nOptimization Results:")
            print("Optimized μ:", mu_list)
            print("Optimized σ:", sigma_list)
            print("Optimized R²:", r_squared_list)
        else:
            print("Optimization did not converge.")
    except ValueError as e:
        print(f"Error in optimization: {e}. Moving to the next folder.")



    try:
    # Generate the fitting curve
        x_fit = np.linspace(min(Laplace_pr), max(Laplace_pr), 1000)
        y_fit_optimized = model(x_fit, optimized_mu, optimized_sigma)
        
        
    
    # Append results to the main lists
        all_mu.append(optimized_mu)
        all_sigma.append(optimized_sigma)
        all_r_squared.append(r_squared_optimized)
        
        
    # Save the plot
        destination = fr"D:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\48_1012_nodes_var_{shape}_Sh_local_S6_Cb_0.002_Gamma_14"
        save_folder = os.path.join(destination, rf"Laplace_pressure_fits_First{num}folders")
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{folder_number}.png")

        # Calculate the min and max values for Laplace_pr and len_J
        min_laplace = min(Laplace_pr)
        max_laplace = 0.0
        min_len_j = min(len_J)
        max_len_j = max(len_J)

# Determine the number of ticks based on the range
# For Laplace_pr axis (x-axis)
        x_ticks = MaxNLocator(nbins=6, integer=False).tick_values(min_laplace, max_laplace)

# For len_J axis (y-axis)
        y_ticks = MaxNLocator(nbins=6, integer=False).tick_values(min_len_j, max_len_j)

# Plot the data and the fitting curve
        plt.figure(figsize=(10, 8), dpi=500)
        plt.plot(Laplace_pr, len_J, 'bo', label='Simulation', markersize=10)
        plt.plot(x_fit, y_fit_optimized, 'r-', linewidth=3, label=f'Fitting (R²={r_squared_optimized:.3f})')

        plt.xlabel('Mean Laplace Pressure [N/m²]', fontsize=25, labelpad=15)
        plt.ylabel(r'$\mathrm{f_t} \ [-]$', fontsize=25, labelpad=15)

        plt.legend(fontsize=20.0, loc='upper left')

# Set tick format and limits
        plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

# Apply the dynamic ticks for both x and y axes
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6, prune='lower'))
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6, prune='lower'))

# Optionally, you can use the calculated `x_ticks` and `y_ticks` to explicitly set the tick values
        plt.xticks(x_ticks, fontsize=20)
        plt.yticks(y_ticks, fontsize=20)

# Adjust the plot's appearance
        plt.tick_params(axis='both', direction='in', length=6, width=1.5)
        plt.tight_layout()

# Save the plot
        save_path = os.path.join(save_folder, f"{folder_number}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()


    # Track successful folder numbers
        folders_with_plots.append(folder_number)
        all_min_len_j.append(min_len_j)

    except Exception as e:
        print(f"Error processing folder {folder_number}: {str(e)}")


# Print summary of results
print("\nProcessing complete!")
print("\nAll optimized μ values:", all_mu)
print("All optimized σ values:", all_sigma)
print("All R² values:", all_r_squared)
print("All len_j values:",all_min_len_j)

#sys.exit()

# Calculate means
if all_mu:
    mean_mu = np.mean(all_mu)
    print(f"\nMean of all optimized μ values: {mean_mu:.4f}")
else:
    print("\nNo optimized μ values to calculate mean.")

if all_sigma:
    mean_sigma = np.mean(all_sigma)
    print(f"\nMean of all optimized σ values: {mean_sigma:.4f}")
else:
    print("\nNo optimized σ values to calculate mean.")

if all_r_squared:
    mean_r_squared = np.mean(all_r_squared)
    print(f"\nMean of all optimized R² values: {mean_r_squared:.4f}")
else:
    print("\nNo optimized R² values to calculate mean.")
    
if all_min_len_j:
    mean_min_len_j = np.mean(all_min_len_j)
    print(f"\nMean of all percolation threshold  values: {mean_min_len_j:.4f}")
else:
    print("\nNo optimized percolation threshold")    

# Print folders that generated plots
print("\nFolders that generated plots:", folders_with_plots)

#sys.exit()

save_folder = os.path.join(destination, rf"Laplace_pressure_fits_First{num}folders")
os.makedirs(save_folder, exist_ok=True)

# Define the output file path
output_file = os.path.join(save_folder, "optimization_results.txt")
all_mu_sigma = [i / j for i, j in zip(all_mu, all_sigma)]


with open(output_file, 'w', encoding='utf-8') as file:
    def write_heading_and_values(heading, values):
        underline = "-" * len(heading)
        file.write(f"{heading}\n{underline}\n{values}\n\n")
    
    write_heading_and_values("All optimized μ values", all_mu)
    write_heading_and_values("All optimized σ values", all_sigma)
    write_heading_and_values("All R² values", all_r_squared)
    write_heading_and_values("All optimized percolation threshold values", all_min_len_j)
    write_heading_and_values("All folders with plots", folders_with_plots)
    write_heading_and_values("Count of folders with plots", len(folders_with_plots))
    write_heading_and_values("All mu/sigma", all_mu_sigma)  # Write the computed list to the file
    



# Filter the results where R² > 0.9
filtered_mu = [mu for mu, r2 in zip(all_mu, all_r_squared) if r2 > 0.9]
filtered_sigma = [sigma for sigma, r2 in zip(all_sigma, all_r_squared) if r2 > 0.9]
filtered_r_squared = [r2 for r2 in all_r_squared if r2 > 0.9]
filtered_min_len_j = [min_len_j for min_len_j, r2 in zip(all_min_len_j, all_r_squared) if r2 > 0.9]
# Filter the folders with R² > 0.9
filtered_folders_with_plots = [folder for folder, r2 in zip(folders_with_plots, all_r_squared) if r2 > 0.9]
#print("here =",filtered_folders_with_plots)
#sys.exit()


# Print and save filtered results
if filtered_mu:
    print(f"\nFiltered optimized μ values (R² > 0.9): {filtered_mu}")
else:
    print("\nNo optimized μ values with R² > 0.9.")

if filtered_sigma:
    print(f"\nFiltered optimized σ values (R² > 0.9): {filtered_sigma}")
else:
    print("\nNo optimized σ values with R² > 0.9.")

if filtered_r_squared:
    print(f"\nFiltered optimized R² values (R² > 0.9): {filtered_r_squared}")
else:
    print("\nNo R² values > 0.9.")
    
# Print and save filtered results for min_len_j
if filtered_min_len_j:
    print(f"\nFiltered optimized percolation threshold values (R² > 0.9): {filtered_min_len_j}")
else:
    print("\nNo optimized percolation threshold values with R² > 0.9.")    
    
# Print filtered folders with plots
if filtered_folders_with_plots:
    print(f"\nFiltered folders with plots (R² > 0.9): {filtered_folders_with_plots}")
else:
    print("\nNo folders with plots where R² > 0.9.")    

# Save the filtered results to a text file with UTF-8 encoding
output_file = os.path.join(save_folder, "filtered_optimization_results.txt")

filtered_mu_sigma = [i / j for i, j in zip(filtered_mu, filtered_sigma)]

# Find the folder numbers corresponding to the min and max R² values
min_r_squared_value = min(filtered_r_squared)
max_r_squared_value = max(filtered_r_squared)

min_r_squared_index = filtered_r_squared.index(min_r_squared_value)
max_r_squared_index = filtered_r_squared.index(max_r_squared_value)

min_r_squared_folder = folders_with_plots[min_r_squared_index]
max_r_squared_folder = folders_with_plots[max_r_squared_index]

# Find the folder numbers corresponding to the min and max percolation threshold values
min_percolation_value = min(filtered_min_len_j)
max_percolation_value = max(filtered_min_len_j)

min_percolation_index = filtered_min_len_j.index(min_percolation_value)
max_percolation_index = filtered_min_len_j.index(max_percolation_value)

min_percolation_folder = folders_with_plots[min_percolation_index]
max_percolation_folder = folders_with_plots[max_percolation_index]

with open(output_file, 'w', encoding='utf-8') as file:
    def write_heading_and_values(heading, values):
        underline = "-" * len(heading)
        file.write(f"{heading}\n{underline}\n{values}\n\n")
    
    write_heading_and_values("Filtered optimized μ values", filtered_mu)
    write_heading_and_values("Filtered optimized σ values", filtered_sigma)
    write_heading_and_values("Filtered R² values", filtered_r_squared)
    
    write_heading_and_values("Filtered R² values (Min)", f"{min_r_squared_value} (Folder: {min_r_squared_folder})")
    write_heading_and_values("Filtered R² values (Max)", f"{max_r_squared_value} (Folder: {max_r_squared_folder})")
    
    write_heading_and_values("Filtered optimized percolation threshold values", filtered_min_len_j)
    write_heading_and_values("Filtered percolation threshold values (Min)", f"{min_percolation_value} (Folder: {min_percolation_folder})")
    write_heading_and_values("Filtered percolation threshold values (Max)", f"{max_percolation_value} (Folder: {max_percolation_folder})")
    
    write_heading_and_values("Filtered folders with plots (R² > 0.9)", filtered_folders_with_plots)
    write_heading_and_values("Count of Filtered folders with plots (R² > 0.9)", len(filtered_folders_with_plots))
    
    write_heading_and_values("Filtered mu/sigma", filtered_mu_sigma)




    # Write the mean of the filtered values
    if filtered_mu:
        file.write(f"\nMean of filtered optimized μ values: {np.mean(filtered_mu):.4f}\n")
    else:
        file.write("\nNo filtered μ values to calculate mean.\n")

    if filtered_sigma:
        file.write(f"\nMean of filtered optimized σ values: {np.mean(filtered_sigma):.4f}\n")
    else:
        file.write("\nNo filtered σ values to calculate mean.\n")

    if filtered_r_squared:
        file.write(f"\nMean of filtered optimized R² values: {np.mean(filtered_r_squared):.4f}\n")
    else:
        file.write("\nNo filtered R² values to calculate mean.\n")
        
    
    # Write the mean of the filtered values for min_len_j
    if filtered_min_len_j:
        file.write(f"\nMean of filtered optimized percolation threshold values: {np.mean(filtered_min_len_j):.4f}\n")
    else:
        file.write("\nNo filtered percolation threshold values to calculate mean.\n")
        
    
    if filtered_mu_sigma:
        file.write(f"\nMean of filtered mu_sigma: {np.mean(filtered_mu_sigma):.4f}\n")
    else:
        file.write("\nNo filtered mu_sigma values to calculate mean.\n")

print(f"\nFiltered results saved to {output_file}")

# Check if the lengths of filtered_mu and filtered_sigma are the same
if len(filtered_mu) == len(filtered_sigma):
    filtered_mu_sigma = [i / j for i, j in zip(filtered_mu, filtered_sigma)]
    print("filtered_mu_sigma =", filtered_mu_sigma)
else:
    print("Error: The lengths of filtered_mu and filtered_sigma are not the same.")


bins = 20


######################################################################################################

# Create a separate histogram for filtered mu values (R² > 0.9)
plt.figure(figsize=(10, 8), dpi=500)
n, bin_edges, _ = plt.hist(all_mu, bins=bins, color='green', edgecolor='green', alpha=0.7)
plt.xlabel('μ', fontsize=25, labelpad=15)
plt.ylabel('Counts', fontsize=25, labelpad=15)

# Select alternate ticks from bin_edges and format to 3 decimal places
tick_positions = bin_edges[::2]  # Select every other tick
tick_labels = [f'{x:.3f}' for x in tick_positions]  # Format to 3 decimal places

plt.xticks(tick_positions, tick_labels, fontsize=15)  # Set x-ticks with formatted labels
plt.yticks(fontsize=15)
plt.xlim(min(all_mu), max(all_mu))  # Set x-axis limits to min and max of filtered_mu
plt.tick_params(axis='both', direction='in', length=6)  # Inward ticks for both axes
plt.tight_layout()
plt.savefig(os.path.join(destination, rf"Laplace_pressure_fits_First{num}folders", "all_mu_histogram.png"))
plt.show()


plt.figure(figsize=(10, 8), dpi=500)
n, bin_edges, _ = plt.hist(filtered_mu, bins=bins, color='blue', edgecolor='blue', alpha=0.7)
plt.xlabel('μ', fontsize=25, labelpad=15)
plt.ylabel('Counts', fontsize=25, labelpad=15)

# Select alternate ticks from bin_edges and format to 3 decimal places
tick_positions = bin_edges[::2]  # Select every other tick
tick_labels = [f'{x:.3f}' for x in tick_positions]  # Format to 3 decimal places

plt.xticks(tick_positions, tick_labels, fontsize=15)  # Set x-ticks with formatted labels
plt.yticks(fontsize=15)
plt.xlim(min(filtered_mu), max(filtered_mu))  # Set x-axis limits to min and max of filtered_mu
plt.tick_params(axis='both', direction='in', length=6)  # Inward ticks for both axes
plt.tight_layout()
plt.savefig(os.path.join(destination, rf"Laplace_pressure_fits_First{num}folders", "filtered_mu_histogram.png"))
plt.show()



######################################################################################################


# Create a separate histogram for unfiltered sigma values
plt.figure(figsize=(10, 8), dpi=500)
n, bin_edges, _ = plt.hist(all_sigma, bins=bins, color='green', edgecolor='green', alpha=0.7)
plt.xlabel('σ', fontsize=25, labelpad=15)
plt.ylabel('Counts', fontsize=25, labelpad=15)

# Select alternate ticks from bin_edges and format to 3 decimal places
tick_positions = bin_edges[::2]  # Select every other tick
tick_labels = [f'{x:.3f}' for x in tick_positions]  # Format to 3 decimal places

plt.xticks(tick_positions, tick_labels, fontsize=15)  # Set x-ticks with formatted labels
plt.yticks(fontsize=15)
plt.xlim(min(all_sigma), max(all_sigma))  # Set x-axis limits to min and max of all_sigma
plt.tick_params(axis='both', direction='in', length=6)  # Inward ticks for both axes
plt.tight_layout()
plt.savefig(os.path.join(destination, rf"Laplace_pressure_fits_First{num}folders", "all_sigma_histogram.png"))
plt.show()



# Create a separate histogram for filtered sigma values (R² > 0.9)
plt.figure(figsize=(10, 8), dpi=500)
n, bin_edges, _ = plt.hist(filtered_sigma, bins=bins, color='blue', edgecolor='blue', alpha=0.7)
plt.xlabel('σ', fontsize=25, labelpad=15)
plt.ylabel('Counts', fontsize=25, labelpad=15)

# Select alternate ticks from bin_edges and format to 3 decimal places
tick_positions = bin_edges[::2]  # Select every other tick
tick_labels = [f'{x:.3f}' for x in tick_positions]  # Format to 3 decimal places

plt.xticks(tick_positions, tick_labels, fontsize=15)  # Set x-ticks with formatted labels
plt.yticks(fontsize=15)
plt.xlim(min(filtered_sigma), max(filtered_sigma))  # Set x-axis limits to min and max of filtered_sigma
plt.tick_params(axis='both', direction='in', length=6)  # Inward ticks for both axes
plt.tight_layout()
plt.savefig(os.path.join(destination, rf"Laplace_pressure_fits_First{num}folders", "filtered_sigma_histogram.png"))
plt.show()

######################################################################################################

# Create a separate histogram for unfiltered min_len_j values
plt.figure(figsize=(10, 8), dpi=500)
n, bin_edges, _ = plt.hist(all_min_len_j, bins=bins, color='green', edgecolor='green', alpha=0.7)
plt.xlabel(r'$\mathrm{f_0} \ [-]$', fontsize=25, labelpad=15)
plt.ylabel('Counts', fontsize=25, labelpad=15)

# Select alternate ticks from bin_edges and format to 3 decimal places
tick_positions = bin_edges[::2]  # Select every other tick
tick_labels = [f'{x:.3f}' for x in tick_positions]  # Format to 3 decimal places

plt.xticks(tick_positions, tick_labels, fontsize=15)  # Set x-ticks with formatted labels
plt.yticks(fontsize=15)
plt.xlim(min(all_min_len_j), max(all_min_len_j))  # Set x-axis limits to min and max of all_min_len_j
plt.tick_params(axis='both', direction='in', length=6)  # Inward ticks for both axes
plt.tight_layout()
plt.savefig(os.path.join(destination, rf"Laplace_pressure_fits_First{num}folders", "all_percolation_threshold_histogram.png"))
plt.show()



# Create a separate histogram for filtered min_len_j values (R² > 0.9)
plt.figure(figsize=(10, 8), dpi=500)
n, bin_edges, _ = plt.hist(filtered_min_len_j, bins=bins, color='blue', edgecolor='blue', alpha=0.7)
plt.xlabel(r'$\mathrm{f_0} \ [-]$', fontsize=25, labelpad=15)
plt.ylabel('Counts', fontsize=25, labelpad=25)

# Select alternate ticks from bin_edges and format to 3 decimal places
tick_positions = bin_edges[::2]  # Select every other tick
tick_labels = [f'{x:.3f}' for x in tick_positions]  # Format to 3 decimal places

plt.xticks(tick_positions, tick_labels, fontsize=15)  # Set x-ticks with formatted labels
plt.yticks(fontsize=15)
plt.xlim(min(filtered_min_len_j), max(filtered_min_len_j))  # Set x-axis limits to min and max of filtered_min_len_j
plt.tick_params(axis='both', direction='in', length=6)  # Inward ticks for both axes
plt.tight_layout()
plt.savefig(os.path.join(destination, rf"Laplace_pressure_fits_First{num}folders", "filtered_percolation_threshold_histogram.png"))
plt.show()


######################################################################################################

# Create a separate histogram for unfiltered min_len_j values
plt.figure(figsize=(10, 8), dpi=500)
n, bin_edges, _ = plt.hist(all_mu_sigma, bins=bins, color='green', edgecolor='green', alpha=0.7)
plt.xlabel(r'μ/σ', fontsize=25, labelpad=15)
plt.ylabel('Counts', fontsize=25, labelpad=15)

# Select alternate ticks from bin_edges and format to 3 decimal places
tick_positions = bin_edges[::2]  # Select every other tick
tick_labels = [f'{x:.3f}' for x in tick_positions]  # Format to 3 decimal places

plt.xticks(tick_positions, tick_labels, fontsize=15)  # Set x-ticks with formatted labels
plt.yticks(fontsize=15)
plt.xlim(min(all_mu_sigma), max(all_mu_sigma))  # Set x-axis limits to min and max of all_min_len_j
plt.tick_params(axis='both', direction='in', length=6)  # Inward ticks for both axes
plt.tight_layout()
plt.savefig(os.path.join(destination, rf"Laplace_pressure_fits_First{num}folders", "mu_sigma_histogram.png"))
plt.show()



# Create a separate histogram for filtered min_len_j values (R² > 0.9)
plt.figure(figsize=(10, 8), dpi=500)
n, bin_edges, _ = plt.hist(filtered_mu_sigma, bins=bins, color='blue', edgecolor='blue', alpha=0.7)
plt.xlabel(r'μ/σ', fontsize=25, labelpad=15)
plt.ylabel('Counts', fontsize=25, labelpad=25)

# Select alternate ticks from bin_edges and format to 3 decimal places
tick_positions = bin_edges[::2]  # Select every other tick
tick_labels = [f'{x:.3f}' for x in tick_positions]  # Format to 3 decimal places

plt.xticks(tick_positions, tick_labels, fontsize=15)  # Set x-ticks with formatted labels
plt.yticks(fontsize=15)
plt.xlim(min(filtered_mu_sigma), max(filtered_mu_sigma))  # Set x-axis limits to min and max of filtered_min_len_j
plt.tick_params(axis='both', direction='in', length=6)  # Inward ticks for both axes
plt.tight_layout()
plt.savefig(os.path.join(destination, rf"Laplace_pressure_fits_First{num}folders", "filtered_mu_sigma_histogram.png"))
plt.show()