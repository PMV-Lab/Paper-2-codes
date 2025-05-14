import os
import ast
import re
from glob import glob
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from matplotlib.ticker import ScalarFormatter
import sys
from matplotlib.ticker import MaxNLocator
import numpy as np
import json

# Folder numbers to process
num = 1000
folder_numbers = [i+1 for i in range(num)]

all_mu = []
all_sigma = []
all_r_squared = []
folders_with_plots = []
all_min_len_j=[]
filtered_km_list = []
filtered_r_list = []
filtered_Sh_list = []
second_distinct_indices = []

shape=10.0
tol= 1e-7 
L=500e-6
h=10e-6

t_start = 0.0
t_end = 1e2
timesteps = int(1e3* t_end) + 1
t1 = np.linspace(t_start, t_end, timesteps)

for folder_number in folder_numbers:
    folder_path = fr"D:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\48_1012_nodes_var_{shape}_Sh_local_S6_Cb_0.002_Gamma_14\{folder_number}"
    
    
    # Check if the folder contains files starting with "len_J_new"
    len_J_files = [file for file in os.listdir(folder_path) if file.startswith('len_J')]
    
        
    if len_J_files:
        #print(f"7th numeric value in filenames beginning with 'len_J' in folder '{folder}':")
        for file in len_J_files:
            # Extract numeric values from the filename using regular expressions
            numeric_values = re.findall(r'\d+\.\d+|\d+', file)
            # Convert list of strings to list of floats
            values_float = [float(val) for val in numeric_values]
            # Retrieve the 7th value
            seventh_value = values_float[7]  # Index 6 corresponds to the 7th value
            
            
            #sys.exit()
            #print(f"Filename: {file}, 7th Numeric Value: {seventh_value}")

            # Check if there exists a file name beginning with "km" and containing the seventh value
            seventh_value_str = str(seventh_value)  # Convert seventh value to string
            #print(seventh_value_str)
            km_files = [km_file for km_file in os.listdir(folder_path) if km_file.startswith('km') and seventh_value_str in km_file]
            #print(km_files)
         
            size_files = [size_file for size_file in os.listdir(folder_path) if size_file.startswith('size') and seventh_value_str in size_file 
                          and 'initial' in size_file]
            
            #print(size_files)
            
            
            
            Sh_files = [
                Sh_file for Sh_file in os.listdir(folder_path)
                if Sh_file.startswith('Sh') and seventh_value_str in Sh_file and 'initial' in Sh_file
                ]
            
            

            
            
            if km_files:
                #print(f"Files beginning with 'km' and containing the 7th value in folder '{folder}':")
                for km_file in km_files:
                    #print(km_file)
                    # Read the content of the km_file and concatenate it
                    with open(os.path.join(folder_path, km_file), 'r') as f:
                        km = f.read().strip()
                        km_list = ast.literal_eval(km)
                        print("len km_list =",len(km_list))
                        
                        filtered_indices = [i for i, val in enumerate(km_list) if val < tol]
                        # Filter non-zero elements greater than 1e-6

                        filtered_km = [val for val in km_list if val > tol]
                        print("len filtered_km =",len(filtered_km))

                        #filtered_km_list.append(filtered_km)
                        
                #sys.exit()
            
            if size_files:
                for size_file in size_files:
                    # Read the content of the size_file and concatenate it
                    with open(os.path.join(folder_path, size_file), 'r') as f:
                        size = f.read().strip()
                        size_list = ast.literal_eval(size)
                        filtered_size = [size_list[i] for i in range(len(size_list)) if i not in filtered_indices]
                        print("len filtered_size =",len(filtered_size))
                        
                        filtered_r=[1/i for i in filtered_size]
                        print("len filtered_r =",len(filtered_r))
                        
                        #filtered_r_list.append(filtered_r)
                        
                        V_A = [L * h * i / (2 * L * i + 2 * L * h) for i in filtered_r]
                        print("len V_A =",len(V_A))
                        
                        
            if Sh_files:
                for Sh_file in Sh_files:
                    # Read the content of the size_file and concatenate it
                    with open(os.path.join(folder_path, Sh_file), 'r') as f:
                        Sh = f.read().strip()
                        Sh_list = ast.literal_eval(Sh)
                        filtered_Sh = [Sh_list[i] for i in range(len(Sh_list)) if i not in filtered_indices]
                        print("len filtered_Sh =",len(filtered_Sh))
                        
                        # filtered_Sh=[1/i for i in filtered_Sh]
                        # print("len filtered_r =",len(filtered_r))
                        
                        #filtered_Sh_list.append(filtered_Sh)
                        
            # Calculate the mean of each sublist and append to a new list

# mean_Sh = [np.mean(sublist) for sublist in filtered_Sh_list]

# print("Means of each sublist:", mean_Sh)     

                        # max_Sh = np.max(filtered_Sh)

                        # print("Maximum values of each sublist:", max_Sh) 
                        
                        
            scaling = [i / j for i, j in zip(V_A, filtered_km)]
            print("scaling here =",scaling)
            
            scaling_mean=np.mean(scaling)
            print("scaling mean here =",scaling_mean)
                        
                        
            
            

        #sys.exit()
        ##########################################################################################################

        #Replace s with every new var



        t_start = 0.0
        t_end = 1e2
        timesteps = int(1e3* t_end) + 1
        t1 = np.linspace(t_start, t_end, timesteps)
        #print(timesteps)

        time_values=[(i-1)*(t1[1]-t1[0]) for i in second_distinct_indices]
        #print("s =",time_values)

 
        #sys.exit()
        
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
                    #print(f"\nExtracted Pr_entry from filename: {Pr_entry}")
                except Exception as e:
                    print(f"Error extracting Pr_entry from filename: {e}")
            
            # Process len_J from the file
            with open(len_j_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines:
                    #print(f"The file '{len_j_file}' is empty. Moving to the next folder.")
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
            
        
        # print(len_J)
        
        # sys.exit()
        
        # len_J=[172, 171, 170, 167, 163, 161, 159, 154, 147, 142, 137, 130, 122, 114, 107, 99, 89, 77, 67, 60, 52, 45, 41, 33, 30, 24, 21, 21, 18, 16, 15, 14, 13, 13, 13, 13, 11, 8, 7, 7, 7, 6, 5, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        # len_J=[1-(i/1012) for i in len_J]
    

    # Define the model function
        def model(t1, mu, sigma):
            return (1 - len_J[0]) * stats.norm.cdf(t1[:len(len_J)] / scaling_mean, loc=mu, scale=sigma) + len_J[0]

    # Define the function to calculate negative R-squared
        def negative_r_squared(params, x, y):
            mu, sigma = params
            y_pred = model(x, mu, sigma)
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot)
            return -r_squared
    
        
        # Initial guesses for mu and sigma
        initial_params = [np.mean(t1[:len(len_J)]), np.std(t1[:len(len_J)])]
        
        x = t1[:len(len_J)]
        y = np.array(len_J)

# Minimize negative R-squared
        result = minimize(negative_r_squared, initial_params, args=(x, y), bounds=[(-2,2),(0,1.0)], method='L-BFGS-B')

# Extract optimized parameters
        mu_opt, sigma_opt = result.x
        
        print("mu opt =",mu_opt)
        print("sigma opt =",sigma_opt)

# Generate predicted values
        # Calculate R² value after optimization
        # Generate predicted values using the optimized parameters
        y_pred = model(x, mu_opt, sigma_opt)

# Calculate R² value after optimization
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # mu_list = [round(mu_opt, 4)]
        # sigma_list = [round(sigma_opt, 4)]
        # r_squared_list = [round(r_squared, 4)]
    
        # print("\nOptimization Results:")
        # print("Optimized μ:", mu_list)
        # print("Optimized σ:", sigma_list)
        # print("Optimized R²:", r_squared_list)
        
        all_mu.append(mu_opt)
        all_sigma.append(sigma_opt)
        all_r_squared.append(r_squared)
        folders_with_plots.append(folder_number)
        
        destination = fr"D:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\48_1012_nodes_var_{shape}_Sh_local_S6_Cb_0.002_Gamma_14"
        save_folder = os.path.join(destination, rf"Mass_transfer_fits_First{num}folders")
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{folder_number}.png")
    
        
        x_ticks = MaxNLocator(nbins=6, integer=False).tick_values(np.min(x), np.max(x))

# For len_J axis (y-axis)
        y_ticks = MaxNLocator(nbins=6, integer=False).tick_values(np.min(y), np.max(y))

# Plot the data and model fit
        plt.figure(figsize=(10, 8), dpi=500)
        plt.plot(x, y, 'bo', label='Simulation', markersize=10)
        plt.plot(x, y_pred, 'r-', linewidth=3, label=f'Fitting (R²={r_squared:.3f})')
        plt.xlabel('Elapsed time [s]', fontsize=25, labelpad=15)
        plt.ylabel(r'$\mathrm{f_t} \ [-]$', fontsize=25, labelpad=15)
        plt.legend(fontsize=20.0, loc='lower right')
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


print("\nFolders that generated plots:", folders_with_plots)

save_folder = os.path.join(destination, rf"Mass_transfer_fits_First{num}folders")
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
    write_heading_and_values("All mu/sigma", all_mu_sigma)  # Write the computed list to th        


filtered_mu = [mu for mu, r2 in zip(all_mu, all_r_squared) if r2 > 0.9]
filtered_sigma = [sigma for sigma, r2 in zip(all_sigma, all_r_squared) if r2 > 0.9]
filtered_r_squared = [r2 for r2 in all_r_squared if r2 > 0.9]
#filtered_min_len_j = [min_len_j for min_len_j, r2 in zip(all_min_len_j, all_r_squared) if r2 > 0.9]
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
# if filtered_min_len_j:
#     print(f"\nFiltered optimized percolation threshold values (R² > 0.9): {filtered_min_len_j}")
# else:
#     print("\nNo optimized percolation threshold values with R² > 0.9.")    
    
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
# min_percolation_value = min(filtered_min_len_j)
# max_percolation_value = max(filtered_min_len_j)

# min_percolation_index = filtered_min_len_j.index(min_percolation_value)
# max_percolation_index = filtered_min_len_j.index(max_percolation_value)

# min_percolation_folder = folders_with_plots[min_percolation_index]
# max_percolation_folder = folders_with_plots[max_percolation_index]

with open(output_file, 'w', encoding='utf-8') as file:
    def write_heading_and_values(heading, values):
        underline = "-" * len(heading)
        file.write(f"{heading}\n{underline}\n{values}\n\n")
    
    write_heading_and_values("Filtered optimized μ values", filtered_mu)
    write_heading_and_values("Filtered optimized σ values", filtered_sigma)
    write_heading_and_values("Filtered R² values", filtered_r_squared)
    
    write_heading_and_values("Filtered R² values (Min)", f"{min_r_squared_value} (Folder: {min_r_squared_folder})")
    write_heading_and_values("Filtered R² values (Max)", f"{max_r_squared_value} (Folder: {max_r_squared_folder})")
    
    # write_heading_and_values("Filtered optimized percolation threshold values", filtered_min_len_j)
    # write_heading_and_values("Filtered percolation threshold values (Min)", f"{min_percolation_value} (Folder: {min_percolation_folder})")
    # write_heading_and_values("Filtered percolation threshold values (Max)", f"{max_percolation_value} (Folder: {max_percolation_folder})")
    
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
    # if filtered_min_len_j:
    #     file.write(f"\nMean of filtered optimized percolation threshold values: {np.mean(filtered_min_len_j):.4f}\n")
    # else:
    #     file.write("\nNo filtered percolation threshold values to calculate mean.\n")
        
    
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
plt.savefig(os.path.join(destination, rf"Mass_transfer_fits_First{num}folders", "all_mu_histogram.png"))
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
plt.savefig(os.path.join(destination, rf"Mass_transfer_fits_First{num}folders", "filtered_mu_histogram.png"))
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
plt.savefig(os.path.join(destination, rf"Mass_transfer_fits_First{num}folders", "all_sigma_histogram.png"))
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
plt.savefig(os.path.join(destination, rf"Mass_transfer_fits_First{num}folders", "filtered_sigma_histogram.png"))
plt.show()

######################################################################################################
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
plt.savefig(os.path.join(destination, rf"Mass_transfer_fits_First{num}folders", "mu_sigma_histogram.png"))
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
plt.savefig(os.path.join(destination, rf"Mass_transfer_fits_First{num}folders", "filtered_mu_sigma_histogram.png"))
plt.show()