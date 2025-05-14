These are the codes for the manuscript "The evolution of invasion patterns due to surfactant adsorption in anomalous pore distribution: Role of Mass Transfer and Laplace Pressure".

Gamma_2.py - This code Generates Gamma distribution 

S6_AboveCMC_Cb_0.002_Sh1_local_Pr_2.py - This code iterates through various capillary pressures until there is a connected path from inlet to outlet

S6_AboveCMC_Cb_0.002_Sh1_local_2.py - After a connected path is established, this code tracks the interfaces which are invaded over time due to effect of surfactant. This also includes functions which calculates the pressure drop, flux, velocities. 

Data analysis:

Laplace_IF_6_Gamma.py - This code plots invaded fraction with changing Laplace pressure and fits the trend with the cdf of the normal distribution, with the Laplace pressure scaled with the entry pressure. 

MT_IF_Gamma.py - This code plots invaded fraction over time and fits the trend with the cdf of the normal distribution, with the time scaled with the mass transfer timescale at the onset of secondary invasion. 




