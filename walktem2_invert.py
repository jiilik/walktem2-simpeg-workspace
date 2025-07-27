#! /usr/bin/env python3
"""
Copyright Troy Unrau, troy@lithogen.ca, 2025

Installation:
Use the instructions here to install simpeg, following the Conda-Forge version of the guide. 

conda install --channel conda-forge simpeg
conda install --channel conda-forge discretize

Usage:

"""
# SimPEG functionality
import simpeg.electromagnetics.time_domain as tdem
from simpeg.utils import plot_1d_layer_model, download, mkvc
from simpeg import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
)

# discretize functionality
from discretize import TensorMesh

# Basic Python functionality
import os
import tarfile
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.size": 14})


#data_filename = 'inv_tdem_1d_files/em1dtm_data.txt'
usf_filename = '20231101_230517_345_Station1.usf'

with open(usf_filename, 'r') as f:
    # need two arrays: "times" and "dobs" -- should be numpy arrays of floats
    # since we're reading in a loop, us a python list initially
    times = []
    dobs = []
    ingest = False
    for line in f:
        
        if ingest:
            if line.startswith("/END"):
                ingest = False
                break
            time, dob, q = line.split()
            if q == "1":
            #if True:
                times += [float(time[:-1])]
                dobs += [float(dob)]
            continue
        
        if line.startswith("/CURRENT:"):
            current = float(line.split()[1])
            print("Current: ", current)

        if line.strip().startswith("TIME"):
            ingest = True
            
times = np.array(times)
dobs = np.array(dobs)

#print(times)
#print(dobs)

#fig = plt.figure(figsize=(5, 5))
#ax = fig.add_axes([0.15, 0.15, 0.8, 0.75])
#ax.loglog(times, np.abs(dobs), "k-o", lw=3)
#ax.grid(which="both")
#ax.set_xlabel("Times (s)")
#ax.set_ylabel("|B| (T)")
#ax.set_title("Observed Data")
#plt.show()



# Source loop geometry
source_location = np.array([0.0, 0.0, 1.0])  # (3, ) numpy.array_like
source_orientation = "z"  # "x", "y" or "z"
source_current = current  # maximum on-time current (A)
source_radius = 40.0  # source loop radius (m)

# Receiver geometry
receiver_location = np.array([0.0, 0.0, 1.0])  # or (N, 3) numpy.ndarray
receiver_orientation = "z"  # "x", "y" or "z"

# Receiver list
receiver_list = []
receiver_list.append(
    tdem.receivers.PointMagneticFluxDensity(
        receiver_location, times, orientation=receiver_orientation
    )
)

# Define the source waveform.
waveform = tdem.sources.StepOffWaveform()

# Sources
source_list = [
    tdem.sources.CircularLoop(
        receiver_list=receiver_list,
        location=source_location,
        waveform=waveform,
        current=source_current,
        radius=source_radius,
    )
]

# Survey
survey = tdem.Survey(source_list)

# 5% of the absolute value
uncertainties = 0.05 * np.abs(dobs) * np.ones(np.shape(dobs))

data_object = data.Data(survey, dobs=dobs, standard_deviation=uncertainties)

# estimated host conductivity (S/m)
estimated_conductivity = 1

# minimum diffusion distance
d_min = 1250 * np.sqrt(times.min() / estimated_conductivity)
print("MINIMUM DIFFUSION DISTANCE: {} m".format(d_min))

# maximum diffusion distance
d_max = 1250 * np.sqrt(times.max() / estimated_conductivity)
print("MAXIMUM DIFFUSION DISTANCE: {} m".format(d_max))

depth_min = 10  # top layer thickness
depth_max = 800.0  # depth to lowest layer
geometric_factor = 1.15  # rate of thickness increase

# Increase subsequent layer thicknesses by the geometric factors until
# it reaches the maximum layer depth.
layer_thicknesses = [depth_min]
while np.sum(layer_thicknesses) < depth_max:
    layer_thicknesses.append(geometric_factor * layer_thicknesses[-1])

n_layers = len(layer_thicknesses) + 1  # Number of layers

log_conductivity_map = maps.ExpMap(nP=n_layers)

# Starting model is log-conductivity values (S/m)
starting_conductivity_model = np.log(1e-1 * np.ones(n_layers))

# Reference model is also log-resistivity values (S/m)
reference_conductivity_model = starting_conductivity_model.copy()

simulation_L2 = tdem.Simulation1DLayered(
    survey=survey, thicknesses=layer_thicknesses, sigmaMap=log_conductivity_map
)

dmis_L2 = data_misfit.L2DataMisfit(simulation=simulation_L2, data=data_object)

# Define 1D cell widths
h = np.r_[layer_thicknesses, layer_thicknesses[-1]]
h = np.flipud(h)

# Create regularization mesh
regularization_mesh = TensorMesh([h], "N")
print(regularization_mesh)

reg_L2 = regularization.WeightedLeastSquares(
    regularization_mesh,
    length_scale_x=10.0,
    reference_model=reference_conductivity_model,
    reference_model_in_smooth=False,
)

opt_L2 = optimization.InexactGaussNewton(
    maxIter=100, maxIterLS=20, maxIterCG=20, tolCG=1e-3
)

inv_prob_L2 = inverse_problem.BaseInvProblem(dmis_L2, reg_L2, opt_L2)

update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=5)
beta_schedule = directives.BetaSchedule(coolingFactor=2.0, coolingRate=3)
target_misfit = directives.TargetMisfit(chifact=1.0)

directives_list_L2 = [update_jacobi, starting_beta, beta_schedule, target_misfit]

# Here we combine the inverse problem and the set of directives
inv_L2 = inversion.BaseInversion(inv_prob_L2, directives_list_L2)

# Run the inversion
recovered_model_L2 = inv_L2.run(starting_conductivity_model)

dpred_L2 = simulation_L2.dpred(recovered_model_L2)

fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.75])
ax1.loglog(times, np.abs(dobs), "k-o")
ax1.loglog(times, np.abs(dpred_L2), "b-o")
ax1.grid(which="both")
ax1.set_xlabel("times (s)")
ax1.set_ylabel("Bz (T)")
ax1.set_title("Predicted and Observed Data")
ax1.legend(["Observed", "L2 Inversion"], loc="upper right")
plt.show()

# Load the true model and layer thicknesses
#true_conductivities = np.array([0.1, 1.0, 0.1])
#true_layers = np.r_[40.0, 40.0, 160.0]

# Plot true model and recovered model
fig = plt.figure(figsize=(6, 6))

ax1 = fig.add_axes([0.2, 0.15, 0.7, 0.7])
#plot_1d_layer_model(true_layers, true_conductivities, ax=ax1, color="k")
plot_1d_layer_model(
    layer_thicknesses, log_conductivity_map * recovered_model_L2, ax=ax1, color="b"
)
ax1.grid()
ax1.set_xlabel(r"Resistivity ($\Omega m$)")
#x_min, x_max = true_conductivities.min(), true_conductivities.max()
#ax1.set_xlim(0.8 * x_min, 1.5 * x_max)
#ax1.set_ylim([np.sum(true_layers), 0])
ax1.legend(["True Model", "L2-Model"])
plt.show()

# could cut here

sys.exit()
