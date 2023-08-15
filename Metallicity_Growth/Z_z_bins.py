import numpy as np
import dask.array as da
import h5py
import illustris_python as il
from tqdm import tqdm
import scida 
from scida import load

# Define the base path
basePath = "/virgotng/universe/IllustrisTNG/TNG100-3/output/"


# Function to find the redshift value for a given snapshot
def find_value_in_snap_Z(value, snap_array):
    snap_Z = np.array(snap_array)
    return snap_Z[snap_Z[:, 0] == value, 1][0]

# Initialize the snap_Z list
snap_Z = []
for i in range(100):
    redshift = il.groupcat.loadHeader(basePath, i)["Redshift"]
    snap_Z.append([i, redshift])

# Define mass bins
mass_bins = [1e14, 1e13, 1e12, 1e11, 1e10]
keys = ["14+", "13-14", "12-13", "11-12", "10-11", "-10"]

# Initialize the final result dictionary with mass bin keys
bin_Z_z = {key: [] for key in keys}
bin_Z_z["Redshift"] = []
redshifts = []
# Loop over snapshots from 2 to 100
for snap in tqdm(range(2, 100), desc="Processing snapshots"):
    ds = load(basePath + f"/snapdir_{snap:03d}", units=False)
    h = ds.header["Redshift"]
    gas = ds.data["gas"]
    GFM_Z = gas["GFM_Metallicity"].compute()
    group = ds["Group"]

    group_mass = group["GroupMass"].compute() * 1e10 / h

    # Initialize the bin_of_indexes dictionary
    bin_of_indexes = {key: {'halo_ID': [], 'halo_mean_Z': []} for key in keys}

    # Digitize the array
    bin_indices = np.digitize(group_mass, mass_bins, right=True)

    # Iterate through bin_indices and append the original index to the corresponding bin
    for index, bin_index in enumerate(bin_indices):
        key = keys[bin_index]
        bin_of_indexes[key]['halo_ID'].append(index)

    num_gas_group = group["GroupLenType"][:, 0].compute()
    num_gas_group_cumsum = np.cumsum(num_gas_group)

    # Function to calculate mean metallicity
    def get_meanZ_slice_indices(halo_id, num_gas_group_cumsum):
        start_index = 0 if halo_id == 0 else num_gas_group_cumsum[halo_id - 1]
        end_index = num_gas_group_cumsum[halo_id]
        is_empty_halo = start_index == end_index
        meanZ = 0 if is_empty_halo else GFM_Z[start_index:end_index].mean()
        return start_index, end_index, meanZ, is_empty_halo

    prim_sol = 0.0127

    # Find the redshift key
    z_key = np.round(find_value_in_snap_Z(snap, snap_Z), 3)

    # Iterate through mass bins and calculate mean metallicity
    for mass_bin in bin_of_indexes.keys():
        mean_Z_values = []
        for i in bin_of_indexes[mass_bin]['halo_ID']:
            _, _, mean_Z_halo, _ = get_meanZ_slice_indices(i, num_gas_group_cumsum)
            mean_Z_halo /= prim_sol
            mean_Z_values.append(mean_Z_halo)

        # Update the final result dictionary with redshift and mean metallicity
        mean_Z = np.nanmean(mean_Z_values) if mean_Z_values else 0
        bin_Z_z[mass_bin].append((mean_Z))
        redshifts.append(z_key)

bin_Z_z['Redshift'].append((redshifts))
# Saving the bin_Z_z dictionary into a text file
with open('bin_Z_z.txt', 'w') as file:
    for mass_bin, values in bin_Z_z.items():
        redshift_str = ", ".join([f"{value}" for value in values])
        file.write(f"{mass_bin}: {redshift_str}\n")