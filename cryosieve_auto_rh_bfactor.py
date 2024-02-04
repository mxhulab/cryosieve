import starfile
import os
import argparse
import sys
import pandas as pd
import re
import numpy as np

def halve_particles_in_starfile(starfile_path, iterations = 4):

    base_filename = os.path.splitext(os.path.basename(starfile_path))[0]
    directory = os.path.dirname(starfile_path)
    new_starfiles = [starfile_path]  # Start with the original file in the list

    # Read the original star file
    data = starfile.read(starfile_path)
    original_data_size = len(data)
    current_fraction = 1

    for i in range(iterations):
        current_fraction /= 2  # Update the fraction for each iteration
        reduced_size = int(original_data_size * current_fraction)

        # Sample the reduced number of particles
        reduced_data = data.sample(n=reduced_size).reset_index(drop=True)

        # Update filename to include iteration count
        new_filename = f"{base_filename}_reduced_{2**(i+1)}.star"
        new_filepath = os.path.join(directory, new_filename)

        # Save the reduced data to a new star file
        starfile.write(reduced_data, new_filepath, overwrite=True)

        # Append new starfile path to the list
        new_starfiles.append(new_filepath)

    return new_starfiles

def process_starfile_list(list_file_path, new_list_file_path):
    with open(list_file_path, 'r') as file:
        starfile_paths = file.read().splitlines()

    all_new_starfiles = []

    for starfile_path in starfile_paths:
        new_starfiles = halve_particles_in_starfile(starfile_path)
        all_new_starfiles.extend(new_starfiles)

    # Generate the list file for the new star files
    # list_filename = os.path.join(os.path.dirname(list_file_path), new_list_file)
    with open(new_list_file_path, 'w') as file:
        for path in all_new_starfiles:
            file.write(path + '\n')

# Replace 'your_list_file.txt' with the path to your file containing the list of star files
# process_starfile_list('your_list_file.txt')

print("===================================================================")
print("===============IMPORT PARTICLES FOR STARTING A JOB CHAIN===========")
print("=============YOU MAY START MULTIPLE JOB CHAINS SIMULATENOUSLY =====")

# Set up the argument parser
parser = argparse.ArgumentParser(description = "The cryosieve_auto_rh_bfactor.py is a Python script designed to automatically determined Rosenthal-Henderson B-factor by executing CryoSPARC operations via the command line.")

parser.add_argument("--particles_sheet", type = str, help = "a file containing a list of starfiles; each starfile corresponds to a single-particle dataset; NOTE, absolute directory is mandatory", required = True)
parser.add_argument("--cryosparc_user_id", type = str, help = "the E-mail address of the user of CryoSPARC", required = True)
parser.add_argument("--cryosparc_project_uid", type = str, help = "the project UID in cryoSPARC", required = True)
parser.add_argument("--cryosparc_workspace_uid", type = str, help = "the workspace UID in cryoSPARC", required = True)
parser.add_argument("--cryosparc_lane", type = str, help = "the lane for computing resource in cryoSPARC", required = True)
parser.add_argument("--molecular_symmetry", type = str, help = "molecular symmetry, default: %(default)s", default = 'C1')
parser.add_argument("--force_redo_gs_split", help = "force re-do GS split", action = 'store_true')
parser.add_argument("--nonuniform", help = "use non-uniform refinement instead of homogeneous refinement", action = 'store_true')
parser.add_argument("--halvings_times", type = str, help = "number of times executing halvings, default: %(default)s", default = 4)
parser.add_argument("--particles_sheet_with_reduction", type = str, help = "the output particle sheeting containing a series of halvings; by default, it will be based on the particles_sheet filename", default = None)
parser.add_argument("--num_repeats", type = int, help = "number of repeats for running refinement, default: %(default)s", default = 1)
parser.add_argument("--rh_bfactor_data_points", type = str, help = "the output filename containing number of particles and resolution; they are data points for determining Rosenthal-Henderson B-factor; by default; it will be based on the particle_sheet filename", default = None)
parser.add_argument("--voltage_200kev", help = "micrographs are obtained using 200 keV electron microscopy", action = 'store_true')
parser.add_argument("--voltage_300kev", help = "micrographs are obtained using 300 keV electron microscopy", action = 'store_true')

# If if no arguments were provided, print help information.
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

# Parse the arguments
args = parser.parse_args()
assert(args.voltage_200kev != args.voltage_300kev)

base_name = os.path.basename(args.particles_sheet)
name_without_suffix = os.path.splitext(base_name)[0]

if args.particles_sheet_with_reduction is None:
    args.particles_sheet_with_reduction = name_without_suffix + "_with_halvings.txt"

if args.rh_bfactor_data_points is None:
    args.rh_bfactor_data_points = name_without_suffix + "_rh_bfactor_data_points.csv"

process_starfile_list(args.particles_sheet, args.particles_sheet_with_reduction)

import subprocess

# Command to run the script with arguments
command = ["python", "cryosieve_auto_cryosparc.py", "--particles_sheet", args.particles_sheet_with_reduction, "--cryosparc_user_id", args.cryosparc_user_id, "--cryosparc_project_uid", args.cryosparc_project_uid, "--cryosparc_workspace_uid", args.cryosparc_workspace_uid, "--cryosparc_lane", args.cryosparc_lane, "--molecular_symmetry", args.molecular_symmetry]

if args.force_redo_gs_split:
    command.append("--force_redo_gs_split")
if args.nonuniform:
    command.extend(["--num_repeats_homo", 0, "--num_repeats_nonuniform", args.num_repeats])
else:
    command.extend(["--num_repeats_homo", args.num_repeats, "--num_repeats_nonuniform", 0])
# command.append("--summary")
command.extend(["--summary", "--summary_output_filename", args.rh_bfactor_data_points])
# print(command)

command = [str(item) for item in command]
python_command_string = " ".join(command)
init_command = "eval $(cryosparcm env)"
# Combine commands
combined_command = f"{init_command} && {python_command_string}"
# print(combined_command)

process = subprocess.run(combined_command, shell=True, text=True, stderr=subprocess.PIPE)

if process.returncode != 0:
    print(f"ERROR IN RUNNING CRYOSPARC AUTO-REFINEMENT: {process.stderr}")
    exit(1)

print("===COMPLETING MEASURING ROSENTHAL-HENDERSON B-FACTOR DATA POINTS===")
print("===================================================================")
print("=====CURVE FITTING TO DETERMINE ROSENTHAL-HENDERSON B-FACTOR=======")

from scipy.optimize import curve_fit

# y -> log of number of particle
# x -> resolution
# a -> RH b-factor
def rh_bfactor_curve_fit_func_200kev(x, a):
    return np.log(1437.5695) + a / (2 * x ** 2) - np.log(x)

def rh_bfactor_curve_fit_func_300kev(x, a):
    return np.log(1831.7256) + a / (2 * x ** 2) - np.log(x)

if args.voltage_200kev:
    rh_bfactor_curve_fit_func = rh_bfactor_curve_fit_func_200kev

if args.voltage_300kev:
    rh_bfactor_curve_fit_func = rh_bfactor_curve_fit_func_300kev

def get_number_of_elements(symmetry_group):
    """
    Returns the number of elements for a given molecular symmetry group.

    Args:
    symmetry_group (str): The symmetry group notation (e.g., "C3", "D5", "T", "O", "I").

    Returns:
    int: The number of elements in the symmetry group.
    """
    if symmetry_group.startswith('C'):
        # Cyclic group, number of elements is the number following 'C'
        return int(symmetry_group[1:])
    elif symmetry_group.startswith('D'):
        # Dihedral group, number of elements is twice the number following 'D'
        return 2 * int(symmetry_group[1:])
    elif symmetry_group == 'T':
        # Tetrahedral group
        return 12
    elif symmetry_group == 'O':
        # Octahedral group
        return 24
    elif symmetry_group == 'I':
        # Icosahedral group
        return 60
    else:
        raise ValueError("Invalid symmetry group notation")

# Load the CSV file into a DataFrame
data = pd.read_csv(args.rh_bfactor_data_points)
data['base_filename'] = data['filename'].apply(lambda x: re.sub(r'_reduced_\d+\.star', '.star', x))

grouped_data = data.groupby('base_filename')

# Iterate over each group and store the grouped DataFrame
for base_filename, group_df in grouped_data:

    ln_number_o_particles = np.log(get_number_of_elements(args.molecular_symmetry) * group_df['number of particles'])
    slope, _ = curve_fit(rh_bfactor_curve_fit_func, group_df['resolution(A)'], ln_number_o_particles)

    print("ROSENTHAL-HENDERSON B-FACTOR OF PARTICLES IN {} IS {:.2f}A^2".format(base_filename, slope[0]))
