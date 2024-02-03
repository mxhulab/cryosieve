import os
import time
import csv
import math
import argparse
import sys
import re

from cryosparc_compute import client as cryosparc_client

# OPEN THE HOST AND CLIENT OF CRYOSPARC, SET UP THE COMMAND CORE PORT

print("===================================================================")
print("==================CRYOSPARC ENVIRONMENT INFORMATION================")

host = os.environ['CRYOSPARC_MASTER_HOSTNAME']                                 # host
print('HOST NAME OF CRYOSPARC: {}'.format(host))
command_core_port = os.environ['CRYOSPARC_COMMAND_CORE_PORT']                  # port
print('COMMAND CORE PORT OF CRYOSPARC: {}'.format(command_core_port))
client = cryosparc_client.CommandClient(host = host, port = command_core_port) # client

print("===================================================================")
print("===============IMPORT PARTICLES FOR STARTING A JOB CHAIN===========")
print("=============YOU MAY START MULTIPLE JOB CHAINS SIMULATENOUSLY =====")

# Set up the argument parser
parser = argparse.ArgumentParser(description = "The cryosieve_auto_cryosparc.py is a Python script designed to automate CryoSPARC operations via the command line. Its purpose is to bypass the labor-intensive manual processes.")
parser.add_argument("--particles_sheet", type = str, help = "a file containing a list of starfiles; each starfile corresponds to a single-particle dataset; NOTE, absolute directory is mandatory", required = True)
parser.add_argument("--cryosparc_user_id", type = str, help = "the E-mail address of the user of CryoSPARC", required = True)
parser.add_argument("--cryosparc_project_uid", type = str, help = "the project UID in cryoSPARC", required = True)
parser.add_argument("--cryosparc_workspace_uid", type = str, help = "the workspace UID in cryoSPARC", required = True)
parser.add_argument("--cryosparc_lane", type = str, help = "the lane for computing resource in cryoSPARC", required = True)
parser.add_argument("--molecular_symmetry", type = str, help = "molecular symmetry, default: %(default)s", default = 'C1')
parser.add_argument("--force_redo_gs_split", help = "force re-do GS split", action = 'store_true')
parser.add_argument("--num_repeats_homo", type = int, help = "number of repeats for running homogenous refinement, default: %(default)s", default = 1)
parser.add_argument("--num_repeats_nonuniform", type = int, help = "number of repeats for running non-uniform refinement, default: %(default)s", default = 1)
parser.add_argument("--summary", help = "summarize the refinements, including resolution and B-factors", action = 'store_true')
parser.add_argument("--summary_output_filename", type = str, help = "the output filename for the summary; by default, it will be based on the particles_sheet filename", default = None)

# If if no arguments were provided, print help information.
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

# Parse the arguments
args = parser.parse_args()

# List to store the read particle starfiles
starfile_list = []

# Open the CSV file for reading
with open(args.particles_sheet, 'r') as file:

    reader = csv.reader(file)

    # Append each row (as a string) to the list
    for row in reader:
        starfile_list.extend(row)

if args.summary:

    if args.summary_output_filename is None:
        # Prepare the summary output file
        base_name = os.path.basename(args.particles_sheet)
        name_without_suffix = os.path.splitext(base_name)[0]
        summary_filename = "summary_" + name_without_suffix + ".csv"
    else:
        summary_filename = args.summary_output_filename

    csvfile = open(summary_filename, 'w', newline='')
    # Create a CSV writer object
    csvwriter = csv.writer(csvfile)
    # Write the header row
    csvwriter.writerow(['filename', 'number of particles', 'refinement type', 'repeat index', 'resolution(A)', 'b-factor(A^2)'])

# The function of obtaining number of particles, resolution and B-factor from streamlog
def get_num_particles_resolution_b_factor_from_steamlog(steamlog):

    text_list = [entry['text'].strip() for entry in streamlog if 'text' in entry]

    resolution_value = None
    bfactor_value = None
    num_particles_split_A = None
    num_particles_split_B = None
    for text in text_list:
        if text.startswith('Using Filter Radius'):
            resolution_match = re.search(r'\(([\d.]+)A\)', text)
            resolution_value = float(resolution_match.group(1))
        elif text.startswith('Estimated Bfactor:'):
            bfactor_match = re.search(r'Estimated Bfactor: (-[\d.]+)', text)
            bfactor_value = float(bfactor_match.group(1))
        elif text.startswith('Split A has'):
            np_split_A_match = re.search(r'\b(\d+)\b', text)
            num_particles_split_A = int(np_split_A_match.group(1))
        elif text.startswith('Split B has'):
            np_split_B_match = re.search(r'\b(\d+)\b', text)
            num_particles_split_B = int(np_split_B_match.group(1))

    return num_particles_split_A + num_particles_split_B, resolution_value, bfactor_value

import_job_ids = []

for starfile in starfile_list:

    print("LOADING PARTICLES OF {}".format(starfile))

    import_job_ids.append(client.make_job(job_type = 'import_particles', \
                                          project_uid = args.cryosparc_project_uid, \
                                          workspace_uid = args.cryosparc_workspace_uid, \
                                          user_id = client.GetUser(args.cryosparc_user_id)['_id'], \
                                          params = {'particle_meta_path' : starfile}))

    client.enqueue_job(args.cryosparc_project_uid, import_job_ids[-1], args.cryosparc_lane, client.GetUser(args.cryosparc_user_id)['_id'])
    print("{} SUBMITTED".format(import_job_ids[-1]))

print("CHECKING STATUS OF IMPORTING PARTICLES JOBS")

def check_job_status(job_ids):

    while True:
    
        counter = 0
        for job_id in job_ids:
            job_stat = client.get_job_status(args.cryosparc_project_uid, job_id)
            if job_stat == 'failed':
                print("JOB {} FAILED. PLEASE CHECK.".format(job_id))
            if job_stat == 'completed':
                counter += 1

        if counter == len(job_ids):
            break

        time.sleep(30)

check_job_status(import_job_ids)
print("LOADING PARTICLES JOBS COMPLETED")

print("===================================================================")
print("=====PERFORM AB INIITO REFINEMENT FOR EACH IMPORTED DATASET========")

ab_initio_job_ids = []

for import_job_id in import_job_ids:

    ab_initio_job_ids.append(client.make_job(job_type = 'homo_abinit', \
                                             project_uid = args.cryosparc_project_uid, \
                                             workspace_uid = args.cryosparc_workspace_uid, \
                                             user_id = client.GetUser(args.cryosparc_user_id)['_id'], \
                                             params = {'compute_use_ssd': False}, \
                                             input_group_connects = {'particles': "{}.imported_particles".format(import_job_id)}))

    client.enqueue_job(args.cryosparc_project_uid, ab_initio_job_ids[-1], args.cryosparc_lane, client.GetUser(args.cryosparc_user_id)['_id'])
    print("{} SUBMITTED".format(ab_initio_job_ids[-1]))

print("CHECKING STATUS OF AB INITIO REFINEMENT JOBS")
check_job_status(ab_initio_job_ids)
print("AB INITIO REFINEMENT JOBS COMPLETED")

print("===================================================================")
print("====PERFORM HOMOGENEOUS REFINEMENT FOR EACH IMPORTED DATASET=======")

homo_refine_job_ids = []

for ab_initio_job_id in ab_initio_job_ids:

    for _ in range(args.num_repeats_homo):
        homo_refine_job_ids.append(client.make_job(job_type = 'homo_refine_new', \
                                                   project_uid = args.cryosparc_project_uid, \
                                                   workspace_uid = args.cryosparc_workspace_uid, \
                                                   user_id = client.GetUser(args.cryosparc_user_id)['_id'], \
                                                   params = {'refine_symmetry' : args.molecular_symmetry, 'compute_use_ssd': False, 'refine_gs_resplit' : args.force_redo_gs_split}, \
                                                   input_group_connects = {'particles': "{}.particles_all_classes".format(ab_initio_job_id), \
                                                                           'volume'   : "{}.volume_class_0".format(ab_initio_job_id)}))

        client.enqueue_job(args.cryosparc_project_uid, homo_refine_job_ids[-1], args.cryosparc_lane, client.GetUser(args.cryosparc_user_id)['_id'])
        print("{} SUBMITTED".format(homo_refine_job_ids[-1]))

print("CHECKING STATUS OF HOMOGENEOUS REFINEMENT JOBS")
check_job_status(homo_refine_job_ids)
print("HOMOGENEOUS REFINEMENT JOBS COMPLETED")

if args.summary:
    print("SUMMARYING HOMOGENEOUS REFINEMENT JOBS")
    for i, job_id in enumerate(homo_refine_job_ids):
        streamlog = client.get_job_streamlog(args.cryosparc_project_uid, job_id)
        num_particles, resolution, bfactor = get_num_particles_resolution_b_factor_from_steamlog(streamlog)
        # print("{} : NUMBER OF PARTICLES {}, RESOLUTION {}, B-FACTOR {}".format(job_id, num_particles, resolution, bfactor))
        csvwriter.writerow([starfile_list[i // args.num_repeats_homo], num_particles, 'homogeneous', 1 + i % args.num_repeats_homo, resolution, bfactor]) 

print("===================================================================")
print("=====PERFORM NON-UNIFORM REFINEMENT FOR EACH IMPORTED DATASET======")

nonuniform_refine_job_ids = []

for ab_initio_job_id in ab_initio_job_ids:

    for _ in range(args.num_repeats_nonuniform):
        nonuniform_refine_job_ids.append(client.make_job(job_type = 'nonuniform_refine_new', \
                                                         project_uid = args.cryosparc_project_uid, \
                                                         workspace_uid = args.cryosparc_workspace_uid, \
                                                         user_id = client.GetUser(args.cryosparc_user_id)['_id'], \
                                                         params = {'refine_symmetry' : args.molecular_symmetry, 'compute_use_ssd': False, 'refine_gs_resplit' : args.force_redo_gs_split}, \
                                                         input_group_connects = {'particles': "{}.particles_all_classes".format(ab_initio_job_id), \
                                                                                 'volume'   : "{}.volume_class_0".format(ab_initio_job_id)}))

        client.enqueue_job(args.cryosparc_project_uid, nonuniform_refine_job_ids[-1], args.cryosparc_lane, client.GetUser(args.cryosparc_user_id)['_id'])
        print("{} SUBMITTED".format(nonuniform_refine_job_ids[-1]))

print("CHECKING STATUS OF NON-UNIFORM REFINEMENT JOBS")
check_job_status(nonuniform_refine_job_ids)
print("NON-UNIFORM REFINEMENT JOBS COMPLETED")

if args.summary:
    print("SUMMARYING NON-UNIFORM REFINEMENT JOBS")
    for i, job_id in enumerate(nonuniform_refine_job_ids):
        streamlog = client.get_job_streamlog(args.cryosparc_project_uid, job_id)
        num_particles, resolution, bfactor = get_num_particles_resolution_b_factor_from_steamlog(streamlog)
        # print("{} : NUMBER OF PARTICLES {}, RESOLUTION {}, B-FACTOR {}".format(job_id, num_particles, resolution, bfactor))
        csvwriter.writerow([starfile_list[i // args.num_repeats_nonuniform], num_particles, 'homogeneous', 1 + i % args.num_repeats_nonuniform, resolution, bfactor]) 

if args.summary:
    csvfile.close()
