import os
import time
import csv
import math
import argparse
import sys

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
parser.add_argument("--particles_sheet", type = str, help = "a file containing a list of starfiles; each starfile corresponds to a single-particle dataset; NOTE, absolute directory is mandatory")
parser.add_argument("--cryosparc_user_id", type = str, help = "the E-mail address of the user of CryoSPARC")
parser.add_argument("--cryosparc_project_uid", type = str, help = "the project UID in cryoSPARC")
parser.add_argument("--cryosparc_workspace_uid", type = str, help = "the workspace UID in cryoSPARC")
parser.add_argument("--cryosparc_lane", type = str, help = "the lane for computing resource in cryoSPARC")
parser.add_argument("--molecular_symmetry", type = str, help = "molecular symmetry", default = 'C1')

# Parse the arguments
args = parser.parse_args()

# If if no arguments were provided, print help information.
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

# List to store the read particle starfiles
starfile_list = []

# Open the CSV file for reading
with open(args.particles_sheet, 'r') as file:

    reader = csv.reader(file)

    # Append each row (as a string) to the list
    for row in reader:
        starfile_list.extend(row)

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

    homo_refine_job_ids.append(client.make_job(job_type = 'homo_refine_new', \
                                               project_uid = args.cryosparc_project_uid, \
                                               workspace_uid = args.cryosparc_workspace_uid, \
                                               user_id = client.GetUser(args.cryosparc_user_id)['_id'], \
                                               params = {'refine_symmetry' : args.molecular_symmetry, 'compute_use_ssd': False, 'refine_gs_resplit' : False}, \
                                               input_group_connects = {'particles': "{}.particles_all_classes".format(ab_initio_job_id), \
                                                                       'volume'   : "{}.volume_class_0".format(ab_initio_job_id)}))

    client.enqueue_job(args.cryosparc_project_uid, homo_refine_job_ids[-1], args.cryosparc_lane, client.GetUser(args.cryosparc_user_id)['_id'])
    print("{} SUBMITTED".format(homo_refine_job_ids[-1]))

print("CHECKING STATUS OF HOMOGENEOUS REFINEMENT JOBS")
check_job_status(homo_refine_job_ids)
print("HOMOGENEOUS REFINEMENT JOBS COMPLETED")

print("===================================================================")
print("=====PERFORM NON-UNIFORM REFINEMENT FOR EACH IMPORTED DATASET======")

nonuniform_refine_job_ids = []

for ab_initio_job_id in ab_initio_job_ids:

    nonuniform_refine_job_ids.append(client.make_job(job_type = 'nonuniform_refine_new', \
                                                     project_uid = args.cryosparc_project_uid, \
                                                     workspace_uid = args.cryosparc_workspace_uid, \
                                                     user_id = client.GetUser(args.cryosparc_user_id)['_id'], \
                                                     params = {'refine_symmetry' : args.molecular_symmetry, 'compute_use_ssd': False, 'refine_gs_resplit' : False}, \
                                                     input_group_connects = {'particles': "{}.particles_all_classes".format(ab_initio_job_id), \
                                                                             'volume'   : "{}.volume_class_0".format(ab_initio_job_id)}))

    client.enqueue_job(args.cryosparc_project_uid, nonuniform_refine_job_ids[-1], args.cryosparc_lane, client.GetUser(args.cryosparc_user_id)['_id'])
    print("{} SUBMITTED".format(nonuniform_refine_job_ids[-1]))

print("CHECKING STATUS OF NON-UNIFORM REFINEMENT JOBS")
check_job_status(nonuniform_refine_job_ids)
print("NON-UNIFORM REFINEMENT JOBS COMPLETED")
