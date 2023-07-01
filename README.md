# Overview

CryoSieve is a particle sorting and sieving software for single particle analysis in cryo-EM.

<!-- The paper can be referred as XXX. -->

# Installation

CryoSieve is an open source software developed in Python programming language and released as a Python package. The source code can be found [on GitHub](OUR_GITHUB_REPO).

We recommend using Python version >= 3.7 and NVIDIA CUDA library installed in user's environment. The package dependency includes the following:
```
numpy>=1.18
mrcfile>=1.2
starfile>=0.4
cupy>=10
torch>=1.10
```

To install CryoSieve using pip, run the following command:
```
pip install cryosieve
```
Or using conda, run the following command:
```
conda install --channel conda-forge cryosieve
```
We recommand installing CuPy and PyTorch first, since their installation highly depends on user's environment. The PyTorch package should be CUDA-capable.

# Tutorial

## Quickstart: a toy example

To ensure you have correctly installed CryoSieve and understand how to utilize it, we highly recommand running CryoSieve on this [toy example](URL). Please follow the steps below:

1. Download the dataset and unzip it to any directory, e.g., `~/toy/`.
2. Change the working directory to this directory:
```
cd ~/toy/
```
3. Run the following command:
```
cryosieve-core --i CNG.star --o my_CNG_1.star --angpix 1.32 --volume CNG_A.mrc --volume CNG_B.mrc --mask CNG_mask.mrc --retention-ratio 0.8 --frequency 40
```

If you run it correctly, it should produce two star files named `my_CNG_1.star` and `my_CNG_1_sieved.star`, containing information of remained particles and sieved particles, respectively. You may compare them with given `CNG_1.star` and `CNG_1_sieved.star`, and they are expected to contain same particles.

## Arguments of CryoSieve core

The program `cryosieve-core` is the core particle sieving process.

```
$ cryosieve-core -h
usage: cryosieve-core [-h] --i I --o O [--directory DIRECTORY] --angpix ANGPIX --volume VOLUME [--mask MASK] [--retention_ratio RETENTION_RATIO] --frequency
                      FREQUENCY [--balance] [--num_gpus NUM_GPUS]

CryoSieve core.

options:
  -h, --help            show this help message and exit
  --i I                 input star file path.
  --o O                 output star file path.
  --directory DIRECTORY
                        directory of particles, empty (current directory) by default.
  --angpix ANGPIX       pixelsize in Angstrom.
  --volume VOLUME       list of volume file paths.
  --mask MASK           mask file path.
  --retention_ratio RETENTION_RATIO
                        fraction of retained particles, 0.8 by default.
  --frequency FREQUENCY
                        cut-off highpass frequency.
  --balance             make retained particles in different subsets in same size.
  --num_gpus NUM_GPUS   number of GPUs to execute the cryosieve program, 1 by default.
```

The input of `cryosieve-core` contains:

- Input star file.
- Output star file. `cryosieve-core` will simultaneously produce a `_sieved.star` file at the same directory.
- Directory of particle stacks.
- Pixelsize.
- Reconstructed volume(s). Multiple volumes are accepted. The number of volumes should be equal to the number of classes (see `_rlnRandomSubset`) in star file.
- (Optional) Mask file.
- The ratio of sieved particles.
- (Optional) The highpass cut-off frequency for calculating CryoSieve score.
- (Optional) If `--balance` is given, CryoSieve will make retained particles in different subsets in same size.
- The number of GPUs to be used.

## Run CryoSieve core with multiple GPUs on single machine

When `--num_gpus` is given and larger than 1. The CryoSieve core program will utilize multiple GPUs to accelerate the sieving process. In this case, it uses the [elastic launch](https://pytorch.org/docs/1.10/elastic/run.html?highlight=torchrun) provided by PyTorch to launch multiple processes. Each process will occupy exactly one GPU. As an example, in a machine with 4 GPUs, we can use the following command to run the toy example.
```
cryosieve-core --i CNG.star --o my_CNG_1.star --angpix 1.32 --volume CNG_A.mrc --volume CNG_B.mrc --mask CNG_mask.mrc --retention-ratio 0.8 --frequency 40 --num_gpus 4
```

## Process real world dataset

Here we provide an example of using CryoSieve to process final stack in an experimental dataset.

### Download the dataset

We will use the final stack of [EMPIAR-11233](https://www.ebi.ac.uk/empiar/EMPIAR-11233/) in this tutorial. This dataset includes a final particle stack of TRPM8 bound to calcium collected on a 300 kV FEI Titan Krios microscope. Change your working directory to any directory you like and use the following command to download the final particle stack:
```
wget -nH -m ftp.ebi.ac.uk/empiar/world_availability/11233/data/Final_Particle_Stack/
```
After downloading you would have a folder `XXX/data/Final_Particle_Stack`. This folder contains a star file recording all information about particles and a mrcs file which is the final stack. In addition, we need a mask file. You can use any cryo-EM software you like to do the 3D reconstruction and generate a mask according to the reconstructed volume. Here we also provide the [mask file](URL) we used in our experiments, and you can download it if you don't want to generate a mask by yourself. After obtaining the mask file, put it into this directory.

### Iteratively doing reconstruction and sieving

To get best result on real world dataset, the sieving process usually contains several rounds. In each iteration, we do 3D reconstruction (and maybe postprocessing to obtain the FSC curve and resolution), and then apply CryoSieve to sieve a fraction of particles according to the reconstructed map. The highpass cut-off frequency usually increase by round. Here we provide an automatic program `cryosieve` implementing all above steps in one command. Please follow the steps below:

1. Change the working directory to `XXX/data/Final_Particle_Stack`:
```
cd XXX/data/Final_Particle_Stack
```
1. Currently, our automatic program use Relion to do the 3D reconstruction and postprocessing. Please ensure that `relion_reconstruct` or `relion_reconstruct_mpi` and `relion_postprocess` are available. Then run the following command:
```
cryosieve --reconstruct_software relion_reconstruct --postprocess_software relion_postprocess --i diver2019_pmTRPM8_calcium_Krios_6Feb18_finalParticleStack_EMPIAR_composite.star --o output/ --mask mask.mrc --angpix 1.059 --num_iters 10 --frequency_start 40 --frequency_end 3 --retention_ratio 0.8 --sym C4
```
The whole process may take more than one hour, depending on your computer resources. Several result files will be written into the directory `output/`. For example, `_iter{n}.star` contains particles remained after n-th iteration of sieving, folder `_postprocess_iter{n}` contains the postprocessing result after n-th iteration.

## Arguments of CryoSieve

The program `cryosieve` is an integreted program iteratively calling relion and `cryosieve-core` to do sieving process.

```
$ cryosieve -h
usage: cryosieve [-h] --reconstruct_software RECONSTRUCT_SOFTWARE [--postprocess_software POSTPROCESS_SOFTWARE] --i I --o O --angpix ANGPIX [--sym SYM]
                 [--num_iters NUM_ITERS] [--frequency_start FREQUENCY_START] [--frequency_end FREQUENCY_END] [--retention_ratio RETENTION_RATIO] --mask MASK
                 [--balance] [--num_gpus NUM_GPUS]

CryoSieve: a particle sorting and sieving software for single particle analysis in cryo-EM.

options:
  -h, --help            show this help message and exit
  --reconstruct_software RECONSTRUCT_SOFTWARE
                        command for reconstruction.
  --postprocess_software POSTPROCESS_SOFTWARE
                        command for postprocessing.
  --i I                 input star file path.
  --o O                 output path prefix.
  --angpix ANGPIX       pixelsize in Angstrom.
  --sym SYM             molecular symmetry, c1 by default.
  --num_iters NUM_ITERS
                        number of iterations for applying CryoSieve, 10 by default.
  --frequency_start FREQUENCY_START
                        starting threshold frquency, in Angstrom, 50A by default.
  --frequency_end FREQUENCY_END
                        ending threshold frquency, in Angstrom, 3A by default.
  --retention_ratio RETENTION_RATIO
                        fraction of retained particles in each iteration, 0.8 by default.
  --mask MASK           mask file path.
  --balance             make remaining particles in different subsets in same size.
  --num_gpus NUM_GPUS   number of gpus to execute CryoSieve core program, 1 by default.
```

There are several useful remarks:

- CryoSieve will use RECONSTRUCT_SOFTWARE as the prefix of reconstruction command. It allows you to use `--reconstruct_software "mpirun -n 5 relion_reconstruct_mpi"` to accelerate reconstruction step by multi-processing.
- If POSTPROCESS_SOFTWARE is not given, CryoSieve will skip the postprocessing step. Notice that postprocessing is not necessary for the sieving procedure.
- Since `relion_reconstruct` use current directory as its default working directory, user should ensure that `relion_reconstruct` can correctly access the particles.
