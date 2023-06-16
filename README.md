# Overview

CryoSieve is a particle sorting and sieving software for single particle analysis in cryo-EM.

<!-- The paper can be referred as XXX. -->

# Installation

CryoSieve is an open source software developed in Python programming language and released as a Python package.

We recommend using Python version >= 3.7 and NVIDIA CUDA library installed in user's environment. The package dependency includes the following:
```
numpy>=1.18
mrcfile>=1.2
torch>=1.10
cupy>=10
```
To install CryoSieve, run the following command:
```
pip install CryoSieve
```

# Tutorial

## Quickstart: a toy example

To ensure you have correctly installed CryoSieve and understand how to utilize it, we highly recommand running CryoSieve on this [toy example](URL). Please follow the steps below:

1. Download the dataset and unzip it to any directory, e.g., `~/toy/`.
1. Change the working directory to this directory:
```
cd ~/toy/
```
1. Run the following command:
```
cryosieve-core --i CNG.star --o my_CNG_1.star --angpix 1.32 --volume CNG_A.mrc --volume CNG_B.mrc --mask CNG_mask.mrc --retention-ratio 0.8 --frequency 40
```

If you run it correctly, it should produce two star files named `my_CNG_1.star` and `my_CNG_1_sieved.star`, containing information of remained particles and sieved particles, respectively. You may compare them with given `CNG_1.star` and `CNG_1_sieved.star`, and they are expected to be the same.

## Arguments of CryoSieve

```
$ CryoSieve -h
usage: CryoSieve [-h] -i INPUT -d DIRECTORY -p PIXELSIZE -v VOLUME [-m MASK] -r RATIO [-f FREQUENCY] -o OUTPUT

CryoSieve: a particle sorting and sieving software for single particle analysis in cryo-EM.

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input star file path.
  -d DIRECTORY, --directory DIRECTORY
                        directory of particles.
  -p PIXELSIZE, --pixelsize PIXELSIZE
                        particle pixelsize.
  -v VOLUME, --volume VOLUME
                        list of volume file paths.
  -m MASK, --mask MASK  mask file path.
  -r RATIO, --ratio RATIO
                        fraction of sieved particles.
  -f FREQUENCY, --frequency FREQUENCY
                        cut-off high-pass frequency, 40 Angstrom by default.
  -o OUTPUT, --output OUTPUT
                        output star file path.
```

The input of CryoSieve contains:

- Input star file.
- Directory of particle stacks.
- Pixelsize.
- Reconstructed volume(s). Multiple copies are accepted. The number of volumes should be equal to the number of classes (see `_rlnRandomSubset`) in star file.
- (Optional) Mask file.
- The ratio of sieved particles.
- (Optional) The highpass cut-off frequency for calculating CryoSieve score, 40$\AA$ by default.
- Output star file. CryoSieve will simultaneously produce a `_sieved.star` file at the same directory.

## Run CryoSieve with multiple GPUs on single machine

We use the [elastic launch](https://pytorch.org/docs/1.10/elastic/run.html?highlight=torchrun) provided by PyTorch. For example, you can launch CryoSieve in a cluster node with 4 GPUs with the following command:

```
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m CryoSieve -i "CNG.star" -d "./" -p 1.32 -v "CNG_A.mrc" -v "CNG_B.mrc" -m "CNG_mask.mrc" -r 0.2 -f 40 -o "my_CNG_1.star"
```

Here the option `--nproc_per_node=4` means 4 processes will be launched and run simultaneously. Each process will utilize one GPU according to its local rank.

## Process real world dataset

Here we provide an example of using CryoSieve to process final stack in an experimental dataset.

### Download the dataset

We will use the final stack of [EMPIAR-11233](https://www.ebi.ac.uk/empiar/EMPIAR-11233/) in this tutorial. This dataset includes a final particle stack of TRPM8 bound to calcium collected on a 300 kV FEI Titan Krios microscope. Change your working directory to any directory you like and use the following command to download the final particle stack:
```
wget -nH -m ftp.ebi.ac.uk/empiar/world_availability/11233/data/Final_Particle_Stack/
```
After downloading you would have a folder `XXX/data/Final_Particle_Stack`. The folder contains a star file recording all information about particles and final stacks.

Currently our program does not support STAR format after Relion 3.1 (see Relion's document on [Conventions](https://relion.readthedocs.io/en/release-4.0/Reference/Conventions.html)). Hence we convert this star file into Relion 2.0 version by [UCSF pyem](https://github.com/asarnow/pyem). Here we provide the converted [star file](URL).

In addition, we need a mask file. You can use any cryo-EM software you like to do the 3D reconstruction and generate a mask according to the reconstructed volume. Here we also provide the [mask file](URL) we used in our experiments, and you can download it if you don't want to generate a mask by yourself.

### Iteratively doing reconstruction and sieving

The sieving process usually contains several rounds. In each iteration, we first use Relion to do 3D reconstruction (and postprocessing for obtaining the FSC curve and resolution), and then apply CryoSieve to sieve a fraction of particles. Here we show how to process one iteration.

First make a working directory for the whole process:
```
mkdir XXX/data/CryoSieve_output/
```
Then copy the star file and mask file to the working directory for the following usage:
```
cp THE_STAR_FILE XXX/data/CryoSieve_output/TRPM8_0.star
cp THE_MASK_FILE XXX/data/CryoSieve_output/TRPM8_mask.mrc
```
Here the suffix `_0` means this star file is the result after 0 round of sieving.

Next, use Relion to do 3D reconstruction. This step can be replaced by any other cryo-EM 3D reconstruction software you like:
```
cd XXX/data/Final_Particle_Stack/
relion_reconstruct --i XXX/data/CryoSieve_output/TRPM8_0.star --o XXX/data/CryoSieve_output/TRPM8_0A.mrc --sym C4 --angpix 1.059 --ctf true --subset 1
relion_reconstruct --i XXX/data/CryoSieve_output/TRPM8_0.star --o XXX/data/CryoSieve_output/TRPM8_0B.mrc --sym C4 --angpix 1.059 --ctf true --subset 2
cd XXX/data/CryoSieve_output/
```
Here the suffix `_0A` and `_0B` means the reconstructed volumes of round 0 and hemi-sphere A and B, respectively.

If you are like to see the FSC curve and resolution at this moment, you can also use Relion postprocess. But the results will not contribute anything to the next sieving step.
```
relion_postprocess --i TRPM8_0A.mrc --i2 TRPM8_0B.mrc --angpix 1.059 --mask TRPM8_mask.mrc --auto_bfac true --o TRPM8_postprocess_0
```

Now we are ready to use CryoSieve for sieving the particle stack:
```
CryoSieve -i TRPM8_0.star -d XXX/data/Final_Particle_Stack/ -p 1.059 -v TRPM8_0A.mrc -v TRPM8_0B.mrc -m TRPM8_mask.mrc -r 0.2 -f 40 -o TRPM8_1.star
```
Here the parameter `-f 40` means the cut-off frequency of highpass operator in CryoSieve score are set to be 40 Angstrom.

If you run above command successfully and obtained `TRPM8_1.star` and `TRPM8_1_sieved.star`, congratulations! The first file should contains 80% high-scored particles of `TRPM8_0.star`.

You can also continue this process and go for second iteration:
```
cd XXX/data/Final_Particle_Stack/
relion_reconstruct --i XXX/data/CryoSieve_output/TRPM8_1.star --o XXX/data/CryoSieve_output/TRPM8_1A.mrc --sym C4 --angpix 1.059 --ctf true --subset 1
relion_reconstruct --i XXX/data/CryoSieve_output/TRPM8_1.star --o XXX/data/CryoSieve_output/TRPM8_1B.mrc --sym C4 --angpix 1.059 --ctf true --subset 2
cd XXX/data/CryoSieve_output/
relion_postprocess --i TRPM8_1A.mrc --i2 TRPM8_1B.mrc --angpix 1.059 --mask TRPM8_mask.mrc --auto_bfac true --o TRPM8_postprocess_1
CryoSieve -i TRPM8_1.star -d XXX/data/Final_Particle_Stack/ -p 1.059 -v TRPM8_1A.mrc -v TRPM8_1B.mrc -m TRPM8_mask.mrc -r 0.2 -f 16.875 -o TRPM8_2.star
```
In this iteration, the cut-off frequency is set to be 16.875 Angstrom. In our experiments, we usually set this frequency to be monotonously increasing, see our paper for details.

Now you are encouraged to do more iterations by yourself. You may also determine parameters in each round according to you own dataset and expertise.

### An automatic shell script

The procedure above can be easily down atomatically by a shell script. Here we provide a demo:

```[language=shell]
protein=TRPM8
data_dir="XXX/data/Final_Particle_Stack/"
working_dir="XXX/data/CryoSieve_output/"
symmetry="C4"
pixel_size=1.059
ratio=0.2
frequency=(40 16.875 10.693 7.826 6.171 5.094 4.337 3.776 3.344 3)

for i in {0..9}
do
    cd ${data_dir}
    relion_reconstruct --i ${working_dir}${protein}_${i}.star --o ${working_dir}${protein}_${i}A.mrc --sym ${symmetry} --angpix ${pixel_size} --ctf true --subset 1
    relion_reconstruct --i ${working_dir}${protein}_${i}.star --o ${working_dir}${protein}_${i}B.mrc --sym ${symmetry} --angpix ${pixel_size} --ctf true --subset 2
    cd ${working_dir}
    relion_postprocess --i ${protein}_${i}A.mrc --i2 ${protein}_${i}B.mrc --angpix ${pixel_size} --mask ${protein}_mask.mrc --auto_bfac true --o ${protein}_postprocess_${i}
    CryoSieve -i ${protein}_${i}.star -d ${data_dir} -p ${pixel_size} -v ${protein}_${i}A.mrc -v ${protein}_${i}B.mrc -m ${protein}_mask.mrc -r ${ratio} -f ${frequency[$i]} -o ${protein}_$(($i+1)).star
done
```
