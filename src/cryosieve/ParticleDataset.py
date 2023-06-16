import os
import numpy as np
from math import radians, cos, sin
from copy import copy
from torch.utils.data import Dataset
from .utility import mrcread

class ParticleDataset(Dataset):
    '''
    Dataset class for particles.

    The parameters of particles, like ctfs, will be loaded when
    the object is created. However, the data of particles will not
    be loaded until the __getitem__ method is called.
    '''

    def __init__(self, star_path : str, data_dir : str, pixel_size : float):
        if not os.path.exists(star_path):
            raise FileNotFoundError(f'{star_path} does not exist.')
        self.star_path = star_path

        # Read and parse the star file.
        header_dict = {}
        data_lines = []
        f = open(star_path, 'r')
        status = 0
        for i, line in enumerate(f):
            # New data block.
            if status == 0 and line.startswith('data_'):
                block_name = line.split()[0][5:]
                if block_name not in ['', 'images']:
                    raise ValueError(f'Invalid block in line {i + 1}.')
                status = 1

            # New loop.
            elif status == 1 and line.startswith('loop_'):
                status = 2

            # Parse labels.
            elif status == 2:
                if line.startswith('_'):
                    key = line.split()[0][1:]
                    if key in header_dict:
                        raise ValueError(f'Multiple key {key} occured.')
                    header_dict[key] = len(header_dict)
                else:
                    status = 3
                    self.start_line = i

            # Parse data entry.
            if status == 3:
                if line.strip() == '':
                    status = 0
                    self.end_line = i
                    break
                else:
                    data_lines.append(line)
        if status == 3:
            self.end_line = i + 1
        elif status != 0:
            raise ValueError('Invalid star file.')
        f.close()

        # Check header.
        for key in ['rlnOriginX', 'rlnOriginY', 'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi',
                    'rlnVoltage', 'rlnDefocusU', 'rlnDefocusV', 'rlnDefocusAngle',
                    'rlnSphericalAberration', 'rlnAmplitudeContrast', 'rlnImageName']:
            if key not in header_dict:
                raise ValueError(f'Key {key} missed in star file.')

        n = len(data_lines)
        self.paras = np.empty((n, 14), dtype = np.float64)
        self.image_ids = np.empty(n, dtype = np.int32)
        self.stack_paths = []
        self.subsets = np.ones(n, dtype = np.int32)
        self.indices = np.arange(n, dtype = np.int32)

        for i, line in enumerate(data_lines):
            info = line.split()

            self.paras[i, 0] = float(info[header_dict['rlnOriginX']])                      # dx
            self.paras[i, 1] = float(info[header_dict['rlnOriginY']])                      # dy

            psi   = radians(float(info[header_dict['rlnAngleRot']]))
            theta = radians(float(info[header_dict['rlnAngleTilt']]))
            phi   = radians(float(info[header_dict['rlnAnglePsi']]))
            self.paras[i, 2] =  cos((phi + psi) / 2) * cos(theta / 2)                       # qw
            self.paras[i, 3] = -sin((phi - psi) / 2) * sin(theta / 2)                       # qx
            self.paras[i, 4] = -cos((phi - psi) / 2) * sin(theta / 2)                       # qy
            self.paras[i, 5] = -sin((phi + psi) / 2) * cos(theta / 2)                       # qz

            self.paras[i, 6]  = float(info[header_dict['rlnVoltage']]) * 1000               # voltage
            self.paras[i, 7]  = float(info[header_dict['rlnDefocusU']])                     # defocusU
            self.paras[i, 8]  = float(info[header_dict['rlnDefocusV']])                     # defocusV
            self.paras[i, 9]  = radians(float(info[header_dict['rlnDefocusAngle']]))        # theta
            self.paras[i, 10] = float(info[header_dict['rlnSphericalAberration']]) * 1e7    # Cs
            self.paras[i, 11] = float(info[header_dict['rlnAmplitudeContrast']])            # amplitudeContrast
            self.paras[i, 12] = radians(float(info[header_dict['rlnPhaseShift']])) \
                                if 'rlnPhaseShift' in header_dict else 0.                   # phaseShift
            self.paras[i, 13] = pixel_size                                                  # pixelSize

            slc, name, *_ = info[header_dict['rlnImageName']].split('@')
            self.image_ids[i] = int(slc) - 1
            self.stack_paths.append(data_dir + name)
            if 'rlnRandomSubset' in header_dict: self.subsets[i] = int(info[header_dict['rlnRandomSubset']])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i : int):
        assert 0 <= i < len(self.indices)
        idx = self.indices[i]
        return mrcread(self.stack_paths[idx], self.image_ids[idx]), self.paras[idx]

    def save(self, output_path : str):
        fin  = open(self.star_path, 'r')
        fout = open(output_path, 'w')

        mask = np.zeros(len(self.paras), dtype = np.bool_)
        mask[self.indices] = True
        for i, line in enumerate(fin):
            if not (self.start_line <= i < self.end_line and not mask[i - self.start_line]):
                fout.write(line)

        fin.close()
        fout.close()

    def reset(self, subset = None):
        self.indices = np.arange(len(self.paras), dtype = np.int32) if subset is None else \
                       np.where(self.subsets == subset)[0]

    def subset(self, indices):
        sub = copy(self)
        sub.indices = self.indices[indices]
        return sub

    def split(self, mask):
        sub1, sub2 = copy(self), copy(self)
        sub1.indices = self.indices[mask]
        sub2.indices = self.indices[~mask]
        return sub1, sub2
