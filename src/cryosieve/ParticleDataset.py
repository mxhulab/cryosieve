import os
import starfile
import numpy as np
import pandas as pd
from copy import copy
from torch.utils.data import Dataset
from typing import Optional
from .utility import mrcread

class ParticleDataset(Dataset):
    '''
    Dataset class for particles.

    The parameters of particles, like ctfs, will be loaded when
    the object is created. However, the data of particles will not
    be loaded until the __getitem__ method is called.
    '''

    def __init__(self, star_path : str, data_dir : str = '', pixel_size : Optional[float] = None):
        if not os.path.exists(star_path):
            raise FileNotFoundError(f'{star_path} does not exist.')
        star = starfile.read(star_path, always_dict = True)
        self.data_dir = data_dir
        self.pixel_size = pixel_size

        # <Relion 3.1
        # For supporting starfile>=0.5 in the future.
        if len(star) == 1 and (0 in star or '' in star or 'images' in star):
            self.version = 2
            self.optics = None
            self.particles = star[0] if 0 in star else star[''] if '' in star else star['images']

            # Check keys.
            for key in ['rlnOriginX', 'rlnOriginY', 'rlnAngleRot', 'rlnAngleTilt',
                        'rlnAnglePsi', 'rlnVoltage', 'rlnDefocusU', 'rlnDefocusV',
                        'rlnDefocusAngle', 'rlnSphericalAberration',
                        'rlnAmplitudeContrast', 'rlnImageName']:
                if key not in self.particles:
                    raise ValueError(f'Key {key} missed in star file {star_path}.')
            if pixel_size is None:
                raise ValueError('Need pixelsize (--angpix) for star file before RELION version 3.1.')

        # >=Relion 3.1
        elif len(star) == 2 and ('optics' in star and 'particles' in star):
            self.version = 3
            self.optics = star['optics']
            self.particles = star['particles']

            # Check keys.
            for key in ['rlnVoltage', 'rlnImagePixelSize', 'rlnSphericalAberration',
                        'rlnAmplitudeContrast', 'rlnOpticsGroup']:
                if key not in self.optics:
                    raise ValueError(f'Key {key} missed in block data_optics in star file {star_path}.')

            for key in ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnAngleRot', 'rlnAngleTilt',
                        'rlnAnglePsi', 'rlnDefocusU', 'rlnDefocusV', 'rlnDefocusAngle',
                        'rlnOpticsGroup', 'rlnImageName']:
                if key not in self.particles:
                    raise ValueError(f'Key {key} missed in block data_particles in star file {star_path}.')

        else:
            raise ValueError('Invalid particle star file.')

        self._parse_paras()

    def _parse_paras(self):
        '''
        Parsing parameters from self.optics, self.particles.
        '''
        self.paras = np.empty((len(self.particles), 14), dtype = np.float64)

        # Handling RELION 3.1 format by merging tables.
        if self.version == 2:
            particles = self.particles
        else:
            try:
                particles = pd.merge(self.optics, self.particles, left_on = 'rlnOpticsGroup', right_on = 'rlnOpticsGroup', validate = 'one_to_many')
            except pd.errors.MergeError:
                raise ValueError('There are multiple optic groups with same index. Check the star file.')
            if len(particles) != len(self.particles):
                raise ValueError('There are particles with no corresponding optic group. Check the star file.')

        self.paras[:,  0] = particles['rlnOriginX'] if self.version == 2 else particles['rlnOriginXAngst'] / particles['rlnImagePixelSize']
        self.paras[:,  1] = particles['rlnOriginY'] if self.version == 2 else particles['rlnOriginYAngst'] / particles['rlnImagePixelSize']
        psi   = np.radians(particles['rlnAngleRot'])
        theta = np.radians(particles['rlnAngleTilt'])
        phi   = np.radians(particles['rlnAnglePsi'])
        self.paras[:,  2] =  np.cos((phi + psi) / 2) * np.cos(theta / 2)
        self.paras[:,  3] = -np.sin((phi - psi) / 2) * np.sin(theta / 2)
        self.paras[:,  4] = -np.cos((phi - psi) / 2) * np.sin(theta / 2)
        self.paras[:,  5] = -np.sin((phi + psi) / 2) * np.cos(theta / 2)
        self.paras[:,  6] = particles['rlnVoltage'] * 1e3
        self.paras[:,  7] = particles['rlnDefocusU']
        self.paras[:,  8] = particles['rlnDefocusV']
        self.paras[:,  9] = np.radians(particles['rlnDefocusAngle'])
        self.paras[:, 10] = particles['rlnSphericalAberration'] * 1e7
        self.paras[:, 11] = particles['rlnAmplitudeContrast']
        self.paras[:, 12] = np.radians(particles['rlnPhaseShift']) if 'rlnPhaseShift' in particles else 0.
        self.paras[:, 13] = self.pixel_size if self.version == 2 else particles['rlnImagePixelSize']

    def __len__(self) -> int:
        return len(self.paras)

    def __getitem__(self, i : int):
        assert 0 <= i < len(self.paras)
        slc, name, *_ = self.particles.iloc[i]['rlnImageName'].split('@')
        return mrcread(self.data_dir + name, int(slc) - 1), self.paras[i]

    def save(self, output_path : str):
        data_dict = {'images' : self.particles} if self.version == 2 else \
                    {'optics' : self.optics, 'particles' : self.particles}
        starfile.write(data_dict, output_path, overwrite = True)

    def n_random_subset(self) -> int:
        return 1 if 'rlnRandomSubset' not in self.particles else \
            len(self.particles['rlnRandomSubset'].value_counts())

    def get_random_subset(self, i : int):
        if 'rlnRandomSubset' not in self.particles:
            raise ValueError('Key rlnRandomSubset missed in star file.')
        return self.subset(self.particles['rlnRandomSubset'] == i)

    def subset(self, mask):
        sub = copy(self)
        sub.paras = self.paras[mask]
        sub.particles = self.particles[mask]
        return sub

    def balance(self):
        if 'rlnRandomSubset' not in self.particles:
            return

        n = self.particles['rlnRandomSubset'].value_counts().min()
        self.particles = self.particles.groupby('rlnRandomSubset').sample(n = n)
        self.particles.sort_index(inplace = True)
        self._parse_paras()
