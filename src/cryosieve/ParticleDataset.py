import os
import starfile
import numpy as np
import pandas as pd
from copy import copy
from pathlib import Path
from os import PathLike
from typing import Optional
from numpy.typing import NDArray
from .utility import mrcread

class ParticleDataset(object):
    '''
    Dataset class for particles.

    The parameters of particles, like ctfs, will be loaded when
    the object is created. However, the data of particles will not
    be loaded until the __getitem__ method is called.
    '''

    def __init__(
        self,
        star_path : str,
        data_dir : Optional[PathLike] = None,
        pixel_size : Optional[float] = None,
        enable_cache : bool = True
    ):
        if not os.path.exists(star_path):
            raise FileNotFoundError(f'{star_path} does not exist')
        star = starfile.read(star_path, always_dict = True)

        if data_dir is not None:
            self.data_dir = Path(data_dir)
            if not self.data_dir.is_dir():
                raise RuntimeError(f'Invalid particle directory: {self.data_dir}')
        else:
            self.data_dir = Path('.')

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
                    raise ValueError(f'Key {key} missed in star file {star_path}')
            if pixel_size is None:
                raise ValueError('Need pixelsize (--angpix) for star file before RELION version 3.1')

        # >=Relion 3.1
        elif len(star) == 2 and ('optics' in star and 'particles' in star):
            self.version = 3
            self.optics = star['optics']
            self.particles = star['particles']

            # Check keys.
            for key in ['rlnVoltage', 'rlnImagePixelSize', 'rlnSphericalAberration',
                        'rlnAmplitudeContrast', 'rlnOpticsGroup']:
                if key not in self.optics:
                    raise ValueError(f'Key {key} missed in block data_optics in star file {star_path}')

            for key in ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnAngleRot', 'rlnAngleTilt',
                        'rlnAnglePsi', 'rlnDefocusU', 'rlnDefocusV', 'rlnDefocusAngle',
                        'rlnOpticsGroup', 'rlnImageName']:
                if key not in self.particles:
                    raise ValueError(f'Key {key} missed in block data_particles in star file {star_path}')

        else:
            raise ValueError('Invalid particle star file')

        self._parse_paras()
        self.cached_mrc_handles = dict() if enable_cache else None

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
                raise ValueError('There are multiple optic groups with same index. Check the star file')
            if len(particles) != len(self.particles):
                raise ValueError('There are particles with no corresponding optic group. Check the star file')

        self.paras[:,  0] = particles['rlnOriginX'] if self.version == 2 else particles['rlnOriginXAngst'] / particles['rlnImagePixelSize']
        self.paras[:,  1] = particles['rlnOriginY'] if self.version == 2 else particles['rlnOriginYAngst'] / particles['rlnImagePixelSize']
        rot  = np.radians(particles['rlnAngleRot'])
        tilt = np.radians(particles['rlnAngleTilt'])
        psi  = np.radians(particles['rlnAnglePsi'])
        self.psi = psi
        self.paras[:,  2] =  np.cos((psi + rot) / 2) * np.cos(tilt / 2)
        self.paras[:,  3] = -np.sin((psi - rot) / 2) * np.sin(tilt / 2)
        self.paras[:,  4] = -np.cos((psi - rot) / 2) * np.sin(tilt / 2)
        self.paras[:,  5] = -np.sin((psi + rot) / 2) * np.cos(tilt / 2)
        self.paras[:,  6] = particles['rlnVoltage'] * 1e3
        self.paras[:,  7] = particles['rlnDefocusU']
        self.paras[:,  8] = particles['rlnDefocusV']
        self.paras[:,  9] = np.radians(particles['rlnDefocusAngle'])
        self.paras[:, 10] = particles['rlnSphericalAberration'] * 1e7
        self.paras[:, 11] = particles['rlnAmplitudeContrast']
        self.paras[:, 12] = np.radians(particles['rlnPhaseShift']) if 'rlnPhaseShift' in particles else 0.
        self.paras[:, 13] = self.pixel_size if self.version == 2 else particles['rlnImagePixelSize']

        split_data = particles['rlnImageName'].str.split('@', n = 2, expand = True)
        self.i_slcs = split_data[0].to_numpy(dtype = np.int32)
        self.names = split_data[1].to_numpy(dtype = np.str_)

    def __len__(self) -> int:
        return len(self.paras)

    def __getitem__(self, i : int):
        assert 0 <= i < len(self.paras)
        i_slc = self.i_slcs[i]
        name = self.names[i]
        mrc_path : Path = self.data_dir / name
        if not mrc_path.is_file():
            raise FileNotFoundError(f'No such particle stack file: "{str(mrc_path)}"')
        return mrcread(mrc_path, i_slc - 1, self.cached_mrc_handles), self.paras[i]

    @property
    def trans(self) -> NDArray[np.float64]:
        return self.paras[:, 0:2]

    @property
    def quats(self) -> NDArray[np.float64]:
        return self.paras[:, 2:6]

    @property
    def ctfs(self) -> NDArray[np.float64]:
        return self.paras[:, 6:14]

    @property
    def psis(self) -> NDArray[np.float64]:
        return self.psi

    @property
    def data_dict(self) -> dict[str, pd.DataFrame]:
        return {'images' : self.particles} if self.version == 2 else \
            {'optics' : self.optics, 'particles' : self.particles}

    def save(self, output_path : str):
        starfile.write(self.data_dict, output_path, overwrite = True)

    def n_random_subset(self) -> int:
        return 1 if 'rlnRandomSubset' not in self.particles else \
            len(self.particles['rlnRandomSubset'].value_counts())

    def get_random_subset(self, i : int):
        if 'rlnRandomSubset' not in self.particles:
            raise ValueError('Key rlnRandomSubset missed in star file')
        return self.subset(self.particles['rlnRandomSubset'] == i)

    def subset(self, mask):
        sub = copy(self)
        sub.particles = self.particles[mask]
        sub.paras = self.paras[mask]
        sub.i_slcs = self.i_slcs[mask]
        sub.names = self.names[mask]
        return sub

    def balance(self):
        if 'rlnRandomSubset' not in self.particles: return
        n = self.particles['rlnRandomSubset'].value_counts().min()
        self.particles = self.particles.groupby('rlnRandomSubset').sample(n = n)
        self.particles.sort_index(inplace = True)
        self._parse_paras()
