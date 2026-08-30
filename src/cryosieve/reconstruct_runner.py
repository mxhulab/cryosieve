from __future__ import annotations

import hashlib
import os
import shlex
from pathlib import Path
from .utility import run_commands


CRYOSIEVE_LOG_NAME = 'cryosieve.log'
DFR_LOG_NAME = 'copra_spa_3d_reconstruction_dfr.log'
COCO_DFR_PARENT_DISTRIBUTED_ENV = 'COCO_DFR_USE_PARENT_DISTRIBUTED'
COCO_DFR_PARENT_STORE_PREFIX_ENV = 'COCO_DFR_PARENT_STORE_PREFIX'

DISTRIBUTED_ENV_KEYS = (
    'LOCAL_RANK',
    'RANK',
    'WORLD_SIZE',
    'GROUP_RANK',
    'ROLE_RANK',
    'ROLE_WORLD_SIZE',
    'LOCAL_WORLD_SIZE',
    'MASTER_ADDR',
    'MASTER_PORT',
    'TORCHELASTIC_RUN_ID',
    'TORCHELASTIC_RESTART_COUNT',
    'TORCHELASTIC_MAX_RESTARTS',
)

PROGRESS_ENV_KEYS = (
    'COCO_PROGRESS_PATH',
    'COCO_PROGRESS_SOURCE',
    'COCO_PROGRESS_STAGE_ID',
    'COCO_PROGRESS_STEP_KEY',
    'COCO_PROGRESS_STEP_INDEX',
    'COCO_PROGRESS_STEP_TOTAL',
)

COCO_DFR_RECONSTRUCT_MODULES = {
    'copra_spa_3d_reconstruction_dfr.compat.cryosieve_reconstruct',
    'coconut_spa_3d_reconstruction_dfr.compat.cryosieve_reconstruct',
}


def job_log_path(output_dir, log_name):
    job_dir = os.environ.get('COCO_JOB_DIR')
    base = Path(job_dir).absolute() if job_dir else Path(output_dir)
    return base / log_name


def child_env(output_dir, job_log_name=None, cryosieve_log=False):
    env = os.environ.copy()
    for key in DISTRIBUTED_ENV_KEYS:
        env.pop(key, None)
    for key in PROGRESS_ENV_KEYS:
        env.pop(key, None)
    if job_log_name is None:
        env.pop('COCO_JOB_LOG', None)
    else:
        env['COCO_JOB_LOG'] = str(job_log_path(output_dir, job_log_name))
    if cryosieve_log:
        env['COCO_CRYOSIEVE_LOG'] = str(job_log_path(output_dir, CRYOSIEVE_LOG_NAME))
    return env


def is_coco_dfr_reconstruct(command):
    try:
        tokens = shlex.split(command or '')
    except ValueError:
        return False
    for index, token in enumerate(tokens[:-2]):
        executable = Path(token).name
        if executable not in {'python', 'python3'}:
            continue
        if tokens[index + 1] == '-m' and tokens[index + 2] in COCO_DFR_RECONSTRUCT_MODULES:
            return True
    return False


def _parent_store_prefix(output_dir, half_map, iteration=None, env=None):
    environment = os.environ if env is None else env
    half_map_value = int(half_map)
    if half_map_value not in (1, 2):
        raise ValueError('half_map must be 1 or 2')
    iteration_value = int(iteration or 0)
    if iteration_value < 0:
        raise ValueError('iteration must be non-negative')
    run_id = environment.get('TORCHELASTIC_RUN_ID') or environment.get('COCO_RDZV_ID') or ''
    restart_count = environment.get('TORCHELASTIC_RESTART_COUNT') or '0'
    identity = '\0'.join((
        'coco-dfr-parent-store-v1',
        str(run_id),
        str(restart_count),
        str(Path(output_dir).expanduser().resolve()),
        str(iteration_value),
        str(half_map_value),
    ))
    digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()
    return f'coco/dfr/v1/{digest}'


def _parent_store_endpoint(env):
    master_addr = str(env.get('MASTER_ADDR') or '').strip()
    master_port = str(env.get('MASTER_PORT') or '').strip()
    if not master_addr:
        raise ValueError('MASTER_ADDR is required for parent-store DFR')
    try:
        port = int(master_port)
    except ValueError as exc:
        raise ValueError('MASTER_PORT must be an integer between 1 and 65535 for parent-store DFR') from exc
    if not 1 <= port <= 65535:
        raise ValueError('MASTER_PORT must be an integer between 1 and 65535 for parent-store DFR')
    return master_addr, str(port)


def distributed_child_env(output_dir, job_log_name, half_map, iteration=None):
    env = os.environ.copy()
    for key in PROGRESS_ENV_KEYS:
        env.pop(key, None)
    if job_log_name is None:
        env.pop('COCO_JOB_LOG', None)
    else:
        env['COCO_JOB_LOG'] = str(job_log_path(output_dir, job_log_name))
    master_addr, master_port = _parent_store_endpoint(env)
    env[COCO_DFR_PARENT_DISTRIBUTED_ENV] = '1'
    env[COCO_DFR_PARENT_STORE_PREFIX_ENV] = _parent_store_prefix(output_dir, half_map, iteration, env)
    env['MASTER_ADDR'] = master_addr
    env['MASTER_PORT'] = master_port
    return env


def release_parent_gpu_memory_caches():
    try:
        import torch
        cuda = getattr(torch, 'cuda', None)
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()
            ipc_collect = getattr(cuda, 'ipc_collect', None)
            if ipc_collect is not None:
                ipc_collect()
    except Exception:
        pass
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _as_command_list(commands):
    if isinstance(commands, str):
        return [commands]
    return list(commands or [])


def run_reconstruct_commands(commands, jobname, cwd, output_dir, ctx, reconstruct_software, barrier_func, iteration=None):
    command_list = _as_command_list(commands)
    if ctx is not None and ctx.distributed and is_coco_dfr_reconstruct(reconstruct_software):
        if ctx.is_main:
            from .logger import logger
            logger.info('Run CoCo DFR reconstruction commands with parent distributed allocation')
        for index, command in enumerate(command_list, start=1):
            release_parent_gpu_memory_caches()
            run_commands(
                command,
                f'{jobname} half-map {index}',
                cwd=cwd,
                env=distributed_child_env(output_dir, DFR_LOG_NAME, index, iteration),
            )
            barrier_func(ctx)
        return

    if ctx is None or ctx.is_main:
        run_commands(command_list, jobname, cwd=cwd, env=child_env(output_dir, DFR_LOG_NAME))
    if barrier_func is not None:
        barrier_func(ctx)
