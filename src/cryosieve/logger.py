import json
import logging
import os
import re
import sys
import time
from pathlib import Path


PROGRESS_SCHEMA = 'coco.job-progress-event.v1'
LOGGER_NAME = 'CryoSieve'


class _UtcFormatter(logging.Formatter):
    converter = time.gmtime

    def formatTime(self, record, datefmt=None):
        value = time.strftime('%Y-%m-%dT%H:%M:%S', self.converter(record.created))
        return f'{value}.{int(record.msecs):03d}Z'


def _slug(value):
    value = re.sub(r'[^A-Za-z0-9._-]+', '-', str(value or 'cryosieve')).strip('-._')
    return value.lower() or 'cryosieve'


def _log_paths(default_name, source):
    log_dir = os.environ.get('COCO_LOG_DIR')
    if log_dir:
        name = _slug(source or os.environ.get('COCO_LOG_SOURCE') or 'cryosieve')
        path = Path(log_dir).expanduser().resolve() / f'{name}.log'
    else:
        configured = os.environ.get('COCO_CRYOSIEVE_LOG') or os.environ.get('COCO_JOB_LOG')
        if configured:
            base = Path(configured).expanduser().resolve()
        else:
            job_dir = os.environ.get('COCO_JOB_DIR')
            base = (Path(job_dir).expanduser().resolve() if job_dir else Path.cwd()) / default_name
        path = base
    return path, path.with_name(f'{path.stem}.err{path.suffix}')


def configure_logging(default_name='cryosieve.log', source=None):
    target = logging.getLogger(LOGGER_NAME)
    for handler in list(target.handlers):
        target.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    target.setLevel(logging.INFO)
    target.propagate = False
    formatter = _UtcFormatter('[%(asctime)s][%(levelname)s] %(message)s')
    try:
        all_path, error_path = _log_paths(default_name, source)
        all_path.parent.mkdir(parents=True, exist_ok=True)
        all_handler = logging.FileHandler(str(all_path), mode='a', encoding='utf-8')
        all_handler.setLevel(logging.INFO)
        all_handler.setFormatter(formatter)
        target.addHandler(all_handler)
        error_handler = logging.FileHandler(str(error_path), mode='a', encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        target.addHandler(error_handler)
    except Exception:
        fallback = logging.StreamHandler()
        fallback.setLevel(logging.INFO)
        fallback.setFormatter(formatter)
        target.addHandler(fallback)

    def uncaught(exc_type, exc, traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, traceback)
            return
        target.error('Uncaught exception', exc_info=(exc_type, exc, traceback))

    sys.excepthook = uncaught
    return target


def emit_progress(source, phase, completed, total, unit, metadata=None, checkpoint=None):
    path_value = os.environ.get('COCO_PROGRESS_PATH')
    if not path_value:
        return False
    try:
        index = int(os.environ['COCO_PROGRESS_STEP_INDEX'])
        step_total = int(os.environ['COCO_PROGRESS_STEP_TOTAL'])
        work_total = float(total)
        work_completed = max(0.0, min(float(completed), work_total))
        if index < 1 or step_total < index or work_total <= 0:
            return False
    except (KeyError, TypeError, ValueError):
        return False

    def number(value):
        return int(value) if value.is_integer() else value

    event = {
        'schema': PROGRESS_SCHEMA,
        'event': 'UPDATE',
        'epochMillis': int(time.time() * 1000),
        'source': _slug(source or os.environ.get('COCO_PROGRESS_SOURCE')),
        'rank': 0,
        'step': {
            'stageId': os.environ.get('COCO_PROGRESS_STAGE_ID', 'stage'),
            'stepKey': os.environ.get('COCO_PROGRESS_STEP_KEY', 'step'),
            'index': index,
            'total': step_total,
        },
        'phase': str(phase or 'running'),
        'work': {'unit': str(unit or 'item'), 'completed': number(work_completed), 'total': number(work_total)},
    }
    if metadata:
        event['metadata'] = dict(metadata)
    if checkpoint:
        event['checkpoint'] = dict(checkpoint)
    payload = (json.dumps(event, separators=(',', ':'), ensure_ascii=False, default=str) + '\n').encode('utf-8')
    path = Path(path_value).expanduser().resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        return True
    except OSError:
        return False


logger = configure_logging()
