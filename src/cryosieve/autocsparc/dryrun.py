from ..logger import logger


class DryRunClient:
    def __init__(self):
        self.job_count = 0

    def call(self, method, *args, **kwargs):
        logger.debug(f'Dry run: skip CryoSPARC JSON-RPC call {method}')

    def __getattr__(self, method):
        def dry_run_method(*args, **kwargs):
            return self.call(method, *args, **kwargs)
        return dry_run_method

    def make_job(self, *args, **kwargs):
        self.job_count += 1
        return f'_J{self.job_count}'
