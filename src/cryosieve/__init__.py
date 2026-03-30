__version__ = "1.3.0"

import logging
import logging.config
import sys

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s][%(levelname)s] %(message)s',
            'datefmt': r'%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'stderr': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': sys.stderr,
        },
    },
    'loggers': {
        'CryoSieve': {
            'level': 'INFO',
            'handlers': ['stderr'],
            'propagate': False
        }
    }
})
logger = logging.getLogger('CryoSieve')
