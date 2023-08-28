import logging.config

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-8s %(name)s:%(lineno)d -> %(message)s"
LOGGING_CONF = {
    "formatters": {
        "brief": {
            "format": LOG_FORMAT
        },
        "colored": {
            "()": "coloredlogs.ColoredFormatter",
            "format": LOG_FORMAT
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "colored"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console"],
            "level": "ERROR"
        },
        "production": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False
        },
        "__main__": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False
        }
    },
    "version": 1
}
logging.config.dictConfig(LOGGING_CONF)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
