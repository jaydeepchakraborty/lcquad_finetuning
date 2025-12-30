from lcquad_finetuning.util.util_lib import *

class LCQuadLogger:

    def __init__(self, config):
        self.config = config

    def setup_logging(self):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        if root_logger.handlers:
            return  # prevent duplicate handlers

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            fmt="%(levelname)-5s | "
                "%(filename)s:%(lineno)d | "
                "%(funcName)s() | "
                "%(message)s"
        )

        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    def get_logger(self, name="lcquad_finetuning"):
        self.setup_logging()
        return logging.getLogger(name)