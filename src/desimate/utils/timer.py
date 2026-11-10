import logging
import timeit

logger = logging.getLogger(__name__)


def timefunc(func):
    def wrapper(*args, **kwargs):
        logger.debug("Launching %s", func.__qualname__)
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        logger.debug(f"{func.__qualname__} took {end - start:.2f} seconds to run")
        return result

    return wrapper
