import signal
from functools import wraps


class TimeoutError(Exception): pass


# IMPORTANT: this is not thread-safe
def timeout(seconds, error_message='Function call timed out'):
    def _handle_timeout(signum, frame):
        raise TimeoutError(error_message)

    def decorated(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorated
