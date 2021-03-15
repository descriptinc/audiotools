from contextlib import contextmanager
import os
from functools import wraps

def numpy_safe(func):
    """
    Decorator that converts the first argument to a numpy-based
    AudioSignal so it can be safely used by the function.

    Args:
        func (function): Any function of the signature func(audio_signal, ...).
    """
    def decorator(func):
        @wraps(func)
        def numpy_safe_func(audio_signal, *args, **kwargs):
            audio_signal = audio_signal.to('cpu').numpy()
            return func(audio_signal, *args, **kwargs)
        return numpy_safe_func
    return decorator

@contextmanager
def _close_temp_files(tmpfiles):
    """
    Utility function for creating a context and closing all temporary files
    once the context is exited. For correct functionality, all temporary file
    handles created inside the context must be appended to the ```tmpfiles```
    list.

    This function is taken wholesale from Scaper.

    Args:
        tmpfiles (list): List of temporary file handles
    """
    def _close():
        for t in tmpfiles:
            try:
                t.close()
                os.unlink(t.name)
            except:
                pass
    try:
        yield
    except:
        _close()
        raise
    _close()