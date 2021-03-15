from contextlib import contextmanager
import os
from functools import wraps

def _get_value(other):
    from . import AudioSignal
    if isinstance(other, AudioSignal):
        return other.audio_data
    return other

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
    except: # pragma: no cover
        _close()
        raise
    _close()