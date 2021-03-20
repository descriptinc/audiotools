"""
These are optional utilities included in nussl that allow one to embed an AudioSignal
as a playable object in a Jupyter notebook, or to play audio from
the terminal.
"""
from copy import deepcopy
import subprocess
from tempfile import NamedTemporaryFile
from .util import _close_temp_files

def _check_imports(): # pragma: no cover
    try:
        import ffmpy
    except:
        ffmpy = False

    try:
        import IPython
    except:
        raise ImportError('IPython must be installed in order to use this function!')
    return ffmpy, IPython

class PlayMixin:
    def embed(self, ext='.mp3', display=True, batch_idx=0):
        """
        Write a numpy array to a temporary mp3 file using ffmpy, then embeds the mp3
        into the notebook.

        Args:
            audio_signal (AudioSignal): AudioSignal object containing the data.
            ext (str): What extension to use when embedding. '.mp3' is more lightweight 
            leading to smaller notebook sizes. Defaults to '.mp3'.
            display (bool): Whether or not to display the object immediately, or to return
            the html object for display later by the end user. Defaults to True.

        Example:
            >>> import nussl
            >>> audio_file = nussl.efz_utils.download_audio_file('schoolboy_fascination_excerpt.wav')
            >>> audio_signal = nussl.AudioSignal(audio_file)
            >>> audio_signal.embed_audio()

        This will show a little audio player where you can play the audio inline in 
        the notebook.      
        """
        ext = f'.{ext}' if not ext.startswith('.') else ext
        ffmpy, IPython = _check_imports()
        sr = self.sample_rate
        tmpfiles = []

        with _close_temp_files(tmpfiles):
            tmp_wav = NamedTemporaryFile(
                mode='w+', suffix='.wav', delete=False)
            tmpfiles.append(tmp_wav)
            self.write(tmp_wav.name, batch_idx=batch_idx)
            if ext != '.wav' and ffmpy:
                tmp_converted = NamedTemporaryFile(
                    mode='w+', suffix=ext, delete=False)
                tmpfiles.append(tmp_wav)
                ff = ffmpy.FFmpeg(
                    inputs={tmp_wav.name: None},
                    outputs={tmp_converted.name: '-write_xing 0 -codec:a libmp3lame -b:a 128k -y'})
                ff.run()
            else:
                tmp_converted = tmp_wav

            audio_element = IPython.display.Audio(data=tmp_converted.name, rate=sr)
            if display:
                IPython.display.display(audio_element)
        return audio_element

    def play(self, batch_idx=0):
        """
        Plays an audio signal if ffplay from the ffmpeg suite of tools is installed.
        Otherwise, will fail. The audio signal is written to a temporary file
        and then played with ffplay.
        
        Args:
            audio_signal (AudioSignal): AudioSignal object to be played.
        """
        tmpfiles = []
        with _close_temp_files(tmpfiles):
            tmp_wav = NamedTemporaryFile(suffix='.wav', delete=False)
            tmpfiles.append(tmp_wav)
            self.write(tmp_wav.name, batch_idx=batch_idx)
            print(self)
            subprocess.call([
                "ffplay", "-nodisp", "-autoexit", "-hide_banner", 
                "-loglevel", "error", tmp_wav.name
            ])
        return self