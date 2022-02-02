import numpy as np
import os
import subprocess
import shlex
import tempfile
import json

class DisplayMixin:
    def specshow(self, batch_idx=0, x_axis="time", y_axis="linear", **kwargs):
        import librosa
        import librosa.display

        log_mag = librosa.amplitude_to_db(self.magnitude.cpu().numpy(), ref=np.max)
        librosa.display.specshow(
            log_mag[batch_idx].mean(axis=0),
            x_axis=x_axis,
            y_axis=y_axis,
            sr=self.sample_rate,
            **kwargs
        )

    def waveplot(self, batch_idx=0, x_axis="time", **kwargs):
        import librosa
        import librosa.display

        audio_data = self.audio_data[batch_idx].mean(dim=0)
        audio_data = audio_data.cpu().numpy()
        librosa.display.waveplot(
            audio_data, x_axis=x_axis, sr=self.sample_rate, **kwargs
        )

    def upload_to_discourse(self, api_username=None, api_key=None, batch_idx=0, discourse_server=None, ext=".wav"):  # pragma: no cover
        if api_username is None:
            api_username = os.environ.get("DISCOURSE_API_USERNAME", None)
        if api_key is None:
            api_key = os.environ.get("DISCOURSE_API_KEY", None)
        if discourse_server is None:
            discourse_server = os.environ.get("DISCOURSE_SERVER", None)

        if discourse_server is None or api_key is None or api_username is None:
            raise RuntimeError("DISCOURSE_API_KEY, DISCOURSE_SERVER, DISCOURSE_API_USERNAME must be set in your environment!")

        with tempfile.NamedTemporaryFile(suffix=ext) as f:
            self.write(f.name, batch_idx=batch_idx)

            command = (
                f"curl -X POST {discourse_server}/uploads.json "
                f"-H 'content-type: multipart/form-data;' "
                f"-H 'Api-Key: {api_key}' "
                f"-H 'Api-Username: {api_username}' "
                f"-F 'type=composer' "
                f"-F 'files[]=@{f.name}' "
            )
            info = json.loads(subprocess.check_output(shlex.split(command)))

            label = self.path_to_input_file
            if label is None:
                label = "unknown"
        
            formatted = f"![{label}|audio]({info['short_path']})"
            return formatted, info
