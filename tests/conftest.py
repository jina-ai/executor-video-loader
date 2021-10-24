import pytest
from PIL import Image
import librosa
import os
import subprocess
from pathlib import Path

import numpy as np
from video_loader import VideoLoader

data_dir = (Path(__file__).parent / 'toy_data').absolute()

@pytest.fixture(scope='session')
def video_fn():
    return str(data_dir / '1q83w3rehj3.mkv')

@pytest.fixture(scope='function')
def expected_frames(tmp_path, video_fn):
    subprocess.check_call(
        f'ffmpeg -loglevel panic -i {video_fn} -vsync 0 -vf fps=1 -frame_pts true '
        f'{os.path.join(str(tmp_path), f"%d.png")} >/dev/null 2>&1',
        shell=True,
    )
    expected_frames = [np.array(Image.open(os.path.join(str(tmp_path), f'{i}.png'))) for i in range(15)]
    return expected_frames

@pytest.fixture(scope='function')
def videoLoader():
    return VideoLoader()


@pytest.fixture(scope='function')
def expected_audio(tmp_path, video_fn, videoLoader):
    output_fn = os.path.join(str(tmp_path), 'audio.wav')
    subprocess.check_call(
        f'ffmpeg -loglevel panic -i {video_fn} -ab 160000 -ac 2 -ar 44100 {output_fn}'
        f' >/dev/null 2>&1',
        shell=True,
    )
    expected_audio, _ = librosa.load(output_fn, **videoLoader._librosa_load_args)
    return expected_audio


@pytest.fixture(scope='session')
def srt_path():
    return data_dir / 'subs_with_carriage_returns.srt'
