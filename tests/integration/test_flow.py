__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

import os

import librosa
import numpy as np
from PIL import Image
from jina import Document, DocumentArray, Flow

from video_loader import VideoLoader

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(cur_dir, '..', 'toy_data'))


def test_integration():
    encoder = VideoLoader()
    expected_frames = [
        np.array(Image.open(os.path.join(data_dir, "2c2OmN49cj8-{:04n}.png".format(i))))
        for i in range(15)
    ]
    expected_audio, sample_rate = librosa.load(
        os.path.join(data_dir, 'audio.wav'), **encoder._librosa_load_args)
    da = DocumentArray(
        [Document(id='2c2OmN49cj8.mp4', uri='tests/toy_data/2c2OmN49cj8.mp4')]
    )
    with Flow().add(uses=VideoLoader) as flow:
        resp = flow.post(on='/index', inputs=da, return_results=True)

    assert len(resp[0].docs) == 1
    for doc in resp[0].docs:
        c_img = [c.content for c in doc.chunks if c.modality == 'image']
        assert np.allclose(c_img, expected_frames)

        c_audio = [c.content for c in doc.chunks if c.modality == 'audio']
        assert np.allclose(c_audio, expected_audio)
