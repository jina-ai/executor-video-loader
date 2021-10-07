__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

import os
from pathlib import Path

import librosa
import numpy as np
import pytest
from PIL import Image
from jina import Document, DocumentArray, Executor

from video_loader import VideoLoader

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(cur_dir, '..', 'toy_data'))


@pytest.fixture(scope="module")
def encoder() -> VideoLoader:
    return VideoLoader()


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.fps == 1
    assert ex.max_num_frames == 50


def test_no_docucments(encoder: VideoLoader):
    docs = DocumentArray()
    encoder.extract(docs=docs, parameters={})
    assert len(docs) == 0  # SUCCESS


def test_docs_no_uris(encoder: VideoLoader):
    docs = DocumentArray([Document()])

    with pytest.raises(ValueError, match='No uri'):
        encoder.extract(docs=docs, parameters={})

    assert len(docs) == 1
    assert len(docs[0].chunks) == 0


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_encode(encoder: VideoLoader, batch_size: int):
    expected_frames = [
        np.array(Image.open(os.path.join(data_dir, "2c2OmN49cj8-{:04n}.png".format(i)))) for i in range(15)
    ]
    expected_audio, sample_rate = librosa.load(os.path.join(data_dir, 'audio.wav'))
    test_file = os.path.join(data_dir, '2c2OmN49cj8.mp4')
    docs = DocumentArray([Document(uri=test_file) for _ in range(batch_size)])
    encoder.extract(docs=docs, parameters={})
    for doc in docs:
        c_img = [c.content for c in doc.chunks if c.modality == 'image']
        assert np.allclose(c_img, expected_frames)

        c_audio = [c.content for c in doc.chunks if c.modality == 'audio']
        assert np.allclose(c_audio, expected_audio)
