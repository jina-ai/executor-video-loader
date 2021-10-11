__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

import numpy as np
from jina import Document, DocumentArray, Flow

from video_loader import VideoLoader


def test_integration(expected_frames, expected_audio):
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
