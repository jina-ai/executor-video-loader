__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

import numpy as np
from jina import Document, DocumentArray, Flow

from video_loader import VideoLoader


def test_integration(expected_frames, expected_audio):
    da = DocumentArray(
        [Document(id='1bh98dhj3.mkv', uri='tests/toy_data/1q83w3rehj3.mkv')]
    )
    with Flow().add(uses=VideoLoader, uses_requests={'/index': 'extract'}) as flow:
        resp = flow.post(on='/index', inputs=da, return_results=True)

    assert len(resp[0].docs) == 1
    for doc in resp[0].docs:
        c_img = [c.content for c in doc.chunks if c.modality == 'image']
        assert np.allclose(c_img[:5], expected_frames[:5])

        c_audio = [c.content for c in doc.chunks if c.modality == 'audio']
        assert np.allclose(c_audio, expected_audio)

        c_subtitles = [c.content for c in doc.chunks if c.modality == 'text']
        assert len(c_subtitles) == 30
