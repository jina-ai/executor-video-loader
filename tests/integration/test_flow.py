__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

import subprocess

import pytest
from jina import Document, DocumentArray, Flow

from video_loader import VideoLoader


def test_integration():
    da = DocumentArray(
        [Document(id='2c2OmN49cj8.mp4', uri='tests/toy_data/2c2OmN49cj8.mp4')]
    )
    with Flow().add(uses=VideoLoader) as flow:
        resp = flow.post(on='/index', inputs=da, return_results=True)

    assert len(resp[0].docs) == 1
    for doc in resp[0].docs:
        assert len(doc.chunks) == 16
        for image_chunk in filter(lambda x: x.modality == 'image', doc.chunks):
            assert len(image_chunk.blob.shape) == 3

        for audio_chunk in filter(lambda x: x.modality == 'audio', doc.chunks):
            assert audio_chunk.blob is not None
