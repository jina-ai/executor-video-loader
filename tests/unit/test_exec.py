__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from pathlib import Path

import numpy as np
import pytest
from jina import Document, DocumentArray, Executor

from video_loader import VideoLoader


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex._frame_fps == 1


def test_no_documents(videoLoader: VideoLoader):
    docs = DocumentArray()
    videoLoader.extract(docs=docs)
    assert len(docs) == 0  # SUCCESS


def test_docs_no_uris(videoLoader: VideoLoader):
    docs = DocumentArray([Document()])
    videoLoader.extract(docs=docs)
    assert len(docs) == 1
    assert len(docs[0].chunks) == 0


@pytest.mark.parametrize('batch_size', [1, 2])
def test_batch_extract(expected_frames, expected_audio, video_fn, videoLoader: VideoLoader, batch_size: int):
    docs = DocumentArray([Document(uri=video_fn) for _ in range(batch_size)])
    videoLoader.extract(docs=docs)
    for doc in docs:
        c_img = [c.content for c in doc.chunks if c.modality == 'image']
        assert np.allclose(c_img[:5], expected_frames[:5])

        c_audio = [c.content for c in doc.chunks if c.modality == 'audio']
        assert np.allclose(c_audio, expected_audio)

        c_subtitles = [c.content for c in doc.chunks if c.modality == 'text']
        assert len(c_subtitles) == 30


@pytest.mark.parametrize(
    'modality', [('image',), ('audio',), ('text',), ('image', 'audio', 'text')]
)
def test_modality(video_fn, modality):
    videoLoader = VideoLoader(modality_list=modality)
    docs = DocumentArray([Document(uri=video_fn)])
    videoLoader.extract(docs=docs)
    for doc in docs:
        for c in doc.chunks:
            assert c.modality in modality


def test_extract_with_datauri(expected_frames, expected_audio, video_fn, videoLoader: VideoLoader):
    doc = Document(uri=video_fn)
    doc.convert_uri_to_datauri()
    docs = DocumentArray([doc])
    videoLoader.extract(docs=docs)
    for doc in docs:
        c_img = [c.content for c in doc.chunks if c.modality == 'image']
        assert np.allclose(c_img[:5], expected_frames[:5])

        c_audio = [c.content for c in doc.chunks if c.modality == 'audio']
        assert np.allclose(c_audio, expected_audio)

        c_subtitles = [c.content for c in doc.chunks if c.modality == 'text']
        assert len(c_subtitles) == 30


def test_catch_exception(caplog, video_fn, videoLoader):
    docs = DocumentArray([Document(uri='tests/toy_data/dummy.mp4')])  # wrong uri
    videoLoader.logger.propagate = True
    videoLoader.extract(docs=docs)
    assert 'Audio extraction failed' in caplog.text
    assert 'Frame extraction failed' in caplog.text
