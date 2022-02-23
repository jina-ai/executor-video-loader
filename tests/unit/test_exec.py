__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from pathlib import Path

import numpy as np
import pytest
import webvtt
from jina import Executor
from docarray import Document, DocumentArray

from video_loader import VideoLoader


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex._frame_fps == 1


def test_no_documents(videoloader: VideoLoader):
    docs = DocumentArray()
    videoloader.extract(docs=docs, parameters={})
    assert len(docs) == 0  # SUCCESS


def test_docs_no_uris(videoloader: VideoLoader):
    docs = DocumentArray([Document()])
    videoloader.extract(docs=docs, parameters={})
    assert len(docs) == 1
    assert len(docs[0].chunks) == 0


@pytest.mark.parametrize('batch_size', [1, 2])
def test_batch_extract(
    expected_frames, expected_audio, video_fn, videoloader: VideoLoader, batch_size: int
):
    docs = DocumentArray([Document(uri=video_fn) for _ in range(batch_size)])
    videoloader.extract(docs=docs, parameters={})
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
    videoloader = VideoLoader(modality_list=modality)
    docs = DocumentArray([Document(uri=video_fn)])
    videoloader.extract(docs=docs, parameters={})
    for doc in docs:
        for c in doc.chunks:
            assert c.modality in modality


def test_extract_with_datauri(
    expected_frames, expected_audio, video_fn, videoloader: VideoLoader
):
    doc = Document(uri=video_fn)
    doc.convert_uri_to_datauri()
    docs = DocumentArray([doc])
    videoloader.extract(docs=docs, parameters={})
    for doc in docs:
        c_img = [c.content for c in doc.chunks if c.modality == 'image']
        assert np.allclose(c_img[:5], expected_frames[:5])

        c_audio = [c.content for c in doc.chunks if c.modality == 'audio']
        assert np.allclose(c_audio, expected_audio)

        c_subtitles = [c.content for c in doc.chunks if c.modality == 'text']
        assert len(c_subtitles) == 30


def test_catch_exception(caplog, video_fn, videoloader):
    docs = DocumentArray([Document(uri='tests/toy_data/dummy.mp4')])  # wrong uri
    videoloader.logger.propagate = True
    videoloader.extract(docs=docs, parameters={})
    assert 'Audio extraction failed' in caplog.text
    assert 'Frame extraction failed' in caplog.text


def test_process_subtitles(srt_path, tmpdir, videoloader):
    subtitles = videoloader._process_subtitles(
        srt_path, tmpdir / 'sub.vtt', tmpdir / 'sub_tmp.srt'
    )
    print(subtitles)
    assert len(subtitles) == 5


def test_convert_srt_to_vtt(srt_path, tmpdir, videoloader):
    vtt_fn = videoloader._convert_srt_to_vtt(
        srt_path, vtt_path=tmpdir / 'sub.vtt', tmp_srt_path=tmpdir / 'sub_tmp.srt'
    )
    assert vtt_fn.exists()
    assert len(webvtt.read(vtt_fn)) == 10


@pytest.mark.parametrize('copy_uri', (True, False))
def test_modality(video_fn, copy_uri):
    videoloader = VideoLoader(copy_uri=copy_uri)
    docs = DocumentArray([Document(uri=video_fn)])
    videoloader.extract(docs=docs, parameters={})
    for doc in docs:
        for c in doc.chunks:
            uri = c.tags.get('video_uri')
            assert copy_uri == (uri is not None)
            if uri:
                assert uri == video_fn


@pytest.mark.parametrize('fps', [0.5, 1.5, 2.0, 3])
def test_float_fps_count(expected_float_fps_frames, video_fn, fps):
    videoloader = VideoLoader(
        modality_list='image', ffmpeg_video_args={'vf': f'fps={fps}'}
    )
    docs = DocumentArray([Document(uri=video_fn)])
    videoloader.extract(docs=docs, parameters={})
    for doc in docs:
        c_img = [c.content for c in doc.chunks if c.modality == 'image']
        assert len(c_img) == len(expected_float_fps_frames)
        assert np.allclose(c_img[:5], expected_float_fps_frames[:5])
