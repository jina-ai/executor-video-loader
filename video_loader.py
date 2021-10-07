__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import io
import os
import random
import string
import tempfile
import urllib.request
from typing import Dict

import ffmpeg
import librosa
import numpy as np
from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from jina.types.document import _is_datauri

DEFAULT_FPS = 1


class VideoLoader(Executor):
    """
    An executor to extract the image frames, audio, and the subtitle from videos with `ffmpeg`
    """

    def __init__(
        self,
        max_num_frames: int = 50,
        fps=DEFAULT_FPS,
        frame_width: int = 960,
        frame_height: int = 540,
        **kwargs,
    ):
        """
        :param max_num_frames: maximum number of images frames to be extracted from the video
        :param fps: number of frames extracted per second
        :param frame_width: width of the frames extracted
        :param frame_height: height of the frames extracted
        :param kwargs: the **kwargs for Executor
        """
        super().__init__(**kwargs)
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.width = frame_width
        self.height = frame_height
        self.logger = JinaLogger(
            getattr(self.metas, 'name', self.__class__.__name__)
        ).logger

    @requests
    def extract(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """
        Load the video from the Document.uri, extract frames, audio, subtitle
        and save it into chunks.

        :param docs: the input Documents with either the video file name or URL in the `uri` field
        :param parameters: dictionary with additional request parameters. Possible
            values are `frame_width` and the `frame_height`. For example,
               `parameters={'frame_width': 512, 'frame_height': 320}`.
        """
        w = parameters.get('frame_width', self.width)
        h = parameters.get('frame_height', self.height)
        for doc in docs:
            self.logger.info(f'received {doc.id}')

            if doc.uri == '':
                self.logger.error(f'No uri passed for the Document: {doc.id}')
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    source_fn = (
                        self._save_uri_to_tmp_file(doc.uri, tmpdir)
                        if _is_datauri(doc.uri)
                        else doc.uri
                    )

                    # extract all the frames video
                    frame_fn_list = self._convert_video_uri_to_frames(
                        source_fn, doc.uri, w, h
                    )

                    # add frames as chunks to the Document, with modality='image'
                    for idx, frame_fn in enumerate(frame_fn_list):
                        self.logger.debug(f'frame: {idx}')
                        chunk = Document(modality='image')
                        chunk.blob = np.array(frame_fn).astype('uint8')
                        timestamp = idx
                        chunk.location.append(np.uint32(timestamp))
                        doc.chunks.append(chunk)

                    # add audio as chunks too to the same Document but with modality='audio'
                    audio, sample_rate = self._convert_video_uri_to_audio(
                        source_fn, doc.uri
                    )
                    if audio is not None:
                        chunk = Document(modality='audio')
                        chunk.blob, chunk.tags['sample_rate'] = audio, sample_rate
                        doc.chunks.append(chunk)

    def _convert_video_uri_to_frames(self, source_fn, uri, w, h):
        video_frames = []
        try:
            out, _ = (
                ffmpeg.input(source_fn)
                .output(
                    'pipe:',
                    format='rawvideo',
                    pix_fmt='rgb24',
                    frame_pts=True,
                    s=f'{w}x{h}',
                    vsync=0,
                    vf=f'fps={self.fps}',
                )
                .run(capture_stdout=True, quiet=True)
            )
            video_frames = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
        except ffmpeg.Error as e:
            self.logger.error(f'Frame extraction failed, {uri}, {e.stderr}')
            raise ValueError(f'{uri}: No such file or directory or URL') from e

        return video_frames

    def _convert_video_uri_to_audio(self, source_fn, uri):
        data = None
        sample_rate = None
        try:
            out, _ = (
                ffmpeg.input(source_fn)
                .output('pipe:', format='wav', ab=160000, ac=2, ar=44100)
                .run(capture_stdout=True, quiet=True)
            )
            data, sample_rate = librosa.load(io.BytesIO(out))
        except ffmpeg.Error as e:
            self.logger.error(f'Audio extraction failed, {uri}, {e.stderr}')
            raise ValueError(f'{uri}: No such file or directory or URL') from e

        return data, sample_rate

    def _convert_video_uri_to_srt(self, source_fn):
        pass

    def _save_uri_to_tmp_file(self, uri, tmpdir):
        req = urllib.request.Request(uri, headers={'User-Agent': 'Mozilla/5.0'})
        tmp_fn = os.path.join(
            tmpdir,
            ''.join([random.choice(string.ascii_lowercase) for i in range(10)])
            + '.mp4',
        )
        with urllib.request.urlopen(req) as fp:
            buffer = fp.read()
            binary_fn = io.BytesIO(buffer)
            with open(tmp_fn, 'wb') as f:
                f.write(binary_fn.read())
        return tmp_fn
