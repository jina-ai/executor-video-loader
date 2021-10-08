__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import io
import os
import random
import string
import tempfile
import urllib.request
from copy import deepcopy
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
    An executor to extract the image frames, audio, and the subtitle
    from videos with `ffmpeg`
    """

    def __init__(
        self,
        max_num_frames: int = 50,
        fps=DEFAULT_FPS,
        width: int = 960,
        height: int = 540,
        ab: int = 160000,
        ac: int = 2,
        ar: int = 44100,
        **kwargs,
    ):
        """
        :param max_num_frames: maximum number of image frames
         to be extracted from video
        :param fps: number of frames extracted per second
        :param width: width of the frames extracted
        :param height: height of the frames extracted
        :param ab: Set the audio bitrate
        :param ac: Set the number of audio channels
        :param ar: Set the audio sampling frequency
        :param kwargs: the **kwargs for Executor
        """
        super().__init__(**kwargs)
        self._exec_args = {
            'max_num_frames': max_num_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'ab': ab,
            'ac': ac,
            'ar': ar,
        }
        self.logger = JinaLogger(
            getattr(self.metas, 'name', self.__class__.__name__)
        ).logger

    @requests
    def extract(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """
        Load the video from the Document.uri, extract frames, audio, subtitle
        and save it into chunks.

        :param docs: the input Documents with either the video file name
         or URL in the `uri` field
        :param parameters: A dictionary that contains parameters to control
         extractions and overrides default values.
        Possible values are `width`, `height`, 'max_num_frames',
         'fps', 'ab', 'ac, 'ar'.
        For example, `parameters={'width': 512, 'height': 320}`.
        """
        exec_args = deepcopy(self._exec_args)
        exec_args.update(parameters)
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
                        source_fn,
                        doc.uri,
                        exec_args['width'],
                        exec_args['height'],
                        exec_args['fps'],
                    )

                    # add frames as chunks to the Document, modality='image'
                    for idx, frame_fn in enumerate(frame_fn_list):
                        self.logger.debug(f'frame: {idx}')
                        chunk = Document(modality='image')
                        chunk.blob = np.array(frame_fn).astype('uint8')
                        timestamp = idx
                        chunk.location.append(np.uint32(timestamp))
                        doc.chunks.append(chunk)

                    # add audio as chunks to the Document, modality='audio'
                    audio, sr = self._convert_video_uri_to_audio(
                        source_fn,
                        doc.uri,
                        exec_args['ab'],
                        exec_args['ac'],
                        exec_args['ar'],
                    )
                    if audio is not None:
                        chunk = Document(modality='audio')
                        chunk.blob, chunk.tags['sample_rate'] = audio, sr
                        doc.chunks.append(chunk)

    def _convert_video_uri_to_frames(self, source_fn, uri, w, h, fps):
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
                    vf=f'fps={fps}',
                )
                .run(capture_stdout=True, quiet=True)
            )
            video_frames = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
        except ffmpeg.Error as e:
            self.logger.error(f'Frame extraction failed, {uri}, {e.stderr}')
            raise ValueError(f'{uri}: No such file or directory or URL') from e

        return video_frames

    def _convert_video_uri_to_audio(self, source_fn, uri, ab, ac, ar):
        data = None
        sample_rate = None
        try:
            out, _ = (
                ffmpeg.input(source_fn)
                .output('pipe:', format='wav', ab=ab, ac=ac, ar=ar)
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
