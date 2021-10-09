__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import io
import tempfile
import urllib.request
from copy import deepcopy
from typing import Dict, Optional

import ffmpeg
import librosa
import numpy as np
from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from jina.types.document import _is_datauri

DEFAULT_FPS = 1
DEFAULT_FRAME_WIDTH = 960
DEFAULT_FRAME_HEIGHT = 540
DEFAULT_AUDIO_BIT_RATE = 160000
DEFAULT_AUDIO_CHANNELS = 2
DEFAULT_AUDIO_SAMPLING_FREQUENCY = 44100


class VideoLoader(Executor):
    """
    An executor to extract the image frames, audio, and the subtitle
    from videos with `ffmpeg`
    """

    def __init__(
        self,
        ffmpeg_video_args: Optional[Dict] = None,
        ffmpeg_audio_args: Optional[Dict] = None,
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
        self._ffmpeg_video_args = ffmpeg_video_args or {}
        self._ffmpeg_video_args.setdefault('format', 'rawvideo')
        self._ffmpeg_video_args.setdefault('pix_fmt', 'rgb24')
        self._ffmpeg_video_args.setdefault('frame_pts', True)
        self._ffmpeg_video_args.setdefault('s', f'{DEFAULT_FRAME_WIDTH}x{DEFAULT_FRAME_HEIGHT}')
        self._ffmpeg_video_args.setdefault('vsync', 0)
        self._ffmpeg_video_args.setdefault('vf', DEFAULT_FPS)
        w, h = self._ffmpeg_video_args['s'].split('x')
        self._frame_width = int(w)
        self._frame_height = int(h)
        self._frame_fps = self._ffmpeg_video_args['vf']

        self._ffmpeg_audio_args = ffmpeg_audio_args or {}
        self._ffmpeg_audio_args.setdefault('format', 'wav')
        self._ffmpeg_audio_args.setdefault('ab', DEFAULT_AUDIO_BIT_RATE)
        self._ffmpeg_audio_args.setdefault('ac', DEFAULT_AUDIO_CHANNELS)
        self._ffmpeg_audio_args.setdefault('ar', DEFAULT_AUDIO_SAMPLING_FREQUENCY)
        self.logger = JinaLogger(
            getattr(self.metas, 'name', self.__class__.__name__)
        ).logger

    @requests
    def extract(self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs):
        """
        Load the video from the Document.uri, extract frames, audio, subtitle
        and save it into chunks.

        :param docs: the input Documents with either the video file name
         or URL in the `uri` field
        :param parameters: A dictionary that contains parameters to control
         extractions and overrides default values.
        Possible values are `ffmpeg_audio_args`, `ffmpeg_video_args`.
        For example, `parameters={'ffmpeg_video_args': {'s': '512x320'}`.
        """
        if docs is None:
            return
        for doc in docs:
            self.logger.info(f'received {doc.id}')

            if doc.uri == '':
                self.logger.error(f'No uri passed for the Document: {doc.id}')
                continue

            with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp_f:
                source_fn = doc.uri
                if _is_datauri(doc.uri):
                    self._save_uri_to_tmp_file(doc.uri, tmp_f)
                    source_fn = tmp_f.name

                # extract all the frames video
                ffmpeg_video_args = deepcopy(self._ffmpeg_video_args)
                ffmpeg_video_args.update(parameters.get('ffmpeg_video_args', {}))
                frame_blobs = self._convert_video_uri_to_frames(
                    source_fn,
                    doc.uri,
                    ffmpeg_video_args)
                for idx, frame_blob in enumerate(frame_blobs):
                    self.logger.debug(f'frame: {idx}')
                    chunk = Document(modality='image')
                    chunk.blob = np.array(frame_blob).astype('uint8')
                    chunk.location.append(np.uint32(idx))
                    chunk.tags['timestamp'] = idx / self._frame_fps
                    doc.chunks.append(chunk)

                # add audio as chunks to the Document, modality='audio'
                ffmpeg_audio_args = deepcopy(self._ffmpeg_audio_args)
                ffmpeg_audio_args.update(parameters.get('ffmpeg_audio_args', {}))
                audio, sr = self._convert_video_uri_to_audio(
                    source_fn,
                    doc.uri,
                    ffmpeg_audio_args)
                if audio is not None:
                    chunk = Document(modality='audio')
                    chunk.blob, chunk.tags['sample_rate'] = audio, sr
                    doc.chunks.append(chunk)

    def _convert_video_uri_to_frames(self, source_fn, uri, ffmpeg_args):
        w, h = ffmpeg_args.get('s', f'{self._frame_width}x{self._frame_height}').split('x')
        w = int(w)
        h = int(h)
        video_frames = []
        try:
            out, _ = (
                ffmpeg.input(source_fn)
                .output(
                    'pipe:',
                    **ffmpeg_args
                )
                .run(capture_stdout=True, quiet=True)
            )
            video_frames = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
        except ffmpeg.Error as e:
            self.logger.error(f'Frame extraction failed, {uri}, {e.stderr}')
            raise ValueError(f'{uri}: No such file or directory or URL') from e

        return video_frames

    def _convert_video_uri_to_audio(self, source_fn, uri, ffmpeg_args):
        data = None
        sample_rate = None
        try:
            out, _ = (
                ffmpeg.input(source_fn)
                .output('pipe:', **ffmpeg_args)
                .run(capture_stdout=True, quiet=True)
            )
            data, sample_rate = librosa.load(io.BytesIO(out))
        except ffmpeg.Error as e:
            self.logger.error(f'Audio extraction failed, {uri}, {e.stderr}')
            raise ValueError(f'{uri}: No such file or directory or URL') from e

        return data, sample_rate

    def _convert_video_uri_to_srt(self, source_fn):
        pass

    def _save_uri_to_tmp_file(self, uri, tmp_fn):
        req = urllib.request.Request(uri, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as fp:
            buffer = fp.read()
            binary_fn = io.BytesIO(buffer)
            tmp_fn.write(binary_fn.read())
