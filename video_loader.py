__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import re
import io
import string
import tempfile
import urllib.request
from copy import deepcopy
import random
from typing import Dict, Iterable, Optional

import ffmpeg
import librosa
import numpy as np
from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from jina.types.document import _is_datauri
from pathlib import Path

DEFAULT_FPS = 1
DEFAULT_AUDIO_BIT_RATE = 160000
DEFAULT_AUDIO_CHANNELS = 2
DEFAULT_AUDIO_SAMPLING_RATE = 44100  # Hz


class VideoLoader(Executor):
    """
    An executor to extract the image frames, audio from videos with `ffmpeg`.
    """

    def __init__(
        self,
        modality_list: Iterable[str] = ('image', 'audio'),
        ffmpeg_video_args: Optional[Dict] = None,
        ffmpeg_audio_args: Optional[Dict] = None,
        librosa_load_args: Optional[Dict] = None,
        **kwargs,
    ):
        """
        :param modality_list: the data from different modalities to be extracted. By default,
            `modality_list=('image', 'audio')`, both image frames and audio track are extracted.
        :param ffmpeg_video_args: the arguments to `ffmpeg` for extracting frames. By default, `format='rawvideo'`,
            `pix_fmt='rgb24`, `frame_pts=True`, `vsync=0`, `vf=[FPS]`, where the frame per second(FPS)=1. The width and
            the height of the extracted frames are the same as the original video. To reset width=960 and height=540,
            use `ffmpeg_video_args={'s': '960x540'`}.
        :param ffmpeg_audio_args: the arguments to `ffmpeg` for extracting audios. By default, the bit rate of the audio
             `ab=160000`, the number of channels `ac=2`, the sampling rate `ar=44100`
        :param librosa_load_args: the arguments to `librosa.load()` for converting audio data into `blob`. By default,
            the sampling rate (`sr`) is the same as in `ffmpeg_audio_args['ar']`, the flag for converting to mono
            (`mono`) is `True` when `ffmpeg_audio_args['ac'] > 1`
        """
        super().__init__(**kwargs)
        self._modality = modality_list
        self._ffmpeg_video_args = ffmpeg_video_args or {}
        self._ffmpeg_video_args.setdefault('format', 'rawvideo')
        self._ffmpeg_video_args.setdefault('pix_fmt', 'rgb24')
        self._ffmpeg_video_args.setdefault('frame_pts', True)
        self._ffmpeg_video_args.setdefault('vsync', 0)
        self._ffmpeg_video_args.setdefault('vf', f'fps={DEFAULT_FPS}')
        fps = re.findall('.*fps=(\d+).*', self._ffmpeg_video_args['vf'])
        if len(fps) > 0:
            self._frame_fps = int(fps[0])

        self._ffmpeg_audio_args = ffmpeg_audio_args or {}
        self._ffmpeg_audio_args.setdefault('format', 'wav')
        self._ffmpeg_audio_args.setdefault('ab', DEFAULT_AUDIO_BIT_RATE)
        self._ffmpeg_audio_args.setdefault('ac', DEFAULT_AUDIO_CHANNELS)
        self._ffmpeg_audio_args.setdefault('ar', DEFAULT_AUDIO_SAMPLING_RATE)

        self._librosa_load_args = librosa_load_args or {}
        self._librosa_load_args.setdefault(
            'sr', self._ffmpeg_audio_args['ar'])
        self._librosa_load_args.setdefault(
            'mono', self._ffmpeg_audio_args['ac'] > 1)
        self.logger = JinaLogger(
            getattr(self.metas, 'name', self.__class__.__name__)
        ).logger

    @requests
    def extract(self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs):
        """
        Load the video from the Document.uri, extract frames and audio. The extracted data are stored in chunks.

        :param docs: the input Documents with either the video file name or data URI in the `uri` field
        :param parameters: A dictionary that contains parameters to control
         extractions and overrides default values.
        Possible values are `ffmpeg_audio_args`, `ffmpeg_video_args`, `librosa_load_args`. Check out more description in the `__init__()`.
        For example, `parameters={'ffmpeg_video_args': {'s': '512x320'}`.
        """
        if docs is None:
            return
        for doc in docs:
            self.logger.info(f'received {doc.id}')

            if doc.uri == '':
                self.logger.error(f'No uri passed for the Document: {doc.id}')
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                source_fn = (
                    self._save_uri_to_tmp_file(doc.uri, tmpdir)
                    if _is_datauri(doc.uri)
                    else doc.uri
                )

                # extract all the frames video
                if 'image' in self._modality:
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
                if 'audio' in self._modality:
                    ffmpeg_audio_args = deepcopy(self._ffmpeg_audio_args)
                    ffmpeg_audio_args.update(parameters.get('ffmpeg_audio_args', {}))
                    librosa_load_args = deepcopy(self._librosa_load_args)
                    librosa_load_args.update(parameters.get('librosa_load_args', {}))
                    audio, sr = self._convert_video_uri_to_audio(
                        source_fn,
                        doc.uri,
                        ffmpeg_audio_args,
                        librosa_load_args)
                    if audio is None:
                        continue
                    chunk = Document(modality='audio')
                    chunk.blob, chunk.tags['sample_rate'] = audio, sr
                    doc.chunks.append(chunk)

    def _convert_video_uri_to_frames(self, source_fn, uri, ffmpeg_args):
        # get width and height
        video_frames = []
        try:
            video = ffmpeg.probe(source_fn)['streams'][0]
            w, h = ffmpeg_args.get('s', f'{video["width"]}x{video["height"]}').split('x')
            w = int(w)
            h = int(h)
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

        return video_frames

    def _convert_video_uri_to_audio(self, source_fn, uri, ffmpeg_args, librosa_args):
        data = None
        sample_rate = None
        try:
            out, _ = (
                ffmpeg.input(source_fn)
                .output('pipe:', **ffmpeg_args)
                .run(capture_stdout=True, quiet=True)
            )
            data, sample_rate = librosa.load(
                io.BytesIO(out), **librosa_args)
        except ffmpeg.Error as e:
            self.logger.error(f'Audio extraction failed with ffmpeg, uri: {uri}, {e.stderr}')
        except librosa.LibrosaError as e:
            self.logger.error(f'Array conversion failed with librosa, uri: {uri}, {e}')
        finally:
            return data, sample_rate

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
