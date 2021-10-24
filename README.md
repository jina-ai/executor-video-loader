# VideoLoader

The `VideoLoader` [executor](https://docs.jina.ai/fundamentals/executor/) helps in loading the video components into Jina's [`Document`](https://docs.jina.ai/fundamentals/document/) type.
It extracts the image frames, audio, and subtitle from the video using the [`ffmpeg-python`](https://github.com/kkroening/ffmpeg-python).

The extracted image frames, audio, and subtitle are stored as `chunks` with the following attributes,

Image frame chunks have the `modality` of `image`, the audio chunks have the `modality` of `audio`, and the subtitle chunks have the `modality` of `text`. During parsing of subtitles, duplicated captions are merged to form unique captions in a heuristic way and are returned as single chunk with their respective starting and ending time in `tags`. 

| data | stored in | `modality` | `location` | `tags` | 
| --- | --- | --- | --- | --- |
| image frames | `blob` (dtype=`uint8`) | `image` | the index of the frame | `{'timestampe': 0.5}`, the timestamp of the frame in seconds. `{'video_uri': 'video.mp4'}`, the uri of the video |
| audio | `blob` (dtype=`float32`) | `audio` | N/A | `{'sample_rate': 140000}`, the sample rate of the audio. `{'video_uri': 'video.mp4'}`, the uri of the video.  |
| subtitle | `text` (dtype=`str`) | `text` | the index of the subtitle in the video | `{'beg_in_seconds': 0.5}`, the beginning of a caption in seconds, <br /> `{'end_in_seconds': 0.6}`, the end of a caption in seconds. `{'video_uri': 'video.mp4'}`, the uri of the video. |