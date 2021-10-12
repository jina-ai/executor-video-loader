# VideoLoader

The `VideoLoader` [executor](https://docs.jina.ai/fundamentals/executor/) helps in loading the video components into Jina's [`Document`](https://docs.jina.ai/fundamentals/document/) type.
It extracts the image frames and audio from the video using the [`ffmpeg-python`](https://github.com/kkroening/ffmpeg-python).

The extracted image frames and audio are stored as `chunks` with the following attributes,

Image frame chunks have the `modality` of `image`, and the audio chunks have the `modality` of `audio`.

| data | stored in | `modality` | `location` | `tags` | 
| --- | --- | --- | --- | --- |
| image frames | `blob` (dtype=`uint8`) | `image` | the index of the frame | `{'timestampe': 0.5}`, the timestamp of the frame in seconds |
| audio | `blob` (dtype=`float32`) | `audio` | N/A | `{'sample_rate': 140000}`, the sample rate of the audio |