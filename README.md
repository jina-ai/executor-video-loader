# VideoLoader

The `VideoLoader` [executor](https://docs.jina.ai/fundamentals/executor/) helps in loading the video components into Jina's [`Document`](https://docs.jina.ai/fundamentals/document/) type.
It extracts the image frames, audio, and the subtitle(if any) from the video using the [`ffmpeg-python`](https://github.com/kkroening/ffmpeg-python).
The extracted image frames, audio, and the subtitle is then added as `chunks` to the original `Document`


Subtitle chunks have the `modality` of `text`, image frame chunks have the `modality` of `image`, and the audio chunks have the `modality` of `audio`.