
increase volume

```
ffmpeg -i mc2.wav -filter:a "volume=2.0" mc2.1.wav
```

convert to mp3

```
ffmpeg -i mc2.1.wav -b:a 192K mc2.1.mp3
ffmpeg -i mc3.wav -b:a 192K mc3.mp3

```