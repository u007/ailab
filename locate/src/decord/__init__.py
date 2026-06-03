"""Stub decord module for macOS ARM64 where no wheels exist.
Only video reading is unavailable; image inference works fine."""

class VideoReader:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("decord is not available on this platform — video loading is disabled")

def bridge(*args, **kwargs):
    raise RuntimeError("decord is not available on this platform")
