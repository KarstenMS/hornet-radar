from sources import FrameSource

def get_frame_source_from_args(args) -> FrameSource:
    if args.images:
        return FrameSource.IMAGE
    if args.videos:
        return FrameSource.VIDEO
    return FrameSource.CAMERA
