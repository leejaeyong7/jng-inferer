def _noop():
    return None


def build_progress_callback(show_progress):
    """
    Returns:
        callback_ptr: callable for preprocessing callback_ptr API, or None
        close_fn: cleanup callable
    """
    if not show_progress:
        return None, _noop

    try:
        from tqdm import tqdm
    except Exception:
        def callback(desc, frame_id, total_frames):
            frame_num = frame_id + 1
            if frame_num == 1 or frame_num == total_frames or frame_num % 50 == 0:
                print(f"{desc} {frame_num}/{total_frames}")

        return (lambda: callback), _noop

    progress = tqdm(total=0, dynamic_ncols=True)

    def callback(desc, frame_id, total_frames):
        if progress.total != total_frames:
            progress.total = total_frames
        progress.set_description(desc)
        progress.n = frame_id + 1
        progress.refresh()

    def close_fn():
        progress.close()

    return (lambda: callback), close_fn
