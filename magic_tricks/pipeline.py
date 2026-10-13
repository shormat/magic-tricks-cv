"""Multi-threaded processing of one trick segment of a video.

    reader -> detector (GPU, batches of frames) -> N analyze workers -> render (in order) -> writer / preview

Each stage hands a sentinel downstream when it is done, so every thread ends on
its own once the last frame has been written.
"""

import logging
import queue
import threading
from pathlib import Path

import cv2

from .annotation import Annotation
from .detection import DEFAULT_BATCH_SIZE
from .effects import LoopingVideo
from .geometry import MarkerTracker
from .tricks import OUTPUT_FPS, TrickContext, get_trick

log = logging.getLogger(__name__)

FRAME_SIZE = (1920, 1080)
QUEUE_SIZE = 32
_STOP = object()


def to_frame_size(frame):
    if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
        frame = cv2.resize(frame, FRAME_SIZE)
    return frame


def output_path_for(annotation_file, trick_id, video_type):
    group = Path(annotation_file).stem.rsplit("_", 1)[-1]
    return f"output_trick_{trick_id}_{video_type}_{group}.mp4"


class TrickPipeline:
    def __init__(self, annotation_file, trick_id, video_type="Static", *, detector, nthreads=4,
                 prefix_video_name=False, preview=False, overlay_path="video.mp4", output_path=None,
                 fps=OUTPUT_FPS, batch_size=DEFAULT_BATCH_SIZE):
        self.trick_id = trick_id
        self.video_type = video_type
        self.spec = get_trick(trick_id, video_type)
        self.detector = detector
        self.batch_size = max(1, batch_size)
        self.n_workers = max(1, nthreads - 1)
        self.preview = preview
        self.fps = fps
        self.output_path = output_path or output_path_for(annotation_file, trick_id, video_type)

        annotation = Annotation(annotation_file, video_type, prefix_video_name)
        video_name = annotation.get_video_name()
        self.cap = cv2.VideoCapture(video_name)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {video_name}")
        self.input_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.start_time = annotation.get_start_time(trick_id)
        if trick_id < annotation.trick_count:
            self.end_time = annotation.get_start_time(trick_id + 1)
        else:
            self.end_time = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.input_fps)

        # The first frame of the video is the clean background plate.
        ok, first_frame = self.cap.read()
        if not ok:
            raise IOError(f"Cannot read video file: {video_name}")
        first_frame = to_frame_size(first_frame)
        self.ctx = TrickContext(
            first_frame=first_frame,
            markers=MarkerTracker(first_frame) if self.spec.uses_markers else None,
            overlay=LoopingVideo(overlay_path),
            fps=fps,
        )
        self.ctx.state = self.spec.init_state(self.ctx)

        self._frames = queue.Queue(QUEUE_SIZE)
        self._detections = queue.Queue(QUEUE_SIZE)
        self._analyzed = queue.Queue(QUEUE_SIZE)
        self._rendered = queue.Queue(QUEUE_SIZE)
        self._abort = threading.Event()
        self._reader_error = None

        log.info("Video: %s", video_name)
        log.info("Trick %d (%s): %s", trick_id, video_type, annotation.get_trick_name(trick_id))
        log.info("Segment: %ds -> %ds (~%d frames)", self.start_time, self.end_time,
                 (self.end_time - self.start_time) * fps)
        if not preview:
            log.info("Output: %s", self.output_path)

    def run(self):
        threads = [threading.Thread(target=self._read, name="reader", daemon=True),
                   threading.Thread(target=self._detect, name="detector", daemon=True),
                   threading.Thread(target=self._render, name="render", daemon=True)]
        threads += [threading.Thread(target=self._analyze, name=f"analyze-{i}", daemon=True)
                    for i in range(self.n_workers)]
        for t in threads:
            t.start()

        try:
            if self.preview:
                self._show()
            else:
                self._write()
        except BaseException:
            self._abort.set()
            raise
        finally:
            for t in threads:
                t.join()
            self.ctx.overlay.release()

        if self._reader_error is not None:
            raise self._reader_error

    # Queue helpers that give up when the pipeline is aborted.

    def _put(self, q, item):
        while not self._abort.is_set():
            try:
                q.put(item, timeout=0.5)
                return
            except queue.Full:
                pass

    def _get(self, q):
        while not self._abort.is_set():
            try:
                return q.get(timeout=0.5)
            except queue.Empty:
                pass
        return _STOP

    # Stages

    def _read(self):
        """Read the trick segment, resampled to self.fps by dropping frames."""
        try:
            start_frame = int(self.start_time * self.input_fps)
            end_frame = int(self.end_time * self.input_fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            next_time, index = 0.0, 0
            while not self._abort.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    break
                position = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if position >= end_frame:
                    break
                if (position - start_frame) / self.input_fps >= next_time:
                    self._put(self._frames, (index, to_frame_size(frame)))
                    next_time += 1.0 / self.fps
                    index += 1
            log.info("Read %d frames", index)
        except Exception as e:
            log.exception("Reader failed")
            self._reader_error = e
        finally:
            self.cap.release()
            self._put(self._frames, _STOP)

    def _next_batch(self):
        """Up to batch_size frames, fewer at the end of the segment, and None once it is over."""
        batch = []
        while len(batch) < self.batch_size:
            item = self._get(self._frames)
            if item is _STOP:
                break
            batch.append(item)
        return batch or None

    def _detect(self):
        while (batch := self._next_batch()) is not None:
            frames = [frame for _, frame in batch]
            try:
                detections = self.detector.detect_batch(frames, self.spec.prompt, self.spec.text_threshold)
            except Exception:
                log.exception("Detection failed on frames %d-%d", batch[0][0], batch[-1][0])
                detections = [[] for _ in batch]
            for (index, frame), dets in zip(batch, detections):
                self._put(self._detections, (index, frame, dets))
            if len(batch) < self.batch_size:  # the reader is done
                break
        for _ in range(self.n_workers):
            self._put(self._detections, _STOP)

    def _analyze(self):
        while (item := self._get(self._detections)) is not _STOP:
            index, frame, detections = item
            try:
                data = self.spec.analyze(frame, detections, self.ctx)
            except Exception:
                log.exception("Analysis failed on frame %d", index)
                data = None
            self._put(self._analyzed, (index, frame, data))
        self._put(self._analyzed, _STOP)

    def _render(self):
        """Render frames in order, since tricks keep state from one frame to the next."""
        pending, next_index, stopped = {}, 0, 0
        while stopped < self.n_workers and not self._abort.is_set():
            item = self._get(self._analyzed)
            if item is _STOP:
                stopped += 1
                continue
            index, frame, data = item
            pending[index] = (frame, data)
            while next_index in pending:
                frame, data = pending.pop(next_index)
                if data is not None:
                    try:
                        frame = self.spec.render(frame, next_index, data, self.ctx)
                    except Exception:
                        log.exception("Rendering failed on frame %d, keeping it unchanged", next_index)
                self._put(self._rendered, frame)
                next_index += 1
        self._put(self._rendered, _STOP)

    def _write(self):
        writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, FRAME_SIZE)
        if not writer.isOpened():
            raise IOError(f"Cannot open video writer for {self.output_path}")
        count = 0
        try:
            while (frame := self._get(self._rendered)) is not _STOP:
                writer.write(frame)
                count += 1
                if count % 100 == 0:
                    log.info("Wrote %d frames", count)
        finally:
            writer.release()
        log.info("Saved %d frames to %s", count, self.output_path)

    def _show(self):
        """Preview in a window (main thread, as required by macOS). Press q to stop."""
        title = f"Trick {self.trick_id} ({self.video_type})"
        try:
            while (frame := self._get(self._rendered)) is not _STOP:
                cv2.imshow(title, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Preview stopped by user")
                    self._abort.set()
        finally:
            cv2.destroyAllWindows()
