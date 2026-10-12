import cv2
from Annotation import Annotation # For loading annotations json files
import queue

import threading

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from PIL import Image
import numpy as np

from Trick1_dynamic import trick_1_dynamic_part1, trick_1_dynamic_part2
from Trick1_static import trick_1_static
from Trick2_static import trick_2_static_part1, trick_2_static_part2
from Trick2_dynamic import trick_2_dynamic_part1, trick_2_dynamic_part2
from Trick3_static import trick_3_static_part1, trick_3_static_part2
from Trick3_dynamic import trick_3_dynamic_part1, trick_3_dynamic_part2

class MagicTricks():
    def __init__(self, annotation_file, nthreads, video_type="Static", prefix_video_name=False, trick_id=1, debug=False):
        self.annotation_file = annotation_file
        self.number_in_annotation_file = annotation_file.split("_")[-1].split(".")[0]
        self.annotation = Annotation(annotation_file, video_type, prefix_video_name)
        self.trick_id = trick_id
        video_name = self.annotation.get_video_name()
        self.cap = cv2.VideoCapture(video_name)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {video_name}")

        self.start_time = self.annotation.get_start_time(trick_id)

        self.video_type = video_type
        try:
            self.end_time = self.annotation.get_start_time(trick_id+1)
        except IndexError:
            self.end_time = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS))

        self.debug = debug

        self.thread_pool = list()
        self.nthreads = nthreads

        self.frame_queue = queue.Queue(maxsize=100)
        self.gpu_queue = queue.Queue(maxsize=100)
        # Priority queue : (priority, frame_number, frame)
        self.process_queue = queue.Queue(maxsize=100)
        self.writer_queue = queue.PriorityQueue(maxsize=100)

        self.data_video = dict()

        self.writer_stop = threading.Event()
        self.stop_event = threading.Event()
        
        print(f"[INFO] filename : {video_name}")
        print(f"[INFO] trick name : {self.annotation.get_trick_name(trick_id)}")
        print(f"[INFO] start time : {self.start_time} seconds")
        print(f"[INFO] end time : {self.end_time} seconds")
        print(f"[INFO] output file : output_trick_{trick_id}.mp4")
        print(f"[INFO] total frames to process : {(self.end_time - self.start_time) * int(self.cap.get(cv2.CAP_PROP_FPS))}")


    def init_trick_variables(self):
        if self.trick_id == 1 and self.video_type == "Static":
            pass
        elif self.trick_id == 2 and self.video_type == "Static":
            state_ball = 0
            big_can_image = None
            state_can = 0
            state_zombie = 0
            last_detection_ball = 0
            last_detection_can = 0
            last_detection_zombie = 0
            amogus_img = None
            amogus_bbox = None
            amogus_cap2_reader = None
            self.data_video = {
                "state_ball" : state_ball,
                "big_can_image" : big_can_image,
                "state_can" : state_can,
                "state_zombie" : state_zombie,
                "last_detection_ball" : last_detection_ball,
                "last_detection_can" : last_detection_can,
                "last_detection_zombie" : last_detection_zombie,
                "amogus_img" : amogus_img,
                "amogus_bbox" : amogus_bbox,
                "amogus_cap2_reader" : amogus_cap2_reader
            }
        elif self.trick_id == 3 and self.video_type == "Static":
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters()
            aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            first_frame_ids = None
            first_frame_corners = None
            last_roi = None
            self.data_video = {
                "aruco_dict": aruco_dict,
                "aruco_params": aruco_params,
                "aruco_detector": aruco_detector,
                "first_frame_corners" : first_frame_corners,
                "first_frame_ids" : first_frame_ids,
                "last_roi" : last_roi
            }
        elif (self.trick_id == 1 or self.trick_id == 3) and self.video_type == "Dynamic":
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters()
            aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            first_frame_ids = None
            first_frame_corners = None
            last_roi = None
            self.data_video = {
                "aruco_dict": aruco_dict,
                "aruco_params": aruco_params,
                "aruco_detector": aruco_detector,
                "first_frame_corners" : first_frame_corners,
                "first_frame_ids" : first_frame_ids,
                "last_roi" : last_roi
            }
        elif self.trick_id == 2 and self.video_type == "Dynamic":
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters()
            aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            first_frame_ids = None
            first_frame_corners = None
            last_roi = None
            state_ball = 0
            big_can_image = None
            state_can = 0
            state_zombie = 0
            last_detection_ball = 0
            last_detection_can = 0
            last_detection_zombie = 0
            amogus_img = None
            amogus_bbox = None
            amogus_cap2_reader = None
            self.data_video = {
                "aruco_dict": aruco_dict,
                "aruco_params": aruco_params,
                "aruco_detector": aruco_detector,
                "first_frame_corners" : first_frame_corners,
                "first_frame_ids" : first_frame_ids,
                "last_roi" : last_roi,
                "state_ball" : state_ball,
                "big_can_image" : big_can_image,
                "state_can" : state_can,
                "state_zombie" : state_zombie,
                "last_detection_ball" : last_detection_ball,
                "last_detection_can" : last_detection_can,
                "last_detection_zombie" : last_detection_zombie,
                "amogus_img" : amogus_img,
                "amogus_bbox" : amogus_bbox,
                "amogus_cap2_reader" : amogus_cap2_reader
            }

    def run(self):
        self.init_trick_variables()
        # Here we create the threads
        try:
            frame_reader = threading.Thread(target=self.read_frames, )
            frame_reader.start()

            gpu_thread = threading.Thread(target=self.gpu_process, daemon=True)
            gpu_thread.start()
            
            num_first = self.nthreads
            for i in range(self.nthreads-1):
                worker_thread = threading.Thread(target=self.trick_process)
                worker_thread.start()
                self.thread_pool.append(worker_thread)

            process_thread = threading.Thread(target=self.trick_process_next)
            process_thread.start()

            if self.debug:
                writer_thread = threading.Thread(target=self.show_video, args=(self.trick_id,), )
            else:
                writer_thread = threading.Thread(target=self.writer, args=(self.trick_id, self.video_type), )
            writer_thread.start()
            frame_reader.join()
            print("[INFO] Reader thread finished")
            print("[INFO] All frames read and queued for GPU")
            print("[INFO] All GPU processing completed")
            print("[INFO] All frames fully processed")
            self.writer_stop.set()
            writer_thread.join()
            print("[INFO] Writer thread finished")
            print("[INFO] All processing completed successfully!")

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
            self.stop_event.set()
            self.writer_stop.set()
        except Exception as e:
            print(f"[ERROR] Exception in run: {e}")
            import traceback
            traceback.print_exc()
            self.stop_event.set()
            self.writer_stop.set()
            raise
        
        return

    def read_frames(self):
        self.last_frame_index = -1
        ok, first_frame = self.cap.read()

        if not ok:
            raise IOError('[ERROR] Cannot read video file')
        h, w = first_frame.shape[:2]

        if w != 1920 or h != 1080:
            self.first_frame = cv2.resize(first_frame, (1920, 1080))
        else:
            self.first_frame = first_frame.copy()
        if self.trick_id in [1, 2, 3] and self.video_type == "Dynamic":
            aruco_dict = self.data_video["aruco_dict"]
            aruco_params = self.data_video["aruco_params"]
            aruco_detector = self.data_video["aruco_detector"]
            gray = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY)
            self.data_video["first_frame_corners"], ids, _ = aruco_detector.detectMarkers(gray)
            self.data_video["first_frame_ids"] = ids.flatten() if ids is not None else None

        input_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.start_time * input_fps))
        
        start_frame = int(self.start_time * input_fps)
        end_frame = int(self.end_time * input_fps)
        
        
        target_fps = 30.0
        frame_interval = 1.0 / target_fps
        next_output_time = 0.0
        frame_index = 0
        # We drop some frames to have 30FPS
        while True:
            ok, frame = self.cap.read()

            if not ok:
                break
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame >= end_frame:
                break

            current_time = (current_frame - start_frame) / input_fps

            if current_time >= next_output_time:
                # ensure 1080p
                h, w = frame.shape[:2]
                if w != 1920 or h != 1080:
                    frame = cv2.resize(frame, (1920, 1080))
                
                self.frame_queue.put((frame_index, frame))
                next_output_time += frame_interval
                frame_index += 1
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))  
                print(f"\r[INFO] Reading frames... Current frame: {current_frame} AKA {frame_index}")
        
        self.cap.release()
        self.last_frame_index = frame_index
        print("[INFO] Frame reading completed.")
        self.stop_event.set()
        

    def gpu_process(self):
        # idk how to make it work for 2 GPUs yet.. let's do it for 1 GPU first
        MODEL_ID = "IDEA-Research/grounding-dino-tiny"

        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        if self.trick_id == 1:
            PROMPT = "a soda can . a blue ball . a small figurine . a wooden table . a desk . a overhead projector. a projector arm . an alimentation cable . projector button . a grey cylindrical object . a black circle . a coat rack . a coat hanger . a coat ."
        if self.trick_id == 2:
            PROMPT = "a soda can . a blue ball . a small figurine . a magic wand . a black circle ."
        if self.trick_id == 3:
            PROMPT = "a blue ball . a magic wand . a overhead projector. a projector arm . an alimentation cable . a black circle . a coat rack . a coat hanger . a coat ."

        TEXT_THRESHOLD = 0.20
        if self.trick_id == 2 and self.video_type == "Dynamic":
            TEXT_THRESHOLD = 0.3
        total_frames_processed = 0
        while not self.stop_event.is_set() or not self.frame_queue.empty() or (self.last_frame_index != -1 and total_frames_processed < self.last_frame_index):
            try:
                frame_num, frame = self.frame_queue.get(timeout=1)
                self.frame_queue.task_done()

                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                inputs = processor(images=pil_image, text=PROMPT, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    text_threshold=TEXT_THRESHOLD,
                    target_sizes=[pil_image.size[::-1]]
                )[0]

                boxes = results["boxes"].cpu().numpy()
                scores = results["scores"].cpu().numpy()
                labels = results["labels"]
                total_frames_processed += 1
                self.gpu_queue.put((frame_num, frame, boxes, scores, labels))
            except queue.Empty:
                if self.stop_event.is_set() and total_frames_processed >= self.last_frame_index:
                    break
                continue
            except Exception as e:
                print(f"[ERROR][GPU_THREAD] GPU thread encountered an error: {e}")
                break
        


    def trick_process(self):
            box_threshold = 0.3
            bg_green_mask = None
            total_frames_processed = 0
            while not self.stop_event.is_set() or not self.gpu_queue.empty() or (self.last_frame_index != -1 and total_frames_processed < self.last_frame_index):
                try:
                    frame_num, frame, boxes, scores, labels = self.gpu_queue.get(timeout=1)
                    self.gpu_queue.task_done()
                    best_per_label = {}
                    if self.trick_id >= 2:
                        best_per_label = {}
                        for box, score, label in zip(boxes, scores, labels):
                            if score < box_threshold:
                                continue
                            try:
                                if hasattr(label, "item"):
                                    key = int(label.item())
                                else:
                                    key = int(label)
                            except Exception:
                                key = str(label)
                            if key not in best_per_label or score > best_per_label[key][0]:
                                best_per_label[key] = (float(score), box, label)
                        selected = sorted(best_per_label.values(), key=lambda x: x[0], reverse=True)
                    else:
                        table_bbox = None
                        width = frame.shape[1]
                        height = frame.shape[0]
                        table_candidates = []
                        for box, score, label in zip(boxes, scores, labels):
                            if score < box_threshold:
                                continue
                            label_str = str(label).lower()
                            if "table" in label_str or "desk" in label_str:
                                x1, y1, x2, y2 = map(int, box)
                                bbox_area = (x2 - x1) * (y2 - y1)
                                frame_area = width * height
                                bbox_ratio = bbox_area / frame_area
                                if bbox_ratio >= 0.05:
                                    table_candidates.append((score, (x1, y1, x2, y2), label_str))
                        
                        if table_candidates:
                            table_candidates.sort(key=lambda x: x[0], reverse=True)
                            table_bbox = table_candidates[0][1]
                    
                    
                    if self.trick_id == 1 and self.video_type == "Static":
                        if bg_green_mask is None:
                            lower_green = np.array([35, 40, 40])
                            upper_green = np.array([85, 255, 255])
                            bg_hsv = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2HSV)
                            bg_green_mask = cv2.inRange(bg_hsv, lower_green, upper_green)
                            kernel = np.ones((5, 5), np.uint8)
                            bg_green_mask = cv2.morphologyEx(bg_green_mask, cv2.MORPH_OPEN, kernel)
                            bg_green_mask = cv2.morphologyEx(bg_green_mask, cv2.MORPH_CLOSE, kernel)
                        processed_frame = trick_1_static(frame, self.first_frame, table_bbox, zip(boxes, scores, labels), bg_green_mask)
                        self.writer_queue.put((frame_num, processed_frame, ))
                    elif self.trick_id == 2 and self.video_type == "Static": 
                        data_frame = trick_2_static_part1(frame, selected) 
                        self.process_queue.put((frame_num, frame, data_frame, ))
                        print(f"\r[INFO][trick_process] Processing frame {frame_num}")
                    elif self.trick_id == 3 and self.video_type == "Static":
                        data_frame = trick_3_static_part1(frame, selected, self.data_video)
                        self.process_queue.put((frame_num, frame, data_frame, ))
                        print(f"\r[INFO][trick_process] Processing frame {frame_num}")
                    elif self.trick_id == 1 and self.video_type == "Dynamic":
                        data_frame = trick_1_dynamic_part1(frame, table_bbox, zip(boxes, scores, labels), self.data_video)
                        self.process_queue.put((frame_num, frame, data_frame, ))
                        print(f"\r[INFO][trick_process] Processing frame {frame_num}")
                    elif self.trick_id == 2 and self.video_type == "Dynamic":
                        data_frame = trick_2_dynamic_part1(frame, selected, self.data_video)
                        self.process_queue.put((frame_num, frame, data_frame, ))
                        print(f"\r[INFO][trick_process] Processing frame {frame_num}")
                    elif self.trick_id == 3 and self.video_type == "Dynamic":
                        data_frame = trick_3_dynamic_part1(frame, selected, self.data_video)
                        self.process_queue.put((frame_num, frame, data_frame, ))
                        print(f"\r[INFO][trick_process] Processing frame {frame_num}")

                    total_frames_processed += 1

                    
                except queue.Empty:
                    if self.stop_event.is_set() and total_frames_processed >= self.last_frame_index:
                        break
                    continue
                except Exception as e:
                    print(f"[ERROR][trick_process] thread encountered an error: {e}")
                    break

    def trick_process_next(self):
        frame_buffer = {}
        next_frame_to_process = 0
        total_frames_processed = 0
        while not self.stop_event.is_set() or not self.process_queue.empty() or (self.last_frame_index != -1 and total_frames_processed < self.last_frame_index):
            try:
                frame_num, frame, data = self.process_queue.get(timeout=1)
                self.process_queue.task_done()
                
                frame_buffer[frame_num] = {"frame" : frame, "data" : data}
                while next_frame_to_process in frame_buffer:
                    next_frame = frame_buffer.pop(next_frame_to_process)
                    frame = next_frame["frame"]
                    data = next_frame["data"]
                    if self.trick_id == 2 and self.video_type == "Static":
                        print(f"data_video BEFORE = {self.data_video}")
                        print(f"data BEFORE = {data}") 
                        processed_frame = trick_2_static_part2(frame, frame_num, self.first_frame, data, self.data_video) 
                        print(f"data_video AFTER = {self.data_video}")
                        self.writer_queue.put((frame_num, processed_frame, ))
                    elif self.trick_id == 3 and self.video_type == "Static":
                        processed_frame = trick_3_static_part2(frame,frame_num, self.first_frame, data, self.data_video)
                        self.writer_queue.put((frame_num, processed_frame, ))
                    elif self.trick_id == 1 and self.video_type == "Dynamic":
                        processed_frame = trick_1_dynamic_part2(frame, self.first_frame, data, self.data_video)
                        self.writer_queue.put((frame_num, processed_frame, ))
                    elif self.trick_id == 2 and self.video_type == "Dynamic":
                        processed_frame = trick_2_dynamic_part2(frame, frame_num, self.first_frame, data, self.data_video)
                        self.writer_queue.put((frame_num, processed_frame, ))
                    elif self.trick_id == 3 and self.video_type == "Dynamic":
                        processed_frame = trick_3_dynamic_part2(frame,frame_num, self.first_frame, data, self.data_video)
                        self.writer_queue.put((frame_num, processed_frame, ))

                    print(f"\r[INFO][trick_process_next] Processing frame {next_frame_to_process}")
                    next_frame_to_process += 1
                    total_frames_processed += 1
                                    
            except queue.Empty:
                if self.stop_event.is_set() and total_frames_processed >= self.last_frame_index:
                    break
                continue
            except Exception as e:
                print(f"[ERROR][trick_process_next] thread encountered an error: {e}")
                self.process_queue.task_done()
                break        

    def writer(self, trick_id, video_type):
        fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        width_video = 1920
        height_video = 1080
        self.writer = cv2.VideoWriter(f"output_trick_{trick_id}_{video_type}_{self.number_in_annotation_file}.mp4", fourcc, fps, (width_video, height_video))
        if not self.writer.isOpened():
            raise IOError("[ERROR][WRITER_THREAD] Could not open VideoWriter")

        frame_buffer = {}
        next_frame_to_write = 0
        total_frames_written = 0
        while not self.writer_stop.is_set() or not self.writer_queue.empty() or (self.last_frame_index != -1 and total_frames_written < self.last_frame_index):
            
            try:
                frame_num, processed_frame = self.writer_queue.get(timeout=1)
                self.writer_queue.task_done()   
                frame_buffer[frame_num] = processed_frame
                while next_frame_to_write in frame_buffer:
                    next_frame = frame_buffer.pop(next_frame_to_write)
                    self.writer.write(next_frame)
                    print(f"\r[INFO][WRITER_THREAD] Writing frame {next_frame_to_write}")
                    next_frame_to_write += 1
                    total_frames_written += 1
            
            except queue.Empty:
                if self.writer_stop.is_set() and total_frames_written >= self.last_frame_index:
                    break
                continue
            except Exception as e:
                print(f"[ERROR][WRITER_THREAD] Writer thread encountered an error: {e}")
                break

        while next_frame_to_write in frame_buffer:      
            next_frame = frame_buffer.pop(next_frame_to_write)
            self.writer.write(next_frame)
            print(f"\r[INFO][WRITER_THREAD] Writing frame {next_frame_to_write}")
            next_frame_to_write += 1

        if self.writer is not None:
            self.writer.release()

        print("[INFO][WRITER_THREAD] Video writing completed.")

    def show_video(self, trick_id):

        frame_buffer = {}
        next_frame_to_write = 0

        while not self.writer_stop.is_set() or not self.writer_queue.empty():
            try:
                frame_num, processed_frame = self.writer_queue.get(timeout=1)
                self.writer_queue.task_done()   

                frame_buffer[frame_num] = processed_frame
                while next_frame_to_write in frame_buffer:
                    next_frame = frame_buffer.pop(next_frame_to_write)
                    cv2.imshow(f"Trick {trick_id} Output", next_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        exit()
                        break
                    print(f"\r[INFO][WRITER_THREAD] Writing frame {next_frame_to_write}")
                    next_frame_to_write += 1
            
            except queue.Empty:
                if self.writer_stop.is_set():
                    break
                continue
            except Exception as e:
                print(f"[ERROR][WRITER_THREAD] Writer thread encountered an error: {e}")
                break