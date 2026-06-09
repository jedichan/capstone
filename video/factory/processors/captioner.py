import os
import json
import base64
import time
from io import BytesIO

import cv2
from PIL import Image
from openai import OpenAI


class MinimalQwenVLCaptioner:
    def __init__(
        self,
        seconds_per_caption: int = 2,
        frames_per_caption: int = 4,
        output_dir: str = "outputs",
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    ):
        self.seconds_per_caption = seconds_per_caption
        self.frames_per_caption = frames_per_caption
        self.output_dir = output_dir
        self.model_name = model_name

        os.makedirs(self.output_dir, exist_ok=True)

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def _sample_segment_frames(self, cap, fps: int, start_frame_idx: int):
        """
        Match prior LaViLa-style sampling:
        - one segment every 2 seconds
        - 4 frames per segment
        - skip the rest within the segment window
        """
        frames = []
        sampled_indices = []
        segment_total_frames = fps * self.seconds_per_caption
        frame_interval = segment_total_frames // self.frames_per_caption

        current_idx = start_frame_idx

        for _ in range(self.frames_per_caption):
            success, frame = cap.read()
            if not success:
                return None, None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            sampled_indices.append(current_idx)
            current_idx += 1

            for _ in range(frame_interval - 1):
                success, _ = cap.read()
                if not success:
                    return None, None
                current_idx += 1

        remaining = segment_total_frames - frame_interval * self.frames_per_caption
        for _ in range(remaining):
            success, _ = cap.read()
            if not success:
                return None, None
            current_idx += 1

        return frames, sampled_indices

    def _frame_to_data_url(self, frame_rgb):
        image = Image.fromarray(frame_rgb)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _generate_caption_from_frames(self, frames):
        """
        Feed multiple frames from one segment to Qwen-VL together.
        """
        content = [
            {
                "type": "text",
                "text": (
                    "You are a video captioning assistant.\n"
                    f"These {self.frames_per_caption} images are sampled chronologically from one short video segment.\n"
                    "Write ONE concise caption in English describing the main action/event.\n"
                    "Requirements:\n"
                    "- one sentence only\n"
                    "- focus on visible action\n"
                    "- no speculation\n"
                    "- no bullet points\n"
                    "- no timestamps\n"
                ),
            }
        ]

        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": self._frame_to_data_url(frame)
                }
            })

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            temperature=0.2,
            max_tokens=64,
        )

        text = response.choices[0].message.content.strip()
        return text

    def caption_video(self, video_path: str):
        start_time_proc = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_captions = total_frames // (fps * self.seconds_per_caption)

        video_name = os.path.basename(video_path)
        video_name_no_ext = os.path.splitext(video_name)[0]
        video_out_dir = os.path.join(self.output_dir, video_name_no_ext)
        os.makedirs(video_out_dir, exist_ok=True)

        print(f"\nVideo: {video_path}")
        print(f"FPS: {fps}, total_frames: {total_frames}, total_segments: {total_captions}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        segments_data = []

        for caption_id in range(total_captions):
            start_frame = caption_id * fps * self.seconds_per_caption
            end_frame = (caption_id + 1) * fps * self.seconds_per_caption
            
            frames, sampled_indices = self._sample_segment_frames(cap, fps, start_frame)
            if frames is None:
                break

            text = self._generate_caption_from_frames(frames)

            segment_key = f"{start_frame}_{end_frame}"

            segment_info = {
                "segment_id": caption_id,
                "segment_key": segment_key,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": float(start_frame / fps),
                "end_time": float(end_frame / fps),
                "sampled_frame_indices": sampled_indices,
                "caption": text
            }
            segments_data.append(segment_info)

            print(f"[{caption_id}] {segment_key} -> {text}")

        cap.release()
        
        output_json = {
            "video_name": video_name,
            "fps": fps,
            "seconds_per_caption": self.seconds_per_caption,
            "frames_per_caption": self.frames_per_caption,
            "segments": segments_data
        }

        captions_path = os.path.join(video_out_dir, "captions.json")

        with open(captions_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        end_time_proc = time.time()
        print(f"Saved results to: {captions_path}")
        print(f"Time taken for {video_name}: {end_time_proc - start_time_proc:.2f} seconds")

        return output_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to a video file or a directory of videos")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of concurrent videos to process")
    parser.add_argument("--frames_per_caption", type=int, default=4, help="Number of frames to sample per segment")
    parser.add_argument("--seconds_per_caption", type=int, default=2, help="Length of each segment in seconds")
    args = parser.parse_args()

    captioner = MinimalQwenVLCaptioner(
        seconds_per_caption=args.seconds_per_caption,
        frames_per_caption=args.frames_per_caption,
        output_dir=args.output_dir,
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
    )
    
    start_total = time.time()
    
    if os.path.isdir(args.video_path):
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov')
        video_files = [os.path.join(args.video_path, f) for f in os.listdir(args.video_path) if f.lower().endswith(video_extensions)]
        print(f"Found {len(video_files)} videos in directory '{args.video_path}'")
        
        if args.max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            print(f"Processing with {args.max_workers} concurrent workers...")
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                executor.map(captioner.caption_video, video_files)
        else:
            for vf in video_files:
                captioner.caption_video(vf)
    else:
        captioner.caption_video(args.video_path)
        
    end_total = time.time()
    print(f"\nTotal processing time: {end_total - start_total:.2f} seconds")
