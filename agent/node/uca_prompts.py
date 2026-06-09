"""UCA (UCF-Crime Annotation) 格式 LLM 输出 prompt。

数据集参考: Surveillance-Video-Understanding / UCA (CVPR 2024)
官方格式:
    {
        "VideoName": {
            "duration": float,                         # 视频总时长 (秒)
            "timestamps": [[start, end], ...],         # 每条事件的起止秒数 (精度 0.1s)
            "sentences": ["event description", ...]    # 与 timestamps 一一对应的英文事件描述
        }
    }

本文件提供:
    - UCA_OUTPUT_SCHEMA          : 结构化 JSON schema（可直接喂给 with_structured_output）
    - UCA_SYSTEM_PROMPT          : 系统提示
    - build_uca_dense_caption_prompt(...) : 从帧级 caption / 检测事件生成 UCA JSON 的 prompt
"""

from __future__ import annotations

from typing import Any, Iterable


UCA_OUTPUT_SCHEMA: dict[str, Any] = {
    "title": "uca_dense_caption",
    "type": "object",
    "properties": {
        "video_name": {"type": "string"},
        "duration": {"type": "number"},
        "timestamps": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
        "sentences": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["video_name", "duration", "timestamps", "sentences"],
}


UCA_SYSTEM_PROMPT = (
    "You are a surveillance-video dense captioning assistant. "
    "Your job is to describe a video as a list of non-overlapping or lightly-overlapping events "
    "in the UCA (UCF-Crime Annotation) format. "
    "Rules:\n"
    "1. Output a single JSON object, no markdown fences, no explanation.\n"
    "2. Timestamps are in seconds with 0.1s precision; start < end; end <= duration.\n"
    "3. sentences[i] corresponds to timestamps[i]; lengths MUST match.\n"
    "4. Each sentence is ONE English sentence, target ~20 words, matching the UCA style below.\n"
    "5. Do not invent objects or actions that were not present in the input evidence.\n"
    "6. Keep sentences factual and surveillance-oriented (no speculation, no emotions, no judgment).\n"
    "7. UCA style cues: describe subjects generically (\"a man\", \"two men\", \"a woman in red\", "
    "\"a white car\"); refer to the scene with simple surveillance phrases (\"on the screen\", "
    "\"in the middle of the road\", \"behind the counter\", \"in the corner\"); chain actions "
    "with simple verbs (walk, stand, push, fall, pick up, run, enter, leave).\n\n"
    "=== UCA style examples (study the phrasing + timestamp granularity) ===\n"
    "Example A (Robbery030_x264, duration 99.2s):\n"
    '{"video_name":"Robbery030_x264","duration":99.2,'
    '"timestamps":[[10.1,18.5],[19.9,35.4],[39.7,75.6],[75.6,89.9],[89.9,95.2]],'
    '"sentences":['
    '"The screen lights up with three men entering the house to move things into a bucket",'
    '"Three men moved things, two men stood with their hands raised",'
    '"Several men continue to move things",'
    '"Two men sat in two corners waiting for three men to leave with buckets",'
    '"A man goes out"]}\n\n'
    "Example B (Shoplifting017_x264, duration 15.3s):\n"
    '{"video_name":"Shoplifting017_x264","duration":15.3,'
    '"timestamps":[[0.6,11.7],[11.7,15.0],[11.7,15.0]],'
    '"sentences":['
    '"The man in a gray suit said something to the boss, and the boss stood up and walked away",'
    '"The man in the gray suit put the bag on the table, then reached for something on the counter",'
    '"The woman behind the man in gray suit walked out"]}\n\n'
    "Example C (Fighting009_x264, duration 71.1s):\n"
    '{"video_name":"Fighting009_x264","duration":71.1,'
    '"timestamps":[[0.8,17.8],[19.9,46.7],[52.1,58.8],[52.1,58.8]],'
    '"sentences":['
    '"A few people jumped and danced, and a row of people on the right drank and chatted",'
    '"A man in a shirt continues to provoke another man",'
    '"A man in a shirt walked to the center of the screen and shook another man, causing him to fall",'
    '"Another man pushed the man behind him down and ran out of the screen"]}\n\n'
    "=== end examples ==="
)


def build_uca_dense_caption_prompt(
    video_name: str,
    duration: float,
    frame_events: Iterable[dict[str, Any]],
    extra_hint: str | None = None,
) -> str:
    """构造 UCA 稠密事件描述 prompt.

    参数:
        video_name   : 视频文件名 (不含扩展名)，用于 output.video_name
        duration     : 视频总时长 (秒)
        frame_events : 帧/片段级证据, 每个 dict 形如
                       {"t": 3.2, "objects": ["car","dog"], "caption": "...", "bbox": ...}
                       字段可缺省，prompt 只序列化存在的键
        extra_hint   : 可选的领域提示 (例如异常类型)
    """
    evidence_lines: list[str] = []
    for ev in frame_events:
        t = ev.get("t") or ev.get("time") or ev.get("timestamp")
        parts: list[str] = []
        if t is not None:
            parts.append(f"t={float(t):.1f}s")
        if "objects" in ev and ev["objects"]:
            parts.append(f"objects={list(ev['objects'])}")
        if "caption" in ev and ev["caption"]:
            parts.append(f'caption="{ev["caption"]}"')
        if "action" in ev and ev["action"]:
            parts.append(f'action="{ev["action"]}"')
        if parts:
            evidence_lines.append("- " + ", ".join(parts))

    evidence_block = "\n".join(evidence_lines) if evidence_lines else "- (no frame evidence provided)"
    hint_block = f"\n\nDomain hint: {extra_hint}" if extra_hint else ""

    return (
        f"Video name: {video_name}\n"
        f"Duration: {float(duration):.2f} seconds\n\n"
        f"Frame-level evidence (chronological):\n{evidence_block}{hint_block}\n\n"
        "Produce a UCA-format JSON with fields: video_name, duration, timestamps, sentences.\n"
        "timestamps[i] must be [start_sec, end_sec] with 0.1s precision; sentences[i] describes that segment."
    )


__all__ = [
    "UCA_OUTPUT_SCHEMA",
    "UCA_SYSTEM_PROMPT",
    "build_uca_dense_caption_prompt",
]
