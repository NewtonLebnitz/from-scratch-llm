"""Create original synthetic fixtures for the CPU plumbing demo, not a benchmark."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".runtime/input"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory to preserve existing data.")
    english = [
        "A robot follows a short instruction. Its camera observes the room and the planner selects an action.",
        "Language model training predicts the next token. A document is split into small byte tokens before training.",
        "A student reads a question and writes a clear answer. Evidence in the document supports the answer.",
        "The computer stores keys and values during generation. A new token uses the states of earlier tokens.",
        "A graph can replay operations on the GPU. Creating the graph takes time and should be measured.",
        "The red robot stops near the blue chair. The green robot moves slowly towards the open door.",
    ]
    chinese = [
        "机器人根据简短的语言指令行动。摄像头观察房间，规划器选择下一步动作。",
        "语言模型通过预测下一个词元学习文本。训练前先把中英文文档分成字节词元。",
        "学生阅读问题并给出清楚的答案。文档中的证据应当支持答案，资料不足时说明缺口。",
        "计算机在生成过程中保存键和值。新位置的计算会使用前面位置已经保存的状态。",
        "记录计算图需要准备时间。是否保留计算图，应当根据实际耗时和后续复用机会判断。",
        "红色机器人停在蓝色椅子旁边，绿色机器人慢慢走向打开的门。",
    ]
    for language, texts in (("en", english), ("zh", chinese)):
        for split, count in (("train", 48), ("dev", 12)):
            rows = [{"text": f"{language} {split} document {i}. " + " ".join(texts[i % len(texts):] + texts[:i % len(texts)])} for i in range(count)]
            write_rows(args.output / "pretrain" / f"{language}-{split}.jsonl", rows)
    examples = [
        ("What is a robot?", "A robot observes and acts.", "A robot observes and acts."),
        ("What does the camera do?", "The camera observes the room.", "The camera observes the room."),
        ("机器人做什么？", "机器人根据指令行动。", "机器人根据指令行动。"),
        ("模型预测什么？", "模型预测下一个词元。", "模型预测下一个词元。"),
    ]
    for split, selected in (("train", examples), ("dev", examples[:2])):
        rows = [{"question": q, "answer": a, "context": c, "system": "Be clear.", "family_id": f"synthetic-{split}-{i}", "source_ids": [f"original-synthetic-{split}-{i}"]} for i, (q, a, c) in enumerate(selected)]
        write_rows(args.output / "sft" / f"{split}.jsonl", rows)
    write_rows(args.output / "requests.jsonl", [{"question": q, "context": c} for q, _, c in examples])
    print("Created bilingual CPU fixtures. These tiny, overlapping examples verify execution only; they are not held-out quality evaluation.")


if __name__ == "__main__":
    main()
