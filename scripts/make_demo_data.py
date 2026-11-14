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
        "\u673a\u5668\u4eba\u6839\u636e\u7b80\u77ed\u7684\u8bed\u8a00\u6307\u4ee4\u884c\u52a8。\u6444\u50cf\u5934\u89c2\u5bdf\u623f\u95f4，\u89c4\u5212\u5668\u9009\u62e9\u4e0b\u4e00\u6b65\u52a8\u4f5c。",
        "\u8bed\u8a00\u6a21\u578b\u901a\u8fc7\u9884\u6d4b\u4e0b\u4e00\u4e2a\u8bcd\u5143\u5b66\u4e60\u6587\u672c。\u8bad\u7ec3\u524d\u5148\u628a\u4e2d\u82f1\u6587\u6587\u6863\u5206\u6210\u5b57\u8282\u8bcd\u5143。",
        "\u5b66\u751f\u9605\u8bfb\u95ee\u9898\u5e76\u7ed9\u51fa\u6e05\u695a\u7684\u7b54\u6848。\u6587\u6863\u4e2d\u7684\u8bc1\u636e\u5e94\u5f53\u652f\u6301\u7b54\u6848，\u8d44\u6599\u4e0d\u8db3\u65f6\u8bf4\u660e\u7f3a\u53e3。",
        "\u8ba1\u7b97\u673a\u5728\u751f\u6210\u8fc7\u7a0b\u4e2d\u4fdd\u5b58\u952e\u548c\u503c。\u65b0\u4f4d\u7f6e\u7684\u8ba1\u7b97\u4f1a\u4f7f\u7528\u524d\u9762\u4f4d\u7f6e\u5df2\u7ecf\u4fdd\u5b58\u7684\u72b6\u6001。",
        "\u8bb0\u5f55\u8ba1\u7b97\u56fe\u9700\u8981\u51c6\u5907\u65f6\u95f4。\u662f\u5426\u4fdd\u7559\u8ba1\u7b97\u56fe，\u5e94\u5f53\u6839\u636e\u5b9e\u9645\u8017\u65f6\u548c\u540e\u7eed\u590d\u7528\u673a\u4f1a\u5224\u65ad。",
        "\u7ea2\u8272\u673a\u5668\u4eba\u505c\u5728\u84dd\u8272\u6905\u5b50\u65c1\u8fb9，\u7eff\u8272\u673a\u5668\u4eba\u6162\u6162\u8d70\u5411\u6253\u5f00\u7684\u95e8。",
    ]
    for language, texts in (("en", english), ("zh", chinese)):
        for split, count in (("train", 48), ("dev", 12)):
            rows = [{"text": f"{language} {split} document {i}. " + " ".join(texts[i % len(texts):] + texts[:i % len(texts)])} for i in range(count)]
            write_rows(args.output / "pretrain" / f"{language}-{split}.jsonl", rows)
    examples = [
        ("What is a robot?", "A robot observes and acts.", "A robot observes and acts."),
        ("What does the camera do?", "The camera observes the room.", "The camera observes the room."),
        ("\u673a\u5668\u4eba\u505a\u4ec0\u4e48？", "\u673a\u5668\u4eba\u6839\u636e\u6307\u4ee4\u884c\u52a8。", "\u673a\u5668\u4eba\u6839\u636e\u6307\u4ee4\u884c\u52a8。"),
        ("\u6a21\u578b\u9884\u6d4b\u4ec0\u4e48？", "\u6a21\u578b\u9884\u6d4b\u4e0b\u4e00\u4e2a\u8bcd\u5143。", "\u6a21\u578b\u9884\u6d4b\u4e0b\u4e00\u4e2a\u8bcd\u5143。"),
    ]
    for split, selected in (("train", examples), ("dev", examples[:2])):
        rows = [{"question": q, "answer": a, "context": c, "system": "Be clear.", "family_id": f"synthetic-{split}-{i}", "source_ids": [f"original-synthetic-{split}-{i}"]} for i, (q, a, c) in enumerate(selected)]
        write_rows(args.output / "sft" / f"{split}.jsonl", rows)
    write_rows(args.output / "requests.jsonl", [{"question": q, "context": c} for q, _, c in examples])
    print("Created bilingual CPU fixtures. These tiny, overlapping examples verify execution only; they are not held-out quality evaluation.")


if __name__ == "__main__":
    main()
