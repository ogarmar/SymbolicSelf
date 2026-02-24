import json
import random
from pathlib import Path

DATA_ROOT = Path("data")
ANNOT_ROOT = DATA_ROOT / "annotations"
OUT_FILE = ANNOT_ROOT / "vqa_subset.json"

N_SAMPLES = 5000


def main():
    questions_path = ANNOT_ROOT / "v2_OpenEnded_mscoco_train2014_questions.json"
    ann_path = ANNOT_ROOT / "v2_mscoco_train2014_annotations.json"

    with open(questions_path) as f:
        qs = json.load(f)["questions"]
    with open(ann_path) as f:
        anns = json.load(f)["annotations"]

    ann_by_qid = {a["question_id"]: a for a in anns}
    random.seed(42)
    subset_qs = random.sample(qs, N_SAMPLES)

    subset = []
    for q in subset_qs:
        qid = q["question_id"]
        ann = ann_by_qid.get(qid)
        if not ann:
            continue
        subset.append(
            {
                "image_id": q["image_id"],
                "question": q["question"],
                "answers": [a["answer"] for a in ann["answers"]],
            }
        )

    with open(OUT_FILE, "w") as f:
        json.dump(subset, f)
    print(f"Saved subset with {len(subset)} examples to {OUT_FILE}")


if __name__ == "__main__":
    main()
