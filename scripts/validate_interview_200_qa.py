from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QA_PATH = REPO_ROOT / "docs" / "INTERVIEW_200_QA_ZH.md"
README_PATH = REPO_ROOT / "README.md"


def main() -> None:
    text = QA_PATH.read_text(encoding="utf-8")
    ids = re.findall(r"^### (Q\d{3})：.+$", text, flags=re.MULTILINE)
    expected = [f"Q{number:03d}" for number in range(1, 201)]
    answer_count = len(re.findall(r"^\*\*答案：\*\* .+$", text, flags=re.MULTILINE))
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
    readme = README_PATH.read_text(encoding="utf-8")
    link = "docs/INTERVIEW_200_QA_ZH.md"

    if ids != expected:
        raise SystemExit("question IDs must be exactly Q001..Q200 in order")
    if len(ids) != 200:
        raise SystemExit(f"question count must be 200, got {len(ids)}")
    if answer_count != 200:
        raise SystemExit(f"answer count must be 200, got {answer_count}")
    if duplicate_ids:
        raise SystemExit(f"duplicate IDs: {duplicate_ids}")
    if link not in readme:
        raise SystemExit(f"README must link {link}")

    print(
        "interview_qa_validation=PASS "
        "questions=200 answers=200 duplicates=0 ids=Q001..Q200 "
        "readme_link=present"
    )


if __name__ == "__main__":
    main()
