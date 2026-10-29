"""Sắp xếp lại thư mục project sang cấu trúc mới (chạy 1 lần, chạy lại cũng an toàn).

    cd sentiment_analysis_pipeline
    python scripts/restructure.py            # xem trước (không thay đổi gì)
    python scripts/restructure.py --apply    # thực hiện

Không xóa file nào: file cũ/thừa được chuyển vào _old/ để nhóm tự kiểm tra rồi xóa.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LATEX_SRC = "Latex/Latex (1)"

MOVES = [
    # dữ liệu
    ("data_sample/sample_input.csv", "data/sample/sample_input.csv"),
    ("data/raw/new_review.csv", "data/external/new_review.csv"),
    ("data/raw/new_review_labeled.csv", "data/external/new_review_labeled.csv"),
    ("data/processed/train_cleaned.csv", "_old/data/processed/train_cleaned.csv"),
    # checkpoint
    ("model_checkpoints", "models/phobert-sentiment"),
    # notebook cũ (đã được thay bằng bản mới gọi src/)
    ("notebooks/01_final_data_pipeline.ipynb", "_old/notebooks/01_final_data_pipeline.ipynb"),
    ("notebooks/02_final_model.ipynb", "_old/notebooks/02_final_model.ipynb"),
    ("notebooks/03_error_analysis_demo.ipynb", "_old/notebooks/03_error_analysis_demo.ipynb"),
    # error_analysis.csv cũ được tạo từ dự đoán MÔ PHỎNG, không phải dự đoán thật của mô hình
    ("results/error_analysis.csv", "_old/results/error_analysis_simulated.csv"),
    # báo cáo
    ("report/Nhom4_Tuan4_4.docx", "reports/Nhom4_Tuan4_4.docx"),
    ("slides_week5.pptx", "reports/slides_week5.pptx"),
    (f"{LATEX_SRC}/main (1).tex", "reports/latex/main.tex"),
    ("Latex/Latex (1).zip", "_old/Latex (1).zip"),
    # file rác
    ("test.py", "_old/test.py"),
]


def plan():
    moves = list(MOVES)
    latex = ROOT / LATEX_SRC
    if latex.is_dir():
        for f in latex.iterdir():
            if f.is_file() and f.suffix != ".tex":
                moves.append((f"{LATEX_SRC}/{f.name}", f"reports/latex/{f.name}"))
    res = ROOT / "results"
    if res.is_dir():
        for f in res.iterdir():
            if f.is_file() and f.name != "error_analysis.csv":
                moves.append((f"results/{f.name}", f"results/archive_week5/{f.name}"))
    return moves


def main(apply: bool):
    for src, dst in plan():
        s, d = ROOT / src, ROOT / dst
        if not s.exists():
            continue
        if d.exists():
            print(f"  BỎ QUA (đích đã tồn tại): {dst}")
            continue
        print(f"  {src}  ->  {dst}")
        if apply:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(s), str(d))

    for d in ("data/processed", "data/external", "data/sample", "models", "reports/latex",
              "results/figures", "results/metrics", "results/predictions", "results/error_analysis"):
        if apply:
            (ROOT / d).mkdir(parents=True, exist_ok=True)
    if apply:
        (ROOT / "data/processed/.gitkeep").touch()
        # dọn các thư mục cũ đã rỗng
        for d in (LATEX_SRC, "Latex", "report", "data_sample"):
            p = ROOT / d
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
                print(f"  xóa thư mục rỗng: {d}")
        print("\nXong. Kiểm tra _old/ rồi xóa nếu không cần.")
    else:
        print("\n(Chế độ xem trước – thêm --apply để thực hiện)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    main(ap.parse_args().apply)
