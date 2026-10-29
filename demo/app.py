"""Giao diện Gradio cho mô hình PhoBERT.

Chạy:  python demo/app.py            (mở http://127.0.0.1:7860)
       python demo/app.py --share    (tạo link public tạm thời)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from src.predict import get_predictor

EXAMPLES = [
    ["giảng viên nhiệt tình , giáo trình rất hay và chi tiết ."],
    ["cách truyền đạt khó hiểu và bài tập quá nặng ."],
    ["thầy dạy lý thuyết tạm ổn ."],
    ["tạm thời chưa có ."],
    ["nếu yêu cầu bài tập thì có thể nộp giấy riêng cho cô ."],
    ["cơ sở vật chất phòng học rất tốt, máy lạnh mát mẻ."],
    ["giảng viên thường xuyên đi trễ và ít tương tác với sinh viên."],
    ["dạy tương đối đầy đủ chương trình ."],
]


def gradio_predict(text: str):
    if not text or not text.strip():
        return "⚠️ Vui lòng nhập câu phản hồi", {}, ""
    res = get_predictor().predict(text)
    label = f"{res['label'].upper()}  (độ tin cậy {res['confidence']:.2%})"
    info = f"{res['n_tokens']} token | ngưỡng Trung tính = {res['threshold']}"
    return label, res["probabilities"], info


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PhoBERT Sentiment – UIT-VSFC") as demo:
        gr.Markdown("# 🏫 Phân tích sắc thái phản hồi sinh viên (PhoBERT)\n"
                    "Phân loại câu phản hồi thành **Tiêu cực / Trung tính / Tích cực**.")
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(label="Câu phản hồi", lines=5,
                                 placeholder="Ví dụ: thầy giảng bài rất hay và tâm huyết...")
                with gr.Row():
                    btn_clear = gr.Button("Xóa")
                    btn_run = gr.Button("Phân tích", variant="primary")
            with gr.Column():
                out_label = gr.Textbox(label="Nhãn dự đoán", interactive=False)
                out_probs = gr.Label(label="Xác suất 3 lớp", num_top_classes=3)
                out_info = gr.Textbox(label="Thông tin suy luận", interactive=False)
        gr.Examples(examples=EXAMPLES, inputs=inp)
        outputs = [out_label, out_probs, out_info]
        btn_run.click(gradio_predict, inp, outputs)
        inp.submit(gradio_predict, inp, outputs)
        # Sửa lỗi bản cũ: 4 output nhưng hàm chỉ trả về 3 giá trị
        btn_clear.click(lambda: ("", "", None, ""), outputs=[inp, *outputs])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    get_predictor()  # nạp model trước khi mở giao diện
    build_ui().launch(share=a.share, server_port=a.port)
