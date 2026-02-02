# -*- coding: utf-8 -*-
"""CLIP ViT-Base-Patch32 图像-文本匹配与检索 WebUI 演示（不加载真实模型权重）。"""
from __future__ import annotations

import gradio as gr
import numpy as np
from PIL import Image
import io


def fake_load_model():
    """模拟加载 CLIP ViT-Base-Patch32 模型，仅用于界面演示。"""
    return "模型状态：CLIP ViT-Base-Patch32 已就绪（演示模式，未加载真实权重）"


def fake_image_encode(image):
    """模拟图像编码，返回演示向量。"""
    if image is None:
        return None
    # 返回一个模拟的 512 维向量
    return np.random.rand(512).tolist()


def fake_text_encode(text: str):
    """模拟文本编码，返回演示向量。"""
    if not (text or "").strip():
        return None
    # 返回一个模拟的 512 维向量
    return np.random.rand(512).tolist()


def fake_compute_similarity(image_vec, text_vec):
    """计算图像和文本向量的相似度（演示）。"""
    if image_vec is None or text_vec is None:
        return 0.0
    # 简单的余弦相似度计算（演示）
    img_arr = np.array(image_vec)
    txt_arr = np.array(text_vec)
    similarity = np.dot(img_arr, txt_arr) / (np.linalg.norm(img_arr) * np.linalg.norm(txt_arr))
    return float(similarity)


def fake_image_text_match(image, text: str):
    """模拟图像-文本匹配与可视化结果。"""
    if image is None:
        return "请上传图像以进行匹配。"
    if not (text or "").strip():
        return "请输入文本描述以进行匹配。"
    
    # 模拟编码
    image_vec = fake_image_encode(image)
    text_vec = fake_text_encode(text)
    similarity = fake_compute_similarity(image_vec, text_vec)
    
    lines = [
        "[演示] 已对输入进行 CLIP ViT-Base-Patch32 图像-文本匹配（未加载真实模型）。",
        f"输入图像：已接收图像输入",
        f"输入文本：{text[:200]}{'...' if len(text) > 200 else ''}",
        "",
        "匹配结果（演示）：",
        f"相似度分数：{similarity:.4f}",
        "",
        "说明：",
        "- 图像编码维度：512",
        "- 文本编码维度：512",
        "- 相似度范围：[-1, 1]，值越大表示匹配度越高",
        "",
        "加载真实 CLIP ViT-Base-Patch32 模型后，将在此展示真实的匹配结果与可视化。",
    ]
    return "\n".join(lines)


def fake_text_to_image_search(query_text: str, num_results: int = 5):
    """模拟文本到图像检索（演示）。"""
    if not (query_text or "").strip():
        return "请输入查询文本以进行检索。"
    
    num_results = max(1, min(10, int(num_results) if isinstance(num_results, (int, float)) else 5))
    
    lines = [
        f"[演示] 已对查询进行 CLIP ViT-Base-Patch32 文本到图像检索（未加载真实模型）。",
        f"查询文本：{query_text[:200]}{'...' if len(query_text) > 200 else ''}",
        f"返回结果数：{num_results}",
        "",
        "检索结果（演示）：",
    ]
    
    for i in range(num_results):
        score = 0.9 - i * 0.1
        lines.append(f"结果 {i+1}: 相似度 {score:.4f} - [图像占位符]")
    
    lines.extend([
        "",
        "说明：",
        "- 检索基于图像和文本的联合嵌入空间",
        "- 结果按相似度降序排列",
        "- 加载真实模型后，将展示真实的检索结果",
    ])
    
    return "\n".join(lines)


def build_ui():
    with gr.Blocks(title="CLIP ViT-Base-Patch32 WebUI") as demo:
        gr.Markdown("## CLIP ViT-Base-Patch32 · 图像-文本匹配与检索 WebUI 演示")
        gr.Markdown(
            "本界面以交互方式展示 CLIP ViT-Base-Patch32 的典型使用流程："
            "模型加载、图像-文本匹配、文本到图像检索及结果可视化（演示模式，未加载真实模型）。"
        )

        with gr.Row():
            load_btn = gr.Button("加载模型（演示）", variant="primary")
            status_box = gr.Textbox(label="模型状态", value="尚未加载", interactive=False)
        load_btn.click(fn=fake_load_model, outputs=status_box)

        with gr.Tabs():
            with gr.Tab("图像-文本匹配"):
                gr.Markdown("上传图像并输入文本描述，模型将计算相似度并展示匹配结果。")
                with gr.Row():
                    with gr.Column():
                        image_in = gr.Image(label="输入图像", type="pil")
                        text_in = gr.Textbox(
                            label="文本描述",
                            placeholder="例如：一只可爱的小猫坐在窗台上",
                            lines=3,
                        )
                        match_btn = gr.Button("匹配（演示）", variant="primary")
                    with gr.Column():
                        match_out = gr.Textbox(
                            label="匹配结果",
                            lines=15,
                            interactive=False,
                        )
                match_btn.click(
                    fn=fake_image_text_match,
                    inputs=[image_in, text_in],
                    outputs=match_out,
                )

            with gr.Tab("文本到图像检索"):
                gr.Markdown("输入查询文本，模型将在图像库中检索最相似的图像。")
                query_text = gr.Textbox(
                    label="查询文本",
                    placeholder="例如：一只在草地上奔跑的狗",
                    lines=2,
                )
                num_results = gr.Slider(1, 10, value=5, step=1, label="返回结果数")
                search_btn = gr.Button("检索（演示）", variant="primary")
                search_out = gr.Textbox(
                    label="检索结果",
                    lines=15,
                    interactive=False,
                )
                search_btn.click(
                    fn=fake_text_to_image_search,
                    inputs=[query_text, num_results],
                    outputs=search_out,
                )

            with gr.Tab("模型信息"):
                gr.Markdown(
                    "CLIP ViT-Base-Patch32 为 OpenAI 开发的视觉-语言预训练模型，"
                    "采用 Vision Transformer (ViT) 作为图像编码器，BERT 作为文本编码器，"
                    "通过对比学习实现图像与文本的联合表示。"
                )
                info_out = gr.Textbox(
                    label="模型信息",
                    value=(
                        "[演示] 模型：CLIP ViT-Base-Patch32\n"
                        "图像编码器：Vision Transformer (ViT-Base/32)\n"
                        "文本编码器：Transformer (BERT-like)\n"
                        "图像编码维度：512\n"
                        "文本编码维度：512\n"
                        "图像输入尺寸：224×224\n"
                        "架构：双塔结构，对比学习训练"
                    ),
                    lines=8,
                    interactive=False,
                )

        gr.Markdown(
            "---\n*说明：当前为轻量级演示界面，未实际下载与加载 CLIP ViT-Base-Patch32 模型参数。*"
        )
    return demo


def main():
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=8766, share=False)


if __name__ == "__main__":
    main()
