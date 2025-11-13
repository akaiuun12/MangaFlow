import os

import numpy as np
import pandas as pd
import gradio as gr

from modules.bbox import run_detection
from modules.ocr import run_ocr
    
MODEL_PATHS = [
    'v2023.12.07_s_yv11'
]

INIT_STATE = {
    "input_img": None,
    "detections": None,
    "detect_img": None,
    "ocr_df": pd.DataFrame(columns=["BOX ID","COORDINATES","DETECTED TEXT"]),
#     "image_copy": None,
#     "auto_mask": np.array([]),
#     "inpaint": None,
}

with gr.Blocks(theme=gr.themes.Soft(), title="MangaFlow AI Image Translator") as demo:
    gr.Markdown(
        """
        # 📚 MangaFlow Text Detector
        Upload a manga image and select a YOLO model to detect text regions.
        """
    )

    # ❇️ 탭 간 데이터 공유를 위한 상태 변수
    st = gr.State(value=INIT_STATE)

    # ocr_data_state = gr.State(value=(None, None))
    # ocr_df_state = gr.State(value=pd.DataFrame(columns=["BOX ID", "COORDINATES", "DETECTED TEXT"]))
    # image_copy_state = gr.State(value=None)
    # auto_mask_state = gr.State(value=np.array([])) 
    # inpainting_result_state = gr.State(value=None)

    with gr.Tabs():
        # --- 탭 1: 텍스트 감지 ---
        with gr.TabItem("1. Detect Text", elem_id="tab_detect"):
            detect_button = gr.Button("Detect Textboxes from Image", variant="primary")
                                 
            with gr.Row():
                with gr.Column(scale=1):
                    # 원본 이미지 업로드
                    img_input = gr.Image(type="numpy", label="📤 Upload Manga Image")
                                    
                with gr.Column(scale=1):
                    detect_output = gr.Image(label="Detected Text Regions", interactive=False)
            
            model_dropdown = gr.Dropdown(
                MODEL_PATHS,
                value=MODEL_PATHS[0],
                label="Select YOLO Model"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    iou_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.7,
                        label="IOU Threshold"
                    )
                with gr.Column(scale=1):
                    conf_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.25,
                        label="Confidence Threshold"
                    )
        # --- 탭 2: OCR ---
        with gr.TabItem("2. OCR", elem_id="tab_ocr"):
            ocr_button = gr.Button("Run OCR on Detected Boxes", variant="primary")

            with gr.Row():
                with gr.Column(scale=1):
                    ocr_input_image = gr.Image(label="Original Image with Boxes", 
                                               interactive=False,
                                               scale=1)
                        
                with gr.Column(scale=1):
                    ocr_output_df = gr.Dataframe(
                        label="Extracted Text Table",
                        headers=["BOX ID", "COORDINATES", "DETECTED TEXT"],
                        col_count=(3, "fixed"),
                        row_count="dynamic",
                        value=pd.DataFrame(columns=["BOX ID", "COORDINATES", "DETECTED TEXT"]),
                        datatype=["str", "str", "str"],
                        interactive=True,
                    )
                    
        #                 go_to_translation_button = gr.Button("Go to Translation →", variant="secondary", elem_id="btn_translate")
                    
        # # --- 탭 3: Translate ---
        # with gr.TabItem("3. Translate", elem_id="tab_translate"):
        #     with gr.Row():
        #         with gr.Column(scale=1):
        #             gr.Markdown("### Image Preview (with Boxes)")
        #             # ❇️ 텍스트 박스가 그려진 이미지 미리보기 (번역용)
        #             translation_image_preview = gr.Image(
        #                 label="Image Preview",
        #                 interactive=False
        #             )
        #             api_selector = gr.Radio(
        #                 ["Gemini API (Google)", "ChatGPT API (OpenAI)"],
        #                 value="Gemini API (Google)",
        #                 label="Select Translation API"
        #             )
        #             run_translation_button = gr.Button("Run Translation", variant="primary")
                    
        #         with gr.Column(scale=1):
        #             gr.Markdown("### Translation Results")
        #             translation_output_df = gr.Dataframe(
        #                 headers=["ORIGINAL TEXT", "TRANSLATED TEXT"],
        #                 col_count=(2, "fixed"),
        #                 row_count="dynamic",
        #                 value=pd.DataFrame(columns=["ORIGINAL TEXT", "TRANSLATED TEXT"]),
        #                 datatype=["str", "str"],
        #                 interactive=True,
        #             )
                    
        #             go_to_inpaint_button = gr.Button("Go to Inpainting →", variant="secondary", elem_id="btn_inpaint")
        
        # # --- 탭 4: Inpainting ---
        # with gr.TabItem("4. Inpainting", elem_id="tab_inpaint"):
        #     gr.Markdown("## 🖌️ Auto Inpainting: Remove Original Text")
        #     with gr.Row():
        #         # --- 컬럼 1: 설정 및 입력 이미지 (좌측) ---
        #         with gr.Column(scale=1):
        #             gr.Markdown("### Source Image & Settings")
                    
        #             # ❇️ 인페인팅에 사용할 원본 이미지 (업로드된 이미지)
        #             inpainting_input_image = gr.Image(
        #                 label="Inpainting Source Image (Uploaded Original)",
        #                 interactive=False,
        #                 height=400 
        #             )
                    
        #             inpainting_model_dropdown = gr.Dropdown(
        #                 ["Lama (Default)", "Other Model (Future)"], 
        #                 value="Lama (Default)",
        #                 label="Select Inpainting Model"
        #             )

        #             # ❇️ [추가] Dilation 슬라이더
        #             dilation_slider = gr.Slider(
        #                 minimum=0,
        #                 maximum=20,
        #                 step=1,
        #                 value=5, 
        #                 label="Mask Dilation (Pixel Size for Expansion)"
        #             )
                    
        #             run_inpainting_button = gr.Button("Run Inpainting", variant="primary")
                
        #         # --- 컬럼 2: 자동 마스크 표시 (중앙) ---
        #         with gr.Column(scale=2): 
        #             gr.Markdown("### Auto-Mask Preview")
                    
        #             # ❇️ 자동 마스크 미리보기 (Grayscale)
        #             inpainting_editor = gr.Image(
        #                 label="Auto-Generated Mask (White areas will be erased)",
        #                 type="numpy",
        #                 image_mode="L", # Grayscale로 표시
        #                 height=600 
        #             )

        #         # --- 컬럼 3: 결과 이미지 (우측) ---
        #         with gr.Column(scale=1):
        #             gr.Markdown("### Inpainting Result")
                    
        #             inpainting_output_image = gr.Image(
        #                 label="Cleaned Image",
        #                 interactive=False,
        #                 height=600
        #             )
                    
        #             inpainting_status_output = gr.Textbox(
        #                 label="Status",
        #                 value="Ready.",
        #                 interactive=False
        #             )

        #             # ❇️ 다음 단계로 이동하는 버튼 추가
        #             go_to_compositing_button = gr.Button("Go to Compositing →", variant="secondary", elem_id="btn_compositing")

        # # --- [새 탭] 탭 5: Final Compositing ---
        # with gr.TabItem("5. Final Compositing", elem_id="tab_compositing"):
            # gr.Markdown("## ✨ Final Output: Composited Image")
            # with gr.Row():
            #     with gr.Column(scale=1):
            #         gr.Markdown("### Compositing Input")
            #         compositing_preview_image = gr.Image(
            #             label="Cleaned Image from Step 4",
            #             interactive=False,
            #             height=400
            #         )
            #         compositing_run_button = gr.Button("Run Final Compositing", variant="primary")
                    
            #     with gr.Column(scale=2):
            #         gr.Markdown("### Final Result")
            #         final_composited_image = gr.Image(
            #             label="Translated Image",
            #             interactive=False,
            #             height=600
            #         )
            #         final_compositing_status = gr.Textbox(
            #             label="Status",
            #             value="Ready.",
            #             interactive=False
            #         )
    
    
    # ❇️ 버튼 클릭 이벤트 핸들러 연결
    # 1. 'Detect Text' 버튼 클릭 시
    detect_button.click(
        fn=run_detection,
        inputs=[img_input, model_dropdown, iou_slider, conf_slider, st],
        outputs=[detect_output, ocr_input_image, st]
    )

    # 2. 'Run OCR' 버튼 클릭 시
    ocr_button.click(
        fn=run_ocr,
        inputs=[st], 
        outputs=[ocr_output_df, st]
    )

demo.launch(share=False)