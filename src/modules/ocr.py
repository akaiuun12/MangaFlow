import pandas as pd
import gradio as gr

from manga_ocr import MangaOcr
from PIL import Image

ocr = MangaOcr()

def run_ocr(st):
    """주어진 이미지와 바운딩 박스에 대해 OCR을 실행하고 결과를 Dataframe으로 반환합니다."""

    if not st or st.get("detections") is None:
        return gr.update(), st

    img_np, boxes = st.get("input_img"), st.get("detections")
    
    print(f"Running OCR on {len(boxes)} boxes...")
    
    ocr_results_list = []

    for idx, box in enumerate(boxes):
        box_id = f'{idx+1:03d}'
        x1, y1, x2, y2 = map(int, box[0])

        # 좌표를 문자열로 포맷팅 (Dataframe에 넣기 위해)
        coordinates_str = f"({x1}, {y1}, {x2}, {y2})"
        
        cropped_img_np = img_np[y1:y2, x1:x2]
        cropped_pil_img = Image.fromarray(cropped_img_np).convert('RGB')
        
        text_result = ocr(cropped_pil_img).strip()
        
        # ❇️ 결과 리스트에 튜플 형태로 추가
        ocr_results_list.append((box_id, coordinates_str, text_result))

    print("OCR complete.")

    # Dataframe으로 변환하여 반환
    ocr_df = pd.DataFrame(ocr_results_list, 
                          columns=["BOX ID", "COORDINATES", "DETECTED TEXT"])
    new_st = dict(st)
    new_st["ocr_df"] = ocr_df
    
    return gr.update(value=ocr_df), gr.update(value=ocr_df), new_st