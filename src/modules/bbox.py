import os
import numpy as np
import pandas as pd
import cv2
import gradio as gr

from PIL import Image
from imgutils.generic import YOLOModel

# Hugging Face Repository for the model
HF_REPO = "deepghs/manga109_yolo"

def draw_boxes_on_image(image_np, detections, line_thickness=2):
    """탐지된 바운딩 박스를 이미지 NumPy 배열에 그립니다."""
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # OpenCV는 BGR을 사용
    
    # 클래스별 색상 정의 (4클래스 가정: text, panel, figure, bubble)

    for (x0, y0, x1, y1), label, score in detections:
        # text 클래스만 그리기
        if label != 'text':
            continue

        color = (0, 255, 0)  # 기본: 초록색
        
        # 1. 바운딩 박스 그리기
        cv2.rectangle(
            img=image,
            pt1=(int(x0), int(y0)),
            pt2=(int(x1), int(y1)),
            color=color,
            thickness=line_thickness,
            lineType=cv2.LINE_AA
        )
        
        # 2. 라벨 텍스트 배경 및 점수 표시
        label_text = f"{label} {score:.2f}"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        
        # 텍스트 배경
        cv2.rectangle(
            img=image,
            pt1=(int(x0), int(y0) - h - 5),
            pt2=(int(x0) + w, int(y0)),
            color=color,
            thickness=-1 # 채우기
        )
        
        # 텍스트
        cv2.putText(
            img=image,
            text=label_text,
            org=(int(x0), int(y0) - 4),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(255, 255, 255), # 흰색 텍스트
            thickness=2,
            lineType=cv2.LINE_AA
        )
        
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB로 다시 변환 후 반환

def run_detection(img_np: np.ndarray, model_name: str, iou: float=0.7, conf: float=0.25, st: dict = None) -> np.ndarray:
    """
    텍스트 감지를 실행하고 바운딩 박스가 그려진 이미지를 반환합니다.
    """
    if img_np is None:
        raise gr.Error("이미지를 먼저 업로드해주세요.")
    
    # 1. 모델 로드
    yolo_model = YOLOModel(HF_REPO)
    
    # 2. NumPy 배열을 PIL Image로 변환
    img_pil = Image.fromarray(img_np.astype('uint8'), 'RGB')
    
    # 3. 객체 탐지 실행
    try:
        # conf_threshold=None: repo의 기본값 사용 (일반적으로 0.25)
        # iou_threshold=0.7: NMS 임계값 (원래 코드에서 지정)
        detections = yolo_model.predict(
            image=img_pil,
            model_name=model_name,
            conf_threshold=conf,
            iou_threshold=iou,
            allow_dynamic=True
        )

        # 'text' 클래스만 필터링
        detections = [
            (box, label, score) for box, label, score in detections if label == 'text'
        ]

    except Exception as e:
        raise gr.Error(f"객체 탐지 중 오류가 발생했습니다: {str(e)}")
    
    # 4. 탐지 결과를 NumPy 이미지에 그리기
    detected_image_np = draw_boxes_on_image(img_np, detections)
    
    print(f"✅ 총 {len(detections)}개의 객체 탐지 완료.")

    # 5. gradio 출력 반환
    new_st = dict(st)
    new_st["input_img"] = img_np
    new_st["detections"] = detections
    new_st["detect_img"] = detected_image_np
    
    return gr.update(value=detected_image_np), gr.update(value=detected_image_np), gr.update(value=detected_image_np), new_st