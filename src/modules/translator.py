import os
import time
import pandas as pd

from google import genai
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

TRANSLATION_SEPARATOR = " |||SEP||| " # 배치 번역을 위한 고유 구분자 설정
SYSTEM_PROMPT = (
    "너는 전문 일본어→한국어 만화 번역가다. "
    "주어진 입력을 자연스럽게 번역하여 출력해라."
    f"입력은 {TRANSLATION_SEPARATOR} 로 구분된 여러 항목이며, 출력도 같은 구분자 순서로 맞춰야 한다. "
    "대사체로 자연스럽게 번역하고, 과도한 의역이나 설명은 피한다. "
    f"입력 항목의 수가 {TRANSLATION_SEPARATOR}로 구분된 출력 항목의 수와 일치해야 한다. "
    "번역할 내용이 없는 항목(' ' 또는 ' (OCR Error)')은 빈 문자열로 출력한다. "
    "출력은 번역 결과만 포함해야 하며, 어떤 추가적인 설명이나 문장도 포함하면 안된다."
)

def actual_batch_translate(texts_combined, model_name):
    """
    Gemini API를 호출하여 배치 번역을 수행하는 함수
    """

    if "gemini" in model_name.lower() and client:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=texts_combined,
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0         # 결과의 일관성 유지를 위해 Creativity 낮게 설정
                )
            )
            return response.text.strip()    # API 응답에서 텍스트만 추출하여 반환
            
        except Exception as e:
            print(f"Gemini API Error: {e}")
            
            # API 오류 발생 시, 원본 텍스트 수만큼 오류 메시지 생성 후 반환
            original_texts = texts_combined.split(TRANSLATION_SEPARATOR)
            error_message = f"(API 호출 오류: {e.__class__.__name__})"
            return TRANSLATION_SEPARATOR.join([error_message] * len(original_texts))

    else:
        # API 키가 없거나 다른 API 선택 시 더미 처리
        print("--- Fallback Simulation (No API Key) ---")
        time.sleep(1.0)
        original_texts = texts_combined.split(TRANSLATION_SEPARATOR)
        translated_results = []
        for text in original_texts:
            if not text.strip() or "(OCR Error)" in text:
                translated_results.append("")
            else:
                prefix = "번역됨(Fallback Simul): "
                translated_results.append(prefix + text[:20].strip() + "...")
        return TRANSLATION_SEPARATOR.join(translated_results)

def run_translation(st, api_name):
    """
    OCR Dataframe을 입력받아 배치 번역을 수행하고 Dataframe을 yield로 반환
    """
    ocr_df = st.get("ocr_df")
    
    if ocr_df.empty:
        yield pd.DataFrame(columns=["ORIGINAL TEXT", "TRANSLATED TEXT"])

    # 1. 원본 텍스트 추출 및 결합
    texts_to_translate = ocr_df['DETECTED TEXT'].fillna('').apply(lambda x: x if x.strip() else ' ')
    texts_combined = TRANSLATION_SEPARATOR.join(texts_to_translate.tolist())
    
    if not texts_combined.strip():
        print("No detectable text to translate.")
        yield pd.DataFrame(columns=["ORIGINAL TEXT", "TRANSLATED TEXT"])

    # 2. API 호출
    yield pd.DataFrame({'ORIGINAL TEXT': ocr_df['DETECTED TEXT'], 'TRANSLATED TEXT': "번역 요청 중..."}) # 진행 중 표시
    
    try:
        translated_combined = actual_batch_translate(texts_combined, api_name) # ❇️ 실제 API 호출 함수 사용
    except Exception as e:
        print(f"Critical Translation Error: {e}")
        error_df = pd.DataFrame({'ORIGINAL TEXT': ocr_df['DETECTED TEXT'], 'TRANSLATED TEXT': "(치명적인 오류 발생)"})
        yield error_df[['ORIGINAL TEXT', 'TRANSLATED TEXT']]
        return

    # 3. 결과 분리 및 매핑
    translated_texts = translated_combined.split(TRANSLATION_SEPARATOR)
    
    # ⚠️ API가 분리자를 무시하고 추가 문장을 넣는 등 결과가 오염될 수 있으므로, 예상되는 개수보다 많거나 적을 때 오류 처리
    if len(translated_texts) != len(ocr_df):
        print(f"Warning: Batch output count ({len(translated_texts)}) does not match input count ({len(ocr_df)}). Raw Output: {translated_combined[:100]}...")
        
        # 번역 결과가 예상과 다를 때 (LLM의 잘못된 출력)
        translated_texts = [f"(결과 개수 불일치 오류) {t.strip()[:30]}" for t in translated_texts]
        
        # 원본 개수에 맞추기 위해 잘라내거나 채우기
        if len(translated_texts) > len(ocr_df):
            translated_texts = translated_texts[:len(ocr_df)]
        elif len(translated_texts) < len(ocr_df):
            translated_texts += [f"(누락 항목)"] * (len(ocr_df) - len(translated_texts))

    # 4. Dataframe 구성 및 반환
    translate_df = pd.DataFrame({
        'ORIGINAL TEXT': ocr_df['DETECTED TEXT'],
        'TRANSLATED TEXT': translated_texts
    })
    
    yield translate_df
