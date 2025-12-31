
from PIL import Image
import io
import easyocr
from paddleocr import PaddleOCR
import numpy as np
from io import BytesIO
from rest_framework import status
import os


os.environ["CUDA_VISIBLE_DEVICES"] = ""

ocr_instance = PaddleOCR(use_angle_cls=True, lang='ch')

reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)
"""
reader = easyocr.Reader(
    lang_list=['ch_tra'],     
    recog_network='chinese',  
    gpu=False                   
)
"""
def perform_ocr_by_easyocr(image_file) -> dict:
    
    try:
        image = Image.open(io.BytesIO(image_file.read()))
    except Exception as e:
        raise ValueError(f"Invalid image: {str(e)}")

    result = reader.readtext(image, detail=1)  
    
    full_details = [
        {"text": text, "confidence": round(conf, 2), "box": box}
        for box, text, conf in result
    ]

    texts_non_empty = [item["text"] for item in full_details if item["text"].strip() != ""]

    # print(texts_non_empty)

    ocr_result_text = " ".join(texts_non_empty)

    print(ocr_result_text)

    return {
        "ocr_result_array": texts_non_empty,
        "ocr_result_text": ocr_result_text,
        "total_texts_found": len(full_details),

    }



def perform_ocr_by_paddle(image_file):
    
    if not image_file:
        raise ValueError("No image file provided")

    if not image_file.content_type.startswith('image/'):
        raise ValueError("Uploaded file is not a valid image")

    image_bytes = BytesIO(image_file.read())

    try:
        pil_image = Image.open(image_bytes).convert('RGB')
        img_array = np.array(pil_image)
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image: {str(e)}")

    result = ocr_instance.ocr(img_array)

    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        res_dict = result[0]
    elif isinstance(result, dict):
        res_dict = result
    else:
        raise ValueError("No text detected or unexpected result format")

    if 'rec_texts' not in res_dict or not res_dict['rec_texts']:
        raise ValueError("No text detected in the image")

    texts = res_dict['rec_texts']  

   
    return {
        "extracted_text": "\n".join(texts),       
        "ocr_result_text": " ".join(texts),  
        "ocr_result_array": texts,                       
        "total_texts_found": len(texts)
    }