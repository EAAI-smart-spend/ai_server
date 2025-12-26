# ocr_utils.py or inside your views.py / utils.py

from PIL import Image
import io
import easyocr

# Initialize the reader ONCE at module level (best practice for performance)
reader = easyocr.Reader(
    lang_list=['ch_tra'],      # Traditional Chinese
    recog_network='chinese',   # Your custom model
    gpu=True                   # Set False if no GPU available
)

def perform_ocr(image_file) -> dict:
    """
    Takes an uploaded image file (from request.FILES) and returns OCR results.
    
    Returns:
        {
            "ocr_result": [list of text strings (non-empty)],
            "total_texts_found": int,
            "full_details": [optional: list of dicts with text, conf, box]
        }
    """
    try:
        image = Image.open(io.BytesIO(image_file.read()))
    except Exception as e:
        raise ValueError(f"Invalid image: {str(e)}")

    result = reader.readtext(image, detail=1)  

    # Extract details
    full_details = [
        {"text": text, "confidence": round(conf, 2), "box": box}
        for box, text, conf in result
    ]

    # Extract only non-empty texts
    texts_non_empty = [item["text"] for item in full_details if item["text"].strip() != ""]

    print(texts_non_empty)

    ocr_result_text = " ".join(texts_non_empty)

    print(ocr_result_text)

    return {
        "ocr_result_array": texts_non_empty,
        "ocr_result_text": ocr_result_text,
        "total_texts_found": len(full_details),

        # "full_details": full_details  # Optional: include if you want boxes/confidence
    }