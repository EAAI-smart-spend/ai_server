from paddleocr import PaddleOCR

# Initialize once (downloads models on first run)
ocr = PaddleOCR(use_textline_orientation=True, lang='ch') 

# Extract text from image file path or numpy array/PIL image
result = ocr.ocr('image.jpg')

# Process results
for line in result:
    for word_info in line:
        print(word_info[1][0])  # The recognized text
        print(word_info[1][1])  # Confidence score