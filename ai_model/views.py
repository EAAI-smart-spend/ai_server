
from django.http import HttpRequest
from django.shortcuts import render
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import APIException, ValidationError, NotFound
from rest_framework.parsers import MultiPartParser, FormParser
import joblib
import easyocr
from PIL import Image
import io


try:
    TFIDF_VECTORIZER = joblib.load('saved_models/tfidf_vectorizer.pkl')
    NB_MODEL = joblib.load('saved_models/nb_model.pkl')
    SVM_MODEL = joblib.load('saved_models/svm_model.pkl')
    DT_MODEL = joblib.load('saved_models/dt_model.pkl')
    KNN_MODEL = joblib.load('saved_models/knn_model.pkl')
    LOGISTIC_MODEL = joblib.load('saved_models/logistic_model.pkl')
except Exception as e:
    raise ImportError(f"Failed to load models: {e}. Make sure saved_models/ folder exists.")

CATEGORY_MAP = {
    0: "Groceries (雜貨/超市購物)",
    1: "Transportation (交通)",
    2: "Utilities (公用事業)",
    3: "Entertainment (娛樂)",
    4: "Food & Drinks (食物與飲料)"
}


# Create your views here.
class TestCall (APIView): 

    def get(self, req: HttpRequest):
        print("TEST")
        return Response({'status': 'SUCCESS','response': 'test'})
    
class ExpenseCategorizerView(APIView):
    def get(self, request):
        """
        Expect JSON: {"text": "Sushi and beer with friends HKD120"}
        Returns predictions from all models.
        """
        text = request.data.get('text', '').strip()
        
        if not text:
            return Response(
                {"error": "Please provide 'text' in the request body."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Vectorize the input text
            vec = TFIDF_VECTORIZER.transform([text])

            # Get predictions
            predictions = {
                "Naive Bayes": CATEGORY_MAP[NB_MODEL.predict(vec)[0]],
                "SVM": CATEGORY_MAP[SVM_MODEL.predict(vec)[0]],
                "Decision Tree": CATEGORY_MAP[DT_MODEL.predict(vec)[0]],
                "KNN": CATEGORY_MAP[KNN_MODEL.predict(vec)[0]],
                "Logistic Regression": CATEGORY_MAP[LOGISTIC_MODEL.predict(vec)[0]],
            }

            return Response({
                "input": text,
                "predictions": predictions
            })

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class GetOcrResult(APIView):
    parser_classes = (MultiPartParser, FormParser)  # Important!

    def post(self, request):  # Usually OCR is done via POST, not GET
        image_file = request.FILES.get('image')  # 'image' is the key name from frontend

        if not image_file:
            return Response(
                {"error": "No image provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Now you have the image file object
        # You can access: image_file.name, image_file.size, image_file.content_type

        # Example: save temporarily or process directly
        # content = image_file.read()  # Read bytes

        # TODO: Pass image_file to your OCR function (e.g., pytesseract, easyocr, etc.)

        return Response({"message": "Image received successfully", "filename": image_file.name})

    # Optional: keep GET for health check
    def get(self, request):
        return Response({"message": "Send a POST request with an image file"})
    

class GetOcrResult(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize reader once (loads model into memory)
        self.reader = easyocr.Reader(
            lang_list=['ch_sim'],  # Or ['en'] if mixed
            recog_network='chinese',  # Your custom model name
            gpu=True  # Set False if no GPU
        )

    def post(self, request):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Read image into PIL (EasyOCR accepts file paths, bytes, or PIL)
        try:
            image = Image.open(io.BytesIO(image_file.read()))
        except Exception as e:
            return Response({"error": "Invalid image"}, status=400)

        # Perform OCR
        result = self.reader.readtext(image, detail=1)  # detail=1 for boxes + text + confidence

        # Format output
        ocr_texts = [
            {"text": text, "confidence": round(conf, 2), "box": box}
            for (box, text, conf) in result
        ]

        return Response({
            "ocr_result": ocr_texts,
            "total_texts_found": len(ocr_texts)
        })