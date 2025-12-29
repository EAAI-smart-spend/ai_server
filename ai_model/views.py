
from django.http import HttpRequest
from django.shortcuts import render
from ai_model.predict import extract_total_price
from ai_model.utils import perform_ocr_by_easyocr, perform_ocr_by_paddle
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
    0: "購物",
    1: "交通",
    2: "其他",
    3: "娛樂",
    4: "食飯"
}

CATEGORY_MAP_FINAL = {
    "購物": "Groceries (雜貨/超市購物)",
    "交通":"Transportation (交通)",
    "其他":"Utilities (公用事業)",
    "娛樂":"Entertainment (娛樂)",
    "食飯":"Food & Drinks (食物與飲料)"
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


class getImgByPaddle(APIView):
    def post(self, request):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response(
                {"error": "No image provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # try:
        ocr_data = perform_ocr_by_paddle(image_file)
        return Response(ocr_data)
        # except ValueError as e:
        #     return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        # except Exception as e:
        #     return Response(
        #         {"error": "OCR processing failed"},
        #         status=status.HTTP_500_INTERNAL_SERVER_ERROR
        #     )


class GetOcrResult(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response(
                {"error": "No image provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            ocr_data = perform_ocr_by_easyocr(image_file)
            return Response(ocr_data)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(
                {"error": "OCR processing failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        

class GetOcrResultCategorizer(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response(
                {"error": "No image provided"},
                status=status.HTTP_400_BAD_REQUEST
            )
        ocr_engine = request.data.get('ocr_engine', 'easyocr').lower()
        
        if ocr_engine == 'easyocr':
            ocr_data = perform_ocr_by_easyocr(image_file)
        elif ocr_engine == 'paddle':
            ocr_data = perform_ocr_by_paddle(image_file)


        text = ocr_data['ocr_result_text']

        array_text = ocr_data['ocr_result_array']

        getAmount = extract_total_price(array_text)
        
        if not text:
            return Response(
                {"error": "Please provide 'text' in the request body."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            vec = TFIDF_VECTORIZER.transform([text])

            predictions = {
                "Naive Bayes": CATEGORY_MAP_FINAL[NB_MODEL.predict(vec)[0]],
                "SVM": CATEGORY_MAP_FINAL[SVM_MODEL.predict(vec)[0]],
                "Decision Tree": CATEGORY_MAP_FINAL[DT_MODEL.predict(vec)[0]],
                "KNN": CATEGORY_MAP_FINAL[KNN_MODEL.predict(vec)[0]],
                "Logistic Regression": CATEGORY_MAP_FINAL[LOGISTIC_MODEL.predict(vec)[0]],
            }

            return Response({
                "input": text,
                "TotalAmount":getAmount,
                "predictions": predictions
            })

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            

