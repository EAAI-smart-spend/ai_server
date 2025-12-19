
from django.http import HttpRequest
from django.shortcuts import render
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import APIException, ValidationError, NotFound
import joblib

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
    def post(self, request):
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
