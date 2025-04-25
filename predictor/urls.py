from django.urls import path
from .views import HomeView, PredictView, DownloadReportView

urlpatterns = [
    path('', HomeView.as_view(), name='home'),
    path('predict/', PredictView.as_view(), name='predict'),
    path('download-report/', DownloadReportView.as_view(), name='download_report'),
]