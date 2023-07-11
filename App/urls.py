from django.urls import path
from . import views

urlpatterns = [
    
    path('', views.homepage,name = 'homepage'),
    path('pdfUpload/', views.pdfUpload,name = 'pdfUpload'),
    path('textData/', views.textData,name = 'pdfUpload'),

]