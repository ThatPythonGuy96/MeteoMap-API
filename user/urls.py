from django.urls import path
from . import views

app_name='user'
urlpatterns = [
    path('get-ip', views.get_client_ip.as_view(), name='get_client_ip'),
    path('me', views.VisitorView.as_view(), name='visitor_profile'),
]