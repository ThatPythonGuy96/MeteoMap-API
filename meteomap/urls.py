from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.IndexView.as_view(), name='index'),
    path('map/', include('map.urls', namespace="map")),
    path('user/', include('user.urls', namespace="user")),
    
]
