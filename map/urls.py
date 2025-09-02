from django.urls import path
from . import views
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView

app_name="map"
urlpatterns = [
    path('schema/', SpectacularAPIView.as_view(), name='schema'),
    path('', SpectacularSwaggerView.as_view(url_name='map:schema'), name='swagger-ui'),
    path('schema/redoc/', SpectacularRedocView.as_view(url_name='map:schema'), name='redoc'),

    path('nedcdf-to-cdv/', views.netCDF_to_csv.as_view(), name='netcdf-to-csv'),
    path('nedcdf-to-shp/', views.netCDF_to_Shp.as_view(), name='netcdf-to-shp'),
    path('get-attribute', views.GetAttribute.as_view(), name='get-attribute'),
    path('plot-graph/', views.PlotGraph.as_view(), name='plot-graph'),
    path('spi/', views.SPI.as_view(), name='calculate_spi'),
    path('spei/', views.SPEI.as_view(), name='calculate_spei'),
]
