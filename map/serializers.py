from rest_framework import serializers
import netCDF4 as nc
import os

class NetCDF_to_CSV_Serializer(serializers.Serializer):
    netcdf_file = serializers.FileField(use_url=True)
    csv_file = serializers.FileField()
    attributes = serializers.ListField(child=serializers.CharField(), allow_empty=False)

    def validate_file(self, value):
        allowed_extensions  = ['.nc, .csv']

        ext = os.path.splitext(value.name)[1].lower()

        if ext not in allowed_extensions:
            raise serializers.ValidationError(f"Unsupported file type: {ext}. Please select a {','.join(allowed_extensions)} file type")
        
        return value
    
    def get_netcdf_file(self, obj):
        if obj.netcdf_file:
            request = self.context.get('request')
            if request is not None:
                return request.build_absolute_uri(obj.netcdf_file.url)
            else:
                # Handle cases where request is not available (e.g., manual serialization)
                return obj.netcdf_file.url
            print(bj.netcdf_file.url)
        return None

class NetCDF_to_SHP_Serializer(serializers.Serializer):
    netcdf_file = serializers.FileField(use_url=True)

class GetAttributeSerializer(serializers.Serializer):
    netcdf_file = serializers.FileField()

class AttributeResponseSerializer(serializers.Serializer):
    attributes = serializers.ListField(child=serializers.CharField())
    
class PlotGraphSerializer(serializers.Serializer):
    PARAMETER_CHOICES = [
        ("temperature_2m_max", "Max Temperature"),
        ("temperature_2m_min", "Min Temperature"),
        ("temperature_2m_mean", "Mean Temperature"),
        ("precipitation_sum", "Precipitation Sum"),
        ("et0_fao_evapotranspiration", "Evapotranspiration"),
    ]

    latitude = serializers.FloatField(help_text="Latitude of the location")
    longitude = serializers.FloatField(help_text="Longitude of the location")
    start_date = serializers.DateField(help_text="Start date in YYYY-MM-DD format", format="%Y-%m-%d", input_formats=["%Y-%m-%d"])
    end_date = serializers.DateField(help_text="End date in YYYY-MM-DD format", format="%Y-%m-%d", input_formats=["%Y-%m-%d"])
    parameters = serializers.MultipleChoiceField(choices=PARAMETER_CHOICES, help_text="Select one or more parameters")

class SPISerializer(serializers.Serializer):
    latitude = serializers.FloatField(help_text="Latitude of the location")
    longitude = serializers.FloatField(help_text="Longitude of the location")
    start_date = serializers.DateField(help_text="Start date in YYYY-MM-DD format", format="%Y-%m-%d", input_formats=["%Y-%m-%d"])
    end_date = serializers.DateField(help_text="End date in YYYY-MM-DD format", format="%Y-%m-%d", input_formats=["%Y-%m-%d"])

class IDWSerializer(serializers.Serializer):
    shapefile = serializers.FileField()

