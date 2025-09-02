from drf_spectacular.utils import extend_schema
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status, permissions
from . import serializers
from pathlib import Path
import json, io, tempfile, random, os, sys, time, rioxarray, matplotlib, tempfile, rasterio, folium
import pandas as pd
from scipy import stats as st
matplotlib.use('Agg')
from shapely.geometry import Point, mapping
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import netCDF4 as nc
from django.http import HttpResponse
from rasterio.transform import from_origin
from folium.plugins import Draw, MousePosition
from folium.raster_layers import ImageOverlay
from rasterio.mask import mask
import xarray as xr
from matplotlib.figure import Figure
from user.views import get_client_ip
from map.service.open_meteo import fetch_open_meteo_data

def download_file(request, filename):
    # # Construct the full file path
    # file_path = os.path.join(settings.MEDIA_ROOT, filename)  # or wherever your file is stored
    
    # if os.path.exists(file_path):
    #     return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=filename)
    # else:
    #     raise Http404("File not found")
    pass
    
def temperory_file(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
        for chunk in file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name
    return tmp_path

@extend_schema(tags=["Map"], description="Get variable keys for a Dataset.")
class GetAttribute(APIView):
    serializer_class = serializers.GetAttributeSerializer

    def post(self, request, *args, **kwargs):
        serializer = serializers.GetAttributeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True) 
        netcdf_file = serializer.validated_data['netcdf_file']
        file_path = temperory_file(netcdf_file)
        dataset = nc.Dataset(file_path)
        attributes = list(dataset.variables.keys())
        dataset.close()
        return Response({'attributes': attributes}, status=status.HTTP_200_OK)

@extend_schema(tags=["Map"], description="Convert a NetCDF(.nc) file to CSV using a selected attribute.")
class netCDF_to_csv(APIView):
    serializer_class = serializers.NetCDF_to_CSV_Serializer
    
    def post(self, request, *args, **kwargs):
        serializer = serializers.NetCDF_to_CSV_Serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        netcdf_file = data['netcdf_file']
        csv_file = data['csv_file']
        file_path = temperory_file(netcdf_file)
        dataset = nc.Dataset(file_path)
        
        attributes = list(dataset.variables.keys())
        
        all_dfs = []

        for variable in attributes:
            if variable not in dataset.variables:
                print(f"⚠️ Skipping {variable} (not found in dataset)")
                continue

            var_data = dataset.variables[variable]

            dims = var_data.get_dims()
            if len(dims) < 3:
                print(f"⚠️ Skipping {variable} (not 3D data)")
                continue

            time_dim, lat_dim, lon_dim = dims[:3]

            # ✅ Extract coordinates
            time_var = dataset.variables[time_dim.name]
            times = nc.num2date(time_var[:], time_var.units)
            latitudes = dataset.variables[lat_dim.name][:]
            longitudes = dataset.variables[lon_dim.name][:]

            # ✅ Create grid
            print(f"⚠️ Creating grid.")
            times_grid, lat_grid, lon_grid = [
                x.flatten() for x in np.meshgrid(times, latitudes, longitudes, indexing="ij")
            ]

            # ✅ Flatten variable data
            var_values = var_data[:].flatten()

            # ✅ Build dataframe for this variable
            print(f"⚠️ Creating Dataframe")
            df_var = pd.DataFrame({
                "time": [t.isoformat() for t in times_grid],
                "latitude": lat_grid,
                "longitude": lon_grid,
                variable: var_values
            })

            all_dfs.append(df_var)

        # ✅ Merge all variables (outer join on time/lat/lon)
        if all_dfs:
            df_final = all_dfs[0]
            for extra_df in all_dfs[1:]:
                df_final = df_final.merge(extra_df, on=["time", "latitude", "longitude"], how="outer")

            # ✅ Save to CSV
            print(f"⚠️ Wrtitng data to {csv_file}")
            df_final.to_csv(os.path.join(csv_file), index=False)
        else:
            return Response("No data extracted.", status=status.HTTP)
        dataset.close()
        return Response({'Success': "Convertion complete."}, status=status.HTTP_200_OK)

@extend_schema(tags=["Map"], description="Convert a NetCDF(.nc) file to Shapefile(.shp) using a selected attribute.")
class netCDF_to_Shp(APIView):
    serializer_class = serializers.NetCDF_to_SHP_Serializer

@extend_schema(tags=["Map"], description="Plot Weather variables.")
class PlotGraph(APIView):
    serializer_class = serializers.PlotGraphSerializer
    
    def post(self, request):
        serializer = serializers.PlotGraphSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        serializer.is_valid(raise_exception=True) 
        data = serializer.validated_data
        
        df = fetch_open_meteo_data(
            latitude=data['latitude'],
            longitude=data['longitude'],
            start_date=data['start_date'],
            end_date=data['end_date'],
            parameters=data['parameters']
        )

        if df.empty:
            return Response({"error": "No data returned"}, status=status.HTTP_204_NO_CONTENT)

        # Plotting
        plt.figure(figsize=(10, 6))
        for param in data['parameters']:
            if param in df:
                plt.plot(df['date'], df[param], label=param)
            else:
                return Response({"error": f"No data for parameter: {param}"}, status=400)

        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title(f"Weather Data ({data['start_date']} to {data['end_date']})")
        plt.xticks(rotation=45)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.tight_layout(rect=[0, 0, 1, 1])  

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        return HttpResponse(buffer, content_type="image/png")
        # return Response({"data": weather_data}, status=status.HTTP_502_BAD_GATEWAY)

@extend_schema(tags=["Map"], description="Calculate Standard Precipitation Index(SPI).")
class SPI(APIView):
    serializer_class = serializers.SPISerializer

    def post(self, request):
        serializer = serializers.SPISerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        serializer.is_valid(raise_exception=True) 
        data = serializer.validated_data

        df = fetch_open_meteo_data(
            latitude=data['latitude'],
            longitude=data['longitude'],
            start_date=data['start_date'],
            end_date=data['end_date'],
            parameters=["precipitation_sum"]
        )
        df['precipitation_sum'] = df['precipitation_sum'].fillna(0)
        
        if df.empty:
            return Response({"error": "No data returned"}, status=status.HTTP_204_NO_CONTENT)

        def spi(ds, thresh):
            #Rolling Mean
            ds_ma = ds.rolling(window=thresh, min_periods=1, center=False).mean()

            # Replace zeros in rolling mean with a small value to avoid log issues
            epsilon = 1e-6  # Small value to substitute for zeros
            ds_ma_safe = ds_ma.replace(0, epsilon)
            
            #Nutural log of M. A.
            ds_In = np.log(ds_ma_safe)

            ds_mu = ds_ma.mean()

            #summation of log of M. A.
            ds_sum = ds_In.sum()

            #computing for gamma distribution
            n = len(ds_In)
            A = np.log(ds_mu) - (ds_sum/n)
            alpha = (1 / (4 * A) ) * (1 + (1 + (4 * A) / 3) ** 0.5)
            beta = ds_mu/alpha

            #gamm distribution (CDF)
            gamma = st.gamma.cdf(ds_ma, a=alpha, scale=beta)

            #SPI (Inverse of CDF)
            norm_spi = st.norm.ppf(gamma, loc=0, scale=1)

            return norm_spi

        times = [3, 6, 9, 12]
        for i in times:
            x = spi(df['precipitation_sum'], i)
            df['spi_'+str(i)] = x

        fig, axes = plt.subplots(nrows=4, figsize=(15, 10))
        plt.subplots_adjust(hspace=0.15)

        for i, ax in enumerate(axes):
            col_scheme = np.where(df['spi_'+str(times[i])]>0, 'b', 'r')

            ax.bar(df.index, df['spi_'+str(times[i])], width=25, align='center', color=col_scheme, label='SPI ' + str(times[i]))
            ax.axhline(y=0, color='k')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.legend(loc='upper right')
            ax.set_yticks(range(-3,4), range(-3,4))
            ax.set_ylabel('SPI ' + str(times[i]), fontsize=12)

            if i < len(times)-1:
                ax.set_xticks([], [])

        axes[-1].set_xlabel('Year', fontsize=12)

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        return HttpResponse(buffer, content_type="image/png")
    
@extend_schema(tags=["Map"], description="Calculate Standard Precipitation Evaporationn Index(SPEI).")
class SPEI(APIView):
    serializer_class = serializers.SPISerializer

    def post(self, request):
        serializer = serializers.SPISerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        serializer.is_valid(raise_exception=True) 
        data = serializer.validated_data

        df = fetch_open_meteo_data(
            latitude=data['latitude'],
            longitude=data['longitude'],
            start_date=data['start_date'],
            end_date=data['end_date'],
            parameters=["precipitation_sum", "et0_fao_evapotranspiration"]
        )
        df['precipitation_sum'] = df['precipitation_sum'].fillna(0)
        df["et0_fao_evapotranspiration"] = df["et0_fao_evapotranspiration"].fillna(0)

        if df.empty:
            return Response({"error": "No data returned"}, status=status.HTTP_204_NO_CONTENT)

        def spei(preci, pet, window):
            """
            Calculate the Standardized Precipitation-Evapotranspiration Index (SPEI).

            Parameters:
            - precipitation: pandas Series, precipitation data.
            - pet: pandas Series, potential evapotranspiration data.
            - window: int, rolling window size in months.

            Returns:
            - spei: pandas Series, SPEI values.
            """
            # Calculate the water balance (D = P - PET)
            deficit = preci - pet

            # Rolling sum over the specified window
            rolling_deficit = deficit.rolling(window=window, min_periods=1).sum()

            # Fit Gamma distribution to rolling sums
            epsilon = 1e-6  # Small value to substitute for zeros
            rolling_deficit_safe = rolling_deficit.replace(0, epsilon)  # Avoid log issues with zeros
            
            gamma_cdf = st.gamma.cdf(
                rolling_deficit_safe, a=2, scale=1
            )  # Example params, adjust as needed

            # Clip values to avoid -inf/inf
            gamma_cdf = np.clip(gamma_cdf, 1e-6, 1 - 1e-6)

            # Convert CDF to standard normal distribution (z-scores)
            norm_spei = st.norm.ppf(gamma_cdf)

            return pd.Series(norm_spei, index=preci.index)
        
        times = [3, 6, 9, 12]
        for i in times:
            x = spei(df['precipitation_sum'], df["et0_fao_evapotranspiration"], i)
            df['spei_'+str(i)] = x
        
        fig, axes = plt.subplots(nrows=4, figsize=(15, 10))
        plt.subplots_adjust(hspace=0.15)
        
        for i, ax in enumerate(axes):
            col_scheme = np.where(df['spei_' + str(times[i])] > 0, 'b', 'r')
            # Format x-axis (years)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))

            ax.bar(df.index, df['spei_'+str(times[i])], width=20, align='center', color=col_scheme, label='SPEI ' + str(times[i]))
            ax.axhline(y=0, color='k')
            
            ax.legend(loc='upper right')
            # Y-axis formatting
            ax.set_yticks(range(-6, 7))
            ax.set_yticklabels(range(-6, 7))
            ax.set_ylabel('SPEI ' + str(times[i]), fontsize=12)

            if i < len(times) - 1:
                ax.set_xticklabels([])

        axes[-1].set_xlabel('Year', fontsize=12)

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        return HttpResponse(buffer, content_type="image/png")
        # return Response(df, status=status.HTTP_200_OK)

@extend_schema(tags=["Map"], description='Perform Idw.')
class IDW(APIView):
    serializer_class = serializers.IDWSerializer