from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions, status
from . import serializers

class get_client_ip(APIView):
    def get(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        print(ip)
        return Response(ip, status=status.HTTP_200_OK)

class VisitorView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        visitor = request.user
        serializer = serializers.VisitorSerializer(visitor)
        return Response(serializer.data, status=status.HTTP_200_OK)