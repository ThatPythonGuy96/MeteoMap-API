from rest_framework import serializers
from . import models

class VisitorSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Visitor
        fields = ['ip_address', 'first_seen', 'last_seen', 'request_count']