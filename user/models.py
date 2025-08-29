from django.db import models

class Visitor(models.Model):
    ip_address = models.GenericIPAddressField(unique=True)
    first_seen = models.DateTimeField(auto_now_add=True)
    last_seen = models.DateTimeField(auto_now=True)
    request_count = models.PositiveIntegerField(default=0)

    def __str__(self):
        return self.ip_address
