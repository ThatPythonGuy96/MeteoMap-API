from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from .models import Visitor

class IPAuthentication(BaseAuthentication):
    def authenticated(self, request):
        ip = request.META.get('HTTP_X_FORWARDED_FOR') or request.META.get('REMOTE_ADDR')
        if not ip:
            raise AuthenticationFailed('No IP address found.')
        visitor, created = Visitor.objects.get_or_create(ip_address=ip)
        visitor.request_count += 1
        visitor.save()
        return (visitor, None)