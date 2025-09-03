#! /usr/bin/env bash

set -o errexit   #  exit on error

pip install -r requirements.txt

python manage.py collectstatic --no-input

python manage.py migrate

# Create superuser
# python manage.py shell << END
# from django.contrib.auth import get_user_model
# Account = get_user_model()
# import os

# email = os.getenv("DJANGO_SUPERUSER_EMAIL", "meteo@gmail.com")
# password = os.getenv("DJANGO_SUPERUSER_PASSWORD", "bavtwany")

# if not Account.objects.filter(email=email).exists():
#     Account.objects.create_superuser(email=email, password=password)
#     print("Superuser created")
# else:
#     print("Superuser already exists")
# END
