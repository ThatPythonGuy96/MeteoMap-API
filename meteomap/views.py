from django.shortcuts import render, redirect
from django.views import generic

class IndexView(generic.View):
    template_name = 'index.html'

    def get(self, request):
        return render(request, self.template_name)