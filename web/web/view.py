from django.shortcuts import render
import json


def q(request):
    context = {}
    context['test'] = 'hello'

    return render(request, 'index.html', {'context': json.dumps(context)})