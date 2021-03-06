"""web URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from . import view

urlpatterns = [
    url(r'^$', view.q),
    url(r'^search_mat/', view.search_mat),
    # url(r'^max_/', view.max_),
    url(r'^q/', view.q),
    url(r'^k/', view.k),
    url(r'^v/', view.v),
    url(r'^horizon_softmax/', view.softmax),
    url(r'^search_soft/', view.search_soft)
]
