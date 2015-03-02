"""
COPYRIGHT 2010 RPS ASA

This file is part of SCI-WMS.

    SCI-WMS is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SCI-WMS is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SCI-WMS.  If not, see <http://www.gnu.org/licenses/>.

This file demonstrates writing tests using the unittest module. These will pass
when you run "manage.py test".

Replace this with more appropriate tests for your application.
"""

import os
from time import sleep

from django.test import TestCase
from django.conf import settings
from django.contrib.auth.models import User
import django.contrib.auth.hashers as hashpass

from sciwms.apps.wms.models import Dataset, Group, Server

resource_path = os.path.join(settings.PROJECT_ROOT, 'apps', 'wms', 'resources')


def add_server():
    s = Server.objects.create()
    s.save()


def add_group():
    g = Group.objects.create(name='MyTestGroup',)
    g.save()


def add_dataset(filename):
    add_group()
    d = Dataset.objects.create(uri                   = os.path.join(resource_path, filename),
                               name                  = "test",
                               title                 = "Test dataset",
                               abstract              = "Test data set for sci-wms tests.",
                               display_all_timesteps = False,
                               keep_up_to_date       = False,)
    d.update_cache(force=True)
    d.save()
    return d


def add_user():
    u = User(username="testuser",
             first_name="test",
             last_name="user",
             email="test@yser.comn",
             password=hashpass.make_password("test"),
             is_active=True,
             is_superuser=True,
            )
    u.save()


class SimpleTest(TestCase):
    def test_index(self):
        add_server()
        response = self.client.get('/index.html')
        self.assertEqual(response.status_code, 200)

    def test_post_add(self):
        params = {  "uri"       : os.path.join(resource_path, "201220109.nc"),
                    "id"        : "test",
                    "title"     : "test",
                    "abstract"  : "my test dataset",
                    "update"    : "True",
                    "groups"    : ""
             }
        response = self.client.post("/wms/add_dataset", params)
        self.assertEqual(response.status_code, 200)


class TestUgrid(TestCase):

    def setUp(self):
        add_server()
        add_group()
        add_user()
        self.dataset = add_dataset("201220109.nc")

    def tearDown(self):
        self.dataset.clear_cache()

    def test_web_remove(self):
        response = self.client.get('/wms/remove_dataset/?id=test&username=testuser&password=test')
        self.assertEqual(response.status_code, 200)

    def test_facets(self):
        response = self.client.get('/wms/datasets/test/?LAYERS=u%2Cv&TRANSPARENT=TRUE&STYLES=facets_average_jet_None_None_cell_False&TIME=&ELEVATION=0&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fpng&SRS=EPSG%3A3857&BBOX=-8543030.3273202,5492519.0747705,-8401010.3287862,5542356.0172055&WIDTH=929&HEIGHT=326')
        self.assertEqual(response.status_code, 200)

    def test_pcolor(self):
        response = self.client.get('/wms/datasets/test/?LAYERS=u%2Cv&TRANSPARENT=TRUE&STYLES=pcolor_average_jet_None_None_cell_False&TIME=&ELEVATION=0&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fpng&SRS=EPSG%3A3857&BBOX=-8543030.3273202,5492519.0747705,-8401010.3287862,5542356.0172055&WIDTH=929&HEIGHT=326')
        self.assertEqual(response.status_code, 200)

    def test_contours(self):
        response = self.client.get('/wms/datasets/test/?LAYERS=u%2Cv&TRANSPARENT=TRUE&STYLES=contours_average_jet_None_None_cell_False&TIME=&ELEVATION=0&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fpng&SRS=EPSG%3A3857&BBOX=-8543030.3273202,5492519.0747705,-8401010.3287862,5542356.0172055&WIDTH=929&HEIGHT=326')
        self.assertEqual(response.status_code, 200)

    def test_filledcontours(self):
        response = self.client.get('/wms/datasets/test/?LAYERS=u%2Cv&TRANSPARENT=TRUE&STYLES=filledcontours_average_jet_None_None_cell_False&TIME=&ELEVATION=0&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fpng&SRS=EPSG%3A3857&BBOX=-8543030.3273202,5492519.0747705,-8401010.3287862,5542356.0172055&WIDTH=929&HEIGHT=326')
        self.assertEqual(response.status_code, 200)

    def test_vectors(self):
        response = self.client.get('/wms/datasets/test/?LAYERS=u%2Cv&TRANSPARENT=TRUE&STYLES=vectors_average_jet_None_None_cell_False&TIME=&ELEVATION=0&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fpng&SRS=EPSG%3A3857&BBOX=-8543030.3273202,5492519.0747705,-8401010.3287862,5542356.0172055&WIDTH=929&HEIGHT=326')
        self.assertEqual(response.status_code, 200)

    def test_getLegend(self):
        pass

    def test_getCaps(self):
        response = self.client.get('/wms/datasets/test/?REQUEST=GetCapabilities')
        self.assertEqual(response.status_code, 200)


class TestCgrid(TestCase):

    def setUp(self):
        add_server()
        add_group()
        add_user()
        self.dataset = add_dataset("nasa_scb20111015.nc")

    def tearDown(self):
        self.dataset.clear_cache()

    def test_web_remove(self):
        response = self.client.get('/wms/remove_dataset/?id=test&username=testuser&password=test')
        self.assertEqual(response.status_code, 200)

    def test_pcolor(self):
        response = self.client.get('/wms/datasets/test/?LAYERS=u%2Cv&TRANSPARENT=TRUE&STYLES=pcolor_average_jet_None_None_cell_False&TIME=&ELEVATION=0&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fpng&SRS=EPSG%3A3857&BBOX=-8543030.3273202,5492519.0747705,-8401010.3287862,5542356.0172055&WIDTH=929&HEIGHT=326')
        self.assertEqual(response.status_code, 200)

    def test_contours(self):
        response = self.client.get('/wms/datasets/test/?LAYERS=u%2Cv&TRANSPARENT=TRUE&STYLES=contours_average_jet_None_None_cell_False&TIME=&ELEVATION=0&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fpng&SRS=EPSG%3A3857&BBOX=-8543030.3273202,5492519.0747705,-8401010.3287862,5542356.0172055&WIDTH=929&HEIGHT=326')
        self.assertEqual(response.status_code, 200)

    def test_filledcontours(self):
        response = self.client.get('/wms/datasets/test/?LAYERS=u%2Cv&TRANSPARENT=TRUE&STYLES=filledcontours_average_jet_None_None_cell_False&TIME=&ELEVATION=0&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fpng&SRS=EPSG%3A3857&BBOX=-8543030.3273202,5492519.0747705,-8401010.3287862,5542356.0172055&WIDTH=929&HEIGHT=326')
        self.assertEqual(response.status_code, 200)

    def test_vectors(self):
        response = self.client.get('/wms/datasets/test/?LAYERS=u%2Cv&TRANSPARENT=TRUE&STYLES=vectors_average_jet_None_None_cell_False&TIME=&ELEVATION=0&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fpng&SRS=EPSG%3A3857&BBOX=-8543030.3273202,5492519.0747705,-8401010.3287862,5542356.0172055&WIDTH=929&HEIGHT=326')
        self.assertEqual(response.status_code, 200)

    def test_getLegend(self):
        pass

    def test_getCaps(self):
        response = self.client.get('/wms/datasets/test/?REQUEST=GetCapabilities')
        self.assertEqual(response.status_code, 200)


#class TestDap(TestCase):
    #def test_post_add(self):
    #    post_add(self, "http://tds.glos.us:8080/thredds/dodsC/glos/glcfs/michigan/fcfmrc-2d/Lake_Michigan_-_2D_best.ncd")
