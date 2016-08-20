# -*- coding: utf-8 -*-
from django.http import HttpResponse

import pyproj
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from wms import logger
import wms.add_cmaps


DEFAULT_HATCHES = ['.', '+', '*', '-', '/', ',', '\\', 'x', 'o', '[', ']', '^',
                   '_', '`', '#', '"', "'", '(', ')', '0', '1', '2', '3', '4',
                   '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@',
                   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                   'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                   'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                   'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                   'w', 'x', 'y', 'z', '{', '|', '}', '~']


def _get_common_params(request):
    bbox = request.GET['bbox']
    width = request.GET['width']
    height = request.GET['height']
    colormap = request.GET['colormap']
    colorscalerange = request.GET['colorscalerange']
    cmin = colorscalerange.min
    cmax = colorscalerange.max
    crs = request.GET['crs']
    params = (bbox, width,
              height, colormap,
              cmin, cmax, crs
              )
    return params


def tripcolor_response(tri_subset, data, request, data_location=None, dpi=None):
    """
    triang_subset is a matplotlib.Tri object in lat/lon units (will be converted to projected coordinates)
    xmin, ymin, xmax, ymax is the bounding pox of the plot in PROJETED COORDINATES!!!
    request is the original getMap request object
    """

    dpi = dpi or 80.

    bbox = request.GET['bbox']
    width = request.GET['width']
    height = request.GET['height']
    colormap = request.GET['colormap']
    colorscalerange = request.GET['colorscalerange']
    cmin = colorscalerange.min
    cmax = colorscalerange.max
    crs = request.GET['crs']

    EPSG4326 = pyproj.Proj(init='EPSG:4326')
    tri_subset.x, tri_subset.y = pyproj.transform(EPSG4326, crs, tri_subset.x, tri_subset.y)

    fig = Figure(dpi=dpi, facecolor='none', edgecolor='none')
    fig.set_alpha(0)
    fig.set_figheight(height/dpi)
    fig.set_figwidth(width/dpi)

    ax = fig.add_axes([0., 0., 1., 1.], xticks=[], yticks=[])
    ax.set_axis_off()

    if request.GET['logscale'] is True:
        norm_func = mpl.colors.LogNorm
    else:
        norm_func = mpl.colors.Normalize

    # Set out of bound data to NaN so it shows transparent?
    # Set to black like ncWMS?
    # Configurable by user?
    if cmin is not None and cmax is not None:
        norm = norm_func(vmin=cmin, vmax=cmax, clip=True)
    else:
        norm = norm_func()

    if data_location == 'face':
        ax.tripcolor(tri_subset, facecolors=data, edgecolors='none', norm=norm, cmap=colormap)
    else:
        ax.tripcolor(tri_subset, data, edgecolors='none', norm=norm, cmap=colormap)

    ax.set_xlim(bbox.minx, bbox.maxx)
    ax.set_ylim(bbox.miny, bbox.maxy)
    ax.set_frame_on(False)
    ax.set_clip_on(False)
    ax.set_position([0., 0., 1., 1.])

    canvas = FigureCanvasAgg(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response


def tricontouring_response(tri_subset, data, request, dpi=None):
    """
    triang_subset is a matplotlib.Tri object in lat/lon units (will be converted to projected coordinates)
    xmin, ymin, xmax, ymax is the bounding pox of the plot in PROJETED COORDINATES!!!
    request is the original getMap request object
    """

    dpi = dpi or 80.

    bbox = request.GET['bbox']
    width = request.GET['width']
    height = request.GET['height']
    colormap = request.GET['colormap']
    colorscalerange = request.GET['colorscalerange']
    cmin = colorscalerange.min
    cmax = colorscalerange.max
    crs = request.GET['crs']
    nlvls = request.GET['numcontours']

    EPSG4326 = pyproj.Proj(init='EPSG:4326')
    tri_subset.x, tri_subset.y = pyproj.transform(EPSG4326, crs, tri_subset.x, tri_subset.y)

    fig = Figure(dpi=dpi, facecolor='none', edgecolor='none')
    fig.set_alpha(0)
    fig.set_figheight(height/dpi)
    fig.set_figwidth(width/dpi)

    ax = fig.add_axes([0., 0., 1., 1.], xticks=[], yticks=[])
    ax.set_axis_off()

    if request.GET['logscale'] is True:
        norm_func = mpl.colors.LogNorm
    else:
        norm_func = mpl.colors.Normalize

    # Set out of bound data to NaN so it shows transparent?
    # Set to black like ncWMS?
    # Configurable by user?
    if cmin is not None and cmax is not None:
        lvls = np.linspace(cmin, cmax, nlvls)
        norm = norm_func(vmin=cmin, vmax=cmax, clip=True)
    else:
        lvls = nlvls
        norm = norm_func()

    if request.GET['image_type'] == 'filledcontours':
        ax.tricontourf(tri_subset, data, lvls, norm=norm, cmap=colormap, extend='both')
    elif request.GET['image_type'] == 'contours':
        ax.tricontour(tri_subset, data, lvls, norm=norm, cmap=colormap, extend='both')

    ax.set_xlim(bbox.minx, bbox.maxx)
    ax.set_ylim(bbox.miny, bbox.maxy)
    ax.set_frame_on(False)
    ax.set_clip_on(False)
    ax.set_position([0., 0., 1., 1.])

    canvas = FigureCanvasAgg(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response


def quiver_response(lon, lat, dx, dy, request, dpi=None):

    dpi = dpi or 80.

    bbox = request.GET['bbox']
    width = request.GET['width']
    height = request.GET['height']
    colormap = request.GET['colormap']
    colorscalerange = request.GET['colorscalerange']
    vectorscale = request.GET['vectorscale']
    cmin = colorscalerange.min
    cmax = colorscalerange.max
    crs = request.GET['crs']
    unit_vectors = None  # We don't support requesting these yet, but wouldn't be hard

    EPSG4326 = pyproj.Proj(init='EPSG:4326')
    x, y = pyproj.transform(EPSG4326, crs, lon, lat)  # TODO order for non-inverse?

    fig = Figure(dpi=dpi, facecolor='none', edgecolor='none')
    fig.set_alpha(0)
    fig.set_figheight(height/dpi)
    fig.set_figwidth(width/dpi)

    ax = fig.add_axes([0., 0., 1., 1.], xticks=[], yticks=[])
    ax.set_axis_off()
    mags = np.sqrt(dx**2 + dy**2)

    cmap = mpl.cm.get_cmap(colormap)

    if request.GET['logscale'] is True:
        norm_func = mpl.colors.LogNorm
    else:
        norm_func = mpl.colors.Normalize

    # Set out of bound data to NaN so it shows transparent?
    # Set to black like ncWMS?
    # Configurable by user?
    if cmin is not None and cmax is not None:
        norm = norm_func(vmin=cmin, vmax=cmax, clip=True)
    else:
        norm = norm_func()

    # plot unit vectors
    if unit_vectors:
        ax.quiver(x, y, dx/mags, dy/mags, mags, cmap=cmap, norm=norm, scale=vectorscale)
    else:
        ax.quiver(x, y, dx, dy, mags, cmap=cmap, norm=norm, scale=vectorscale)

    ax.set_xlim(bbox.minx, bbox.maxx)
    ax.set_ylim(bbox.miny, bbox.maxy)
    ax.set_frame_on(False)
    ax.set_clip_on(False)
    ax.set_position([0., 0., 1., 1.])

    canvas = FigureCanvasAgg(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response


def contouring_response(lon, lat, data, request, dpi=None):

    dpi = dpi or 80.

    bbox, width, height, colormap, cmin, cmax, crs = _get_common_params(request)
    nlvls = request.GET['numcontours']

    EPSG4326 = pyproj.Proj(init='EPSG:4326')
    x, y = pyproj.transform(EPSG4326, crs, lon, lat)

    fig = Figure(dpi=dpi, facecolor='none', edgecolor='none')
    fig.set_alpha(0)
    fig.set_figheight(height/dpi)
    fig.set_figwidth(width/dpi)

    ax = fig.add_axes([0., 0., 1., 1.], xticks=[], yticks=[])
    ax.set_axis_off()

    if request.GET['logscale'] is True:
        norm_func = mpl.colors.LogNorm
    else:
        norm_func = mpl.colors.Normalize

    if cmin is not None and cmax is not None:
        lvls = np.linspace(cmin, cmax, nlvls)
        norm = norm_func(vmin=cmin, vmax=cmax, clip=True)
    else:
        lvls = nlvls
        norm = norm_func()

    if request.GET['image_type'] == 'filledcontours':
        ax.contourf(x, y, data, lvls, norm=norm, cmap=colormap, extend='both')
    elif request.GET['image_type'] == 'contours':
        ax.contour(x, y, data, lvls, norm=norm, cmap=colormap, extend='both')
    elif request.GET['image_type'] == 'filledhatches':
        hatches = DEFAULT_HATCHES[:lvls]
        ax.contourf(x, y, data, lvls, norm=norm, cmap=colormap, hatches=hatches, extend='both')
    elif request.GET['image_type'] == 'hatches':
        hatches = DEFAULT_HATCHES[:lvls]
        ax.contourf(x, y, data, lvls, norm=norm, colors='none', hatches=hatches, extend='both')

    ax.set_xlim(bbox.minx, bbox.maxx)
    ax.set_ylim(bbox.miny, bbox.maxy)
    ax.set_frame_on(False)
    ax.set_clip_on(False)
    ax.set_position([0., 0., 1., 1.])

    canvas = FigureCanvasAgg(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response


def pcolormesh_response(lon, lat, data, request, dpi=None):

    dpi = dpi or 80.

    bbox, width, height, colormap, cmin, cmax, crs = _get_common_params(request)

    EPSG4326 = pyproj.Proj(init='EPSG:4326')
    x, y = pyproj.transform(EPSG4326, crs, lon, lat)
    fig = Figure(dpi=dpi, facecolor='none', edgecolor='none')
    fig.set_alpha(0)
    fig.set_figheight(height/dpi)
    fig.set_figwidth(width/dpi)
    ax = fig.add_axes([0., 0., 1., 1.], xticks=[], yticks=[])
    ax.set_axis_off()

    if request.GET['logscale'] is True:
        norm_func = mpl.colors.LogNorm
    else:
        norm_func = mpl.colors.Normalize

    if cmin is not None and cmax is not None:
        norm = norm = norm_func(vmin=cmin, vmax=cmax, clip=True)
    else:
        norm = norm_func()

    masked = np.ma.masked_invalid(data)
    ax.pcolormesh(x, y, masked, norm=norm, cmap=colormap)
    ax.set_xlim(bbox.minx, bbox.maxx)
    ax.set_ylim(bbox.miny, bbox.maxy)
    ax.set_frame_on(False)
    ax.set_clip_on(False)
    ax.set_position([0., 0., 1., 1.])

    canvas = FigureCanvasAgg(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response
