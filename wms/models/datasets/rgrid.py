# -*- coding: utf-8 -*-
import os
import time
import shutil
from datetime import datetime
import bisect
import tempfile
import itertools
from math import sqrt

import numpy as np
import pandas as pd
import netCDF4
import pytz
from pyaxiom.netcdf import EnhancedDataset, EnhancedMFDataset
from rtree import index

from wms import mpl_handler
from wms import gfi_handler
from wms import data_handler
from wms import gmd_handler

from wms.models import Dataset, Layer, VirtualLayer, NetCDFDataset
from wms.utils import DotDict, calc_lon_lat_padding, calc_safety_factor, find_appropriate_time

from wms import logger



class RGrid(object):

    def __init__(self, nc):
        self.nc = nc
        # in case a user as a version netcdf C library < 4.1.2
        try:
            self._filepath = nc.filepath()
        except ValueError:
            self._filepath = None

    def rgrid_compliant_file(self):
        """
        Determine whether a dataset is RGRID compliant.
        :return: True if dataset is compliant, raise an exception if it is not
        :rtype: bool
        """
        longitude = self.nc.get_variables_by_attributes(standard_name='longitude')
        latitude  = self.nc.get_variables_by_attributes(standard_name='latitude')
        if not longitude or not latitude:
            return False
        return True

    def save_as_netcdf(self, filepath):
        with netCDF4.Dataset(filepath, 'w') as nclocal:

            # copy all dimensions
            for d in self.nc.dimensions.values():
                nclocal.createDimension(d.name, d.size)

            # copy remote longitude/latitude (assume both present)
            _lon = self.nc.get_variables_by_attributes(standard_name='longitude')
            _lat = self.nc.get_variables_by_attributes(standard_name='latitude')
            # TODO: throw exception if more than one latitude/longitude
            #if len(_lon) > 1 or len(_lat) > 1:
            #    throw
            _lon = _lon[0]
            _lat = _lat[0]

            # are longitude/latitude 1D or 2D?
            if _lat.ndim > 1:
                _i = _lat.shape[0]
                _j = _lat.shape[1]
            else: # needs meshgrid
                _i = _lat.shape[0]
                _j = _lon.shape[0]

            nclocal.createDimension('i', _i)
            nclocal.createDimension('j', _j)
            latitude = nclocal.createVariable('latitude', 'f', ('i', 'j',))
            latitude.setncattr('standard_name', 'latitude')
            longitude = nclocal.createVariable('longitude', 'f', ('i', 'j',))
            longitude.setncattr('standard_name', 'longitude')

            # EPSG:4326 longitude -180 to 180
            _4326l = _lon[:]
            _4326l[_4326l > 180] = _4326l[_4326l > 180] + 360.0

            if _lat.ndim > 1:
                longitude[:] = _4326l[:] # use EPSG:4326 longitude
                latitude[:]  = _lat[:]
            else:
                longitude[:], latitude[:] = np.meshgrid(_4326l[:], _lat[:])

            nclocal.sync()

class RGridDataset(Dataset, NetCDFDataset):

    @staticmethod
    def is_valid(uri):
        try:
            with EnhancedDataset(uri) as ds:
                nc_ds = RGrid(ds)
                return nc_ds.rgrid_compliant_file() and 'rgrid' in ds.Conventions.lower()
        except RuntimeError:
            try:
                with EnhancedMFDataset(uri, aggdim='time') as ds:
                    nc_ds = RGrid(ds)
                    return nc_ds.rgrid_compliant_file() or 'rgrid' in ds.Conventions.lower()
            except (AttributeError, RuntimeError):
                return False
        except (AttributeError):
            return False

    def has_cache(self):
        return os.path.exists(self.topology_file)


    def make_rtree(self):
        with self.dataset() as nc:
            with self.topology() as topo:
                lon = topo.get_variables_by_attributes(standard_name='longitude')[0]
                lat = topo.get_variables_by_attributes(standard_name='latitude')[0]

                def rtree_generator_function():
                    c = -1
                    for row in range(lon.shape[0]):
                        for col in range(lon.shape[1]):
                            coord = (lon[row, col], lat[row, col], lon[row, col], lat[row, col],)
                            c += 1
                            yield(c, coord, (col, row))

                logger.info("Building Faces (centers) Rtree Topology Cache for {0}".format(self.name))
                _, temp_file = tempfile.mkstemp(suffix='.face')
                start = time.time()
                p = index.Property()
                p.filename = str(temp_file)
                p.overwrite = True
                p.storage   = index.RT_Disk
                p.dimension = 2
                index.Index(p.filename.decode('utf-8'),
                            rtree_generator_function(),
                            properties=p,
                            overwrite=True,
                            interleaved=True)
                logger.info("Built Faces Rtree Topology Cache in {0} seconds.".format(time.time() - start))

                shutil.move('{}.dat'.format(temp_file), self.face_tree_data_file)
                shutil.move('{}.idx'.format(temp_file), self.face_tree_index_file)

    def update_cache(self, force=False):
        with self.dataset() as nc:
            rg = RGrid(nc)

            rg.save_as_netcdf(self.topology_file) # saves topology coordinates (latitude, longitude, time)
            if not os.path.exists(self.topology_file):
                logger.error("Failed to create topology_file cache for Dataset '{}'".format(self.dataset))
                return

            # add time to the cached topology
            time_vars = nc.get_variables_by_attributes(standard_name='time')
            time_dims = list(itertools.chain.from_iterable([time_var.dimensions for time_var in time_vars]))
            unique_time_dims = list(set(time_dims))
            with EnhancedDataset(self.topology_file, mode='a') as cached_nc:
                # create pertinent time dimensions if they aren't already present
                for unique_time_dim in unique_time_dims:
                    dim_size = len(nc.dimensions[unique_time_dim])
                    try:
                        cached_nc.createDimension(unique_time_dim, size=dim_size)
                    except RuntimeError:
                        continue
                # support cases where there may be more than one variable with standard_name='time' in a dataset
                for time_var in time_vars:
                    try:
                        time_var_obj = cached_nc.createVariable(time_var.name,
                                                                time_var.dtype,
                                                                time_var.dimensions)
                    except RuntimeError:
                        time_var_obj = cached_nc.variables[time_var.name]
                    finally:
                        time_var_obj[:] = time_var[:]
                        time_var_obj.units = time_var.units
                        time_var_obj.standard_name = 'time'

            # Now do the RTree index
            self.make_rtree()

        self.cache_last_updated = datetime.utcnow().replace(tzinfo=pytz.utc)
        self.save()

    def getmap(self, layer, request):
        time_index, time_value = self.nearest_time(layer, request.GET['time'])
        wgs84_bbox = request.GET['wgs84_bbox']

        with self.dataset() as nc:
            with self.topology() as topo:

                lon = topo.get_variables_by_attributes(standard_name='longitude')[0][:]
                lat = topo.get_variables_by_attributes(standard_name='latitude')[0][:]

                if isinstance(layer, Layer):
                    raw_var = nc.variables[layer.access_name]
                    if len(raw_var.shape) == 4:
                        z_index, z_value = self.nearest_z(layer, request.GET['elevation'])
                        raw_data = raw_var[time_index, z_index, :]
                    elif len(raw_var.shape) == 3:
                        raw_data = raw_var[time_index, :]
                    elif len(raw_var.shape) == 2:
                        raw_data = raw_var[:]
                    else:
                        raise BaseException('Unable to trim variable {0} data.'.format(layer.access_name))

                    if request.GET['image_type'] == 'pcolor':
                        return mpl_handler.pcolormesh_response(lon, lat, data=raw_data, request=request)
                    elif request.GET['image_type'] in ['filledhatches', 'hatches', 'filledcontours', 'contours']:
                        return mpl_handler.contouring_response(lon, lat, data=raw_data, request=request)
                    else:
                        raise NotImplementedError('Image type "{}" is not supported.'.format(request.GET['image_type']))

                elif isinstance(layer, VirtualLayer):
                    x_var = None
                    y_var = None
                    raw_vars = []
                    for l in layer.layers:
                        data_obj = getattr(cached_sg, l.access_name)
                        raw_var = nc.variables[l.access_name]
                        raw_vars.append(raw_var)
                        if len(raw_var.shape) == 4:
                            z_index, z_value = self.nearest_z(layer, request.GET['elevation'])
                            raw_data = raw_var[time_index, z_index, data_obj.center_slicing[-2], data_obj.center_slicing[-1]]
                        elif len(raw_var.shape) == 3:
                            raw_data = raw_var[time_index, data_obj.center_slicing[-2], data_obj.center_slicing[-1]]
                        elif len(raw_var.shape) == 2:
                            raw_data = raw_var[data_obj.center_slicing]
                        else:
                            raise BaseException('Unable to trim variable {0} data.'.format(l.access_name))

                        raw_data = avg_to_cell_center(raw_data, data_obj.center_axis)
                        if x_var is None:
                            if data_obj.vector_axis and data_obj.vector_axis.lower() == 'x':
                                x_var = raw_data
                            elif data_obj.center_axis == 1:
                                x_var = raw_data

                        if y_var is None:
                            if data_obj.vector_axis and data_obj.vector_axis.lower() == 'y':
                                y_var = raw_data
                            elif data_obj.center_axis == 0:
                                y_var = raw_data

                    if x_var is None or y_var is None:
                        raise BaseException('Unable to determine x and y variables.')

                    dim_lengths = [ len(v.dimensions) for v in raw_vars ]
                    if len(list(set(dim_lengths))) != 1:
                        raise AttributeError('One or both of the specified variables has screwed up dimensions.')

                    if request.GET['image_type'] == 'vectors':

                        vectorstep = request.GET['vectorstep']
                        # don't do this if the vectorstep is 1; let's save a microsecond or two
                        # it's identical to getting all the data
                        if vectorstep > 1:
                            data_dim = len(lon.shape)
                            step_slice = (np.s_[::vectorstep],) * data_dim  # make sure the vector step is used for all applicable dimensions
                            lon = lon[step_slice]
                            lat = lat[step_slice]
                            x_var = x_var[step_slice]
                            y_var = y_var[step_slice]

                        vectorscale = request.GET['vectorscale']
                        padding_factor = calc_safety_factor(vectorscale)
                        # figure out the average distance between lat/lon points
                        # do the math after taking into the vectorstep if specified
                        spatial_idx_padding = calc_lon_lat_padding(lon, lat, padding_factor)
                        spatial_idx = data_handler.lat_lon_subset_idx(lon, lat,
                                                                      lonmin=wgs84_bbox.minx,
                                                                      latmin=wgs84_bbox.miny,
                                                                      lonmax=wgs84_bbox.maxx,
                                                                      latmax=wgs84_bbox.maxy,
                                                                      padding=spatial_idx_padding
                                                                      )

                        subset_lon = self._spatial_data_subset(lon, spatial_idx)
                        subset_lat = self._spatial_data_subset(lat, spatial_idx)

                        spatial_subset_x_var = self._spatial_data_subset(x_var, spatial_idx)
                        spatial_subset_y_var = self._spatial_data_subset(y_var, spatial_idx)
                        return mpl_handler.quiver_response(subset_lon,
                                                           subset_lat,
                                                           spatial_subset_x_var,
                                                           spatial_subset_y_var,
                                                           request,
                                                           vectorscale
                                                           )
                    else:
                        raise NotImplementedError('Image type "{}" is not supported.'.format(request.GET['image_type']))

    def getfeatureinfo(self, layer, request):
        with self.dataset() as nc:
            with self.topology() as topo:
                data_obj = nc.variables[layer.access_name]

                geo_index, closest_x, closest_y, start_time_index, end_time_index, return_dates = self.setup_getfeatureinfo(topo, data_obj, request)

                return_arrays = []
                z_value = None
                if isinstance(layer, Layer):
                    if len(data_obj.shape) == 4:
                        z_index, z_value = self.nearest_z(layer, request.GET['elevation'])
                        data = data_obj[start_time_index:end_time_index, z_index, geo_index[0], geo_index[1]]
                    elif len(data_obj.shape) == 3:
                        data = data_obj[start_time_index:end_time_index, geo_index[0], geo_index[1]]
                    elif len(data_obj.shape) == 2:
                        data = data_obj[geo_index[0], geo_index[1]]
                    else:
                        raise ValueError("Dimension Mismatch: data_obj.shape == {0} and time indexes = {1} to {2}".format(data_obj.shape, start_time_index, end_time_index))

                    return_arrays.append((layer.var_name, data))

                elif isinstance(layer, VirtualLayer):

                    # Data needs to be [var1,var2] where var are 1D (nodes only, elevation and time already handled)
                    for l in layer.layers:
                        if len(data_obj.shape) == 4:
                            z_index, z_value = self.nearest_z(layer, request.GET['elevation'])
                            data = data_obj[start_time_index:end_time_index, z_index, geo_index[0], geo_index[1]]
                        elif len(data_obj.shape) == 3:
                            data = data_obj[start_time_index:end_time_index, geo_index[0], geo_index[1]]
                        elif len(data_obj.shape) == 2:
                            data = data_obj[geo_index[0], geo_index[1]]
                        else:
                            raise ValueError("Dimension Mismatch: data_obj.shape == {0} and time indexes = {1} to {2}".format(data_obj.shape, start_time_index, end_time_index))
                        return_arrays.append((l.var_name, data))

                # Data is now in the return_arrays list, as a list of numpy arrays.  We need
                # to add time and depth to them to create a single Pandas DataFrame
                if len(data_obj.shape) == 4:
                    df = pd.DataFrame({'time': return_dates,
                                       'x': closest_x,
                                       'y': closest_y,
                                       'z': z_value})
                elif len(data_obj.shape) == 3:
                    df = pd.DataFrame({'time': return_dates,
                                       'x': closest_x,
                                       'y': closest_y})
                elif len(data_obj.shape) == 2:
                    df = pd.DataFrame({'x': closest_x,
                                       'y': closest_y})
                else:
                    df = pd.DataFrame()

                # Now add a column for each member of the return_arrays list
                for (var_name, np_array) in return_arrays:
                    df.loc[:, var_name] = pd.Series(np_array, index=df.index)

                return gfi_handler.from_dataframe(request, df)

    def wgs84_bounds(self, layer):
        with netCDF4.Dataset(self.topology_file) as nc:
            longitudes = nc.get_variables_by_attributes(standard_name='longitude')[0][:]
            latitudes  = nc.get_variables_by_attributes(standard_name='latitude')[0][:]
            lon_max = longitudes.max()
            lon_min = longitudes.min()
            lat_max = latitudes.max()
            lat_min = latitudes.min()
            return DotDict(minx=lon_min,
                           miny=lat_min,
                           maxx=lon_max,
                           maxy=lat_max,
                           bbox=(lon_min, lat_min, lon_max, lat_max)
                           )

    def _spatial_data_subset(self, data, spatial_index):
        rows = spatial_index[0, :]
        columns = spatial_index[1, :]
        data_subset = data[rows, columns]
        return data_subset

    def times(self, layer):
        with self.topology() as nc:
            time_vars = nc.get_variables_by_attributes(standard_name='time')
            if len(time_vars) == 1:
                time_var = time_vars[0]
            else:
                # if there is more than variable with standard_name = time
                # fine the appropriate one to use with the layer
                var_obj = nc.variables[layer.access_name]
                time_var_name = find_appropriate_time(var_obj, time_vars)
                time_var = nc.variables[time_var_name]
            return netCDF4.num2date(time_var[:], units=time_var.units)

    def depth_variable(self, layer):
        with self.dataset() as nc:
            try:
                layer_var = nc.variables[layer.access_name]
                coordinates = [d for d in nc.dimensions]
                try:
                    coordinates = coordinates + layer_var.coordinates.strip().split()
                except:
                    pass
                for cv in coordinates:
                    try:
                        coord_var = nc.variables[cv]
                        if hasattr(coord_var, 'axis') and coord_var.axis.lower().strip() == 'z':
                            return coord_var
                        elif hasattr(coord_var, 'positive') and coord_var.positive.lower().strip() in ['up', 'down']:
                            return coord_var
                    except BaseException:
                        pass
            except AttributeError:
                pass

    def depth_direction(self, layer):
        d = self.depth_variable(layer)
        if d is not None:
            if hasattr(d, 'positive'):
                return d.positive
        return 'unknown'

    def depths(self, layer):
        """ sci-wms only deals in depth indexes at this time (no sigma) """
        d = self.depth_variable(layer)
        if d is not None:
            return range(0, d.shape[0])
        return []

    def nearest_z(self, layer, z):
        """
        Return the z index and z value that is closest
        """
        depths = self.depths(layer)
        depth_idx = bisect.bisect_right(depths, z)
        try:
            depths[depth_idx]
        except IndexError:
            depth_idx -= 1
        return depth_idx, depths[depth_idx]

    def humanize(self):
        return "RGRID"
