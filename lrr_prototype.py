import math
import gdrom_rules
import numpy as np
import geopandas as gpd
import netCDF4 as nc
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict
from shapely.geometry import Point


def read_conus_grid_nc(conus_grid_nc_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with nc.Dataset(conus_grid_nc_path) as ds:
        lon_array_conus = ds.variables['lon'][:]
        lat_array_conus = ds.variables['lat'][:]
        grid_id_array_conus = ds.variables['id'][:, :]
        flow_dir_array_conus = ds.variables['flow_dir'][:, :]
    return lat_array_conus, lon_array_conus, grid_id_array_conus, flow_dir_array_conus

def read_conus_reservoir_nc(conus_res_nc_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with nc.Dataset(conus_res_nc_path) as ds:
        res_gid_array = ds.variables['gid'][:]    # reservoir gid, CONUS-wide
        res_grid_id_array = ds.variables['grid_id'][:]    # grid id where reservoir is located, CONUS-wide
        res_max_storage_array = ds.variables['max_storage'][:]    # reservoir max storage, CONUS-wide
        res_lat_array = ds.variables['lat'][:]    # reservoir lat, CONUS-wide
        res_lon_array = ds.variables['lon'][:]    # reservoir lon, CONUS-wide
    return res_gid_array, res_grid_id_array, res_max_storage_array, res_lat_array, res_lon_array

def read_huc4_basin(huc4: str, nhd_data_dir: str, crs: str='epsg:4326') -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:

    # read HUCs
    huc2 = huc4[0:2]
    wbd_gdb_path = f'{nhd_data_dir}/Raw/WBD/WBD_{huc2}_HU2_GDB.gdb'
    gdf_huc2_conus = gpd.read_file(wbd_gdb_path, layer='WBDHU2')
    gdf_huc4_conus = gpd.read_file(wbd_gdb_path, layer='WBDHU4')
    gdf_huc4 = gdf_huc4_conus[gdf_huc4_conus['huc4'] == huc4]
    return gdf_huc2_conus, gdf_huc4_conus, gdf_huc4

def get_grids_in_hu(
        lon_array: np.ndarray,    # lon array of the whole CONUS 
        lat_array: np.ndarray,    # lat array of the whole CONUS
        gdf_huc: gpd.GeoDataFrame    # geodataframe of the huc4
        ) -> Tuple[gpd.GeoDataFrame, np.ndarray, np.ndarray]:
    """
    Get grids (lon-lat) within the target HUC4.    

    Returns
    -------
    gdf_huc4_points : gpd.GeoDataFrame {lon index, lat index, geometry}
        geodataframe of the points within the huc4
    lon_index_array : np.ndarray
        lon index of the points within the huc4, CONUS-wide index
    lat_index_array : np.ndarray
        lat index of the points within the huc4, CONUS-wide index
    """
    
    # create lon-lat pairs from global vars: lon_array and lat_array
    lon_lat_all = [[Point(lon, lat), i, j] for i, lon in enumerate(lon_array) for j, lat in enumerate(lat_array)]    # [Point(lon, lat), i-lon index, j-lat index]
    
    # index the wanted hu from the gdf
    huc_geo = gdf_huc['geometry'].values   # the polygon for this huc
        
    # find (lon, lat) pairs within the area
    lon_lat_sub = [i for i in lon_lat_all if huc_geo.contains(i[0])[0]]

    # create point geodataframe for selected points and check
    d = {'lon index': [i[1] for i in lon_lat_sub], 'lat index': [i[2] for i in lon_lat_sub]}
    gdf_huc4_points = gpd.GeoDataFrame(d, 
                                  geometry=[i[0] for i in lon_lat_sub], crs='EPSG:4326')   # lon index, lat index, geometry
    
    return gdf_huc4_points, np.unique(gdf_huc4_points['lon index']), np.unique(gdf_huc4_points['lat index'])

def get_downstream_cell(i: int, j: int, direction: int) -> Tuple[int, int] or None:
    """
    Returns the downstream cell coordinates based on D8 flow direction.
    d8 directions:
    32  64 128
    16  x   1
    8   4   2

    Parameters:
    i (int): Row index of the current cell
    j (int): Column index of the current cell
    direction (int): D8 flow direction code of the current cell

    Returns:
    tuple: Coordinates (row, col) of the downstream cell, or None if no downstream
    """

    # Define the changes in row (di) and column (dj) for each flow direction
    # PLEASE note: the direction is defined as such, because the latitude array increases as the row index increases & longitude array increases as the col index increases!!!

    direction_map = {
        1: (0, 1),     # East: i, j+1
        2: (-1, 1),    # Southeast: i-1, j+1
        4: (-1, 0),    # South: i-1, j (move south, decrease row index)
        8: (-1, -1),   # Southwest: i-1, j-1
        16: (0, -1),   # West: i, j-1
        32: (1, -1),   # Northwest: i+1, j-1
        64: (1, 0),    # North: i+1, j (move north, increase row index)
        128: (1, 1)    # Northeast: i+1, j+1
    }


    # Get the changes in row and column for the given direction
    di, dj = direction_map.get(direction, (0, 0))    # if direction is not in the direction_map, return (0, 0)

    # If di and dj are both 0, it means the direction is not defined (e.g., a sink)
    if di == 0 and dj == 0:
        return None

    # Calculate the coordinates of the downstream cell
    downstream_i = i + di
    downstream_j = j + dj
    
    return (downstream_i, downstream_j)

def sort_grids_by_flow_dir(
        flow_dir_array_conus: np.ndarray,    # flow direction array of the whole CONUS
        gdf_huc4_points: gpd.GeoDataFrame,    # geodataframe of the points within the huc4
        grid_id_array: np.ndarray,    # grid id array of the whole CONUS
        lat_array: np.ndarray,    # lat array of the whole CONUS
        lon_array: np.ndarray    # lon array of the whole CONUS
    ) -> Tuple[Dict, Dict, nx.DiGraph, np.ndarray]: 
    """
    Sort the basin grids by flow direction.

    Returns
    -------
    upstream_grid_dict : Dict {(i, j): [(upstream i, upstream j) list]}
        (grid row index, grid col index): [(upstream grid row index, upstream grid col index) list] basin-wide
    upstream_grid_id_dict : Dict {grid id: [upstream grid id list]}
        grid id: [upstream grid id list] basin-wide
    G : nx.DiGraph
        the flow network graph
    flow_dir_array_huc4 : np.ndarray
        flow direction array of the target HUC4
    """
    
    lon_index = gdf_huc4_points['lon index'].values    # contains duplicate values
    lat_index = gdf_huc4_points['lat index'].values    # contains duplicate values

    # Subset the flow direction array to the target HU basin (the rectangular area covering the HUC4 basin)
    # technically, I didn't "subset", just set the flow direction values outside the HU to -9999
    # first, set all conus grids outside the target HU to -9999
    mask = np.ones_like(flow_dir_array_conus, dtype=bool)
    mask[lat_index, lon_index] = False
    flow_dir_array_conus[mask] = -1    # set all conus grids outside the target HU to -1
    # second, subset the flow direction array to the target HU basin
    lat_index_unique = np.unique(lat_index)
    lon_index_unique = np.unique(lon_index)
    flow_dir_array_huc4 = flow_dir_array_conus[np.ix_(lat_index_unique, lon_index_unique)]

    # also, get the grid id array for the target HU basin for record
    grid_id_array_huc4 = grid_id_array[np.ix_(lat_index_unique, lon_index_unique)]
    lat_array_huc4 = lat_array[lat_index_unique]
    lon_array_huc4 = lon_array[lon_index_unique]

    ########## Sort the grids ##########

    # create a graph
    G = nx.DiGraph()

    # add nodes
    nrows, ncols = flow_dir_array_huc4.shape
    for i in range(nrows):
        for j in range(ncols):
            # add node for each grid
            # if the node flows out of the huc4 basin, skip
            if flow_dir_array_huc4[i, j] == -1:
                continue
            
            G.add_node((i, j), 
                    flow_dir=flow_dir_array_huc4[i, j], grid_id=grid_id_array_huc4[i, j], grid_lat=lat_array_huc4[i], grid_lon=lon_array_huc4[j])

            # determine downstream grids and add edges
            downstream_grid_ij = get_downstream_cell(i, j, flow_dir_array_huc4[i, j])
            
            # check if downstream grid is outside the flow_dir_array_huc4
            if downstream_grid_ij is not None and downstream_grid_ij[0] >= nrows:
                downstream_grid_ij = None
            elif downstream_grid_ij is not None and downstream_grid_ij[1] >= ncols:
                downstream_grid_ij = None

            if downstream_grid_ij is not None and flow_dir_array_huc4[downstream_grid_ij] != -1:
                # if downstream grid is not None AND the downstream grid is not outside the huc4 basin
                G.add_edge((i, j), downstream_grid_ij)

    # typological sorting
    sorted_grid_list = list(nx.topological_sort(G))

    # store the sorted grid id and upstream grid id for each grid: {grid: [upstream grid list]}
    # the grid order is following the topological sorting
    upstream_grid_dict = {grid: [] for grid in sorted_grid_list}
    for grid in sorted_grid_list:
        upstream_grid_dict[grid] = list(G.predecessors(grid))
    # convert the grid index in upstream_grid_dict to grid id (the grid attribute in the node)
    upstream_grid_id_dict = {G.nodes[grid]['grid_id']: [G.nodes[upstream_grid]['grid_id'] for upstream_grid in upstream_grid_dict[grid]] for grid in upstream_grid_dict}

    return upstream_grid_dict, upstream_grid_id_dict, G, flow_dir_array_huc4

def plot_flow_network(
        G: nx.DiGraph,    # the flow network graph
        gdf_huc4: gpd.GeoDataFrame,    # geodataframe of the target HUC4
        ):
    
    # ---- Visualize the graph on the map ---- #

    nodes_data = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    geometry = [Point(xy) for xy in zip(nodes_data['grid_lon'], nodes_data['grid_lat'])]
    gdf_nodes = gpd.GeoDataFrame(nodes_data, geometry=geometry, crs='EPSG:4326')

    lon_array_huc4 = np.unique(gdf_nodes['grid_lon'].values)
    lat_array_huc4 = np.unique(gdf_nodes['grid_lat'].values)

    # ------------------- Plot ------------------- #
    # Variables to store components of arrows
    x = []
    y = []
    dx = []
    dy = []

    default_lon = lon_array_huc4[0]
    default_lat = lat_array_huc4[0]
    for edge in G.edges():
        start_node = G.nodes[edge[0]]
        end_node = G.nodes[edge[1]]
        try:
            x_start, y_start = start_node['grid_lon'], start_node['grid_lat']
            x_end, y_end = end_node['grid_lon'], end_node['grid_lat']
        except KeyError:
            x_start, y_start = default_lon, default_lat
            x_end, y_end = default_lon, default_lat

        x.append(x_start)
        y.append(y_start)
        dx.append(x_end - x_start)
        dy.append(y_end - y_start)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot arrows
    ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1, color='gray')

    # Plot nodes
    gdf_nodes.plot(ax=ax, marker='o', color='black', alpha=0.7, markersize=10)

    gdf_huc4.plot(ax=ax, facecolor='none', edgecolor='tab:gray', linewidth=1)

    # # plot flow line
    # # specify to which stream order
    # max_order = gdf_flow_huc4['StreamOrde'].max()
    # min_order_to_keep = 4
    # gdf_flow_huc4.loc[gdf_flow_huc4['StreamOrde']>=min_order_to_keep].plot(ax=ax, linewidth=1)

    plt.show()

def read_nldas_runoff(
        nldas_runoff_nc_path: str,    # path to the nldas runoff nc file
        huc4_lat_index_in_conus: np.ndarray,    # lat index of the huc4 basin in the conus grid
        huc4_lon_index_in_conus: np.ndarray,    # lon index of the huc4 basin in the conus grid
        start_date: str,    # start date of the simulation period yyyy-mm-dd
        end_date: str    # end date of the simulation period yyyy-mm-dd
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read NLDAS runoff data for the simulation period.
        The returned arrays are for the huc4 basin only.

    Returns
    -------
    nldas_qs_array : np.ndarray
        NLDAS surface runoff array for the simulation period (mm/day)
    nldas_qsb_array : np.ndarray
        NLDAS baseflow array for the simulation period (mm/day)
    """

    # make sure the huc4_lat_index_in_conus and huc4_lon_index_in_conus are 1D array and sorted
    huc4_lat_index_in_conus = np.unique(huc4_lat_index_in_conus)
    huc4_lon_index_in_conus = np.unique(huc4_lon_index_in_conus)

    with nc.Dataset(nldas_runoff_nc_path, 'r') as ds:
        time = ds.variables['time'][:]    # np array of str yyyy-mm-dd
        # time index for the simulation period
        start_date_index = np.where(time == start_date)[0][0]
        end_date_index = np.where(time == end_date)[0][0]
        time_index = np.array(range(start_date_index, end_date_index+1))

        # nldas_qs_array = ds.variables['Qs'][start_date_index:end_date_index+1, huc4_lat_index_in_conus, huc4_lon_index_in_conus]
        # nldas_qsb_array = ds.variables['Qsb'][start_date_index:end_date_index+1, huc4_lat_index_in_conus, huc4_lon_index_in_conus]
        nldas_qs_array = ds.variables['Qs'][:,:,:][np.ix_(time_index, huc4_lat_index_in_conus, huc4_lon_index_in_conus)]
        nldas_qsb_array = ds.variables['Qsb'][:,:,:][np.ix_(time_index, huc4_lat_index_in_conus, huc4_lon_index_in_conus)]

    # fill nan with 0
    nldas_qs_array = np.nan_to_num(nldas_qs_array)
    nldas_qsb_array = np.nan_to_num(nldas_qsb_array)

    return nldas_qs_array, nldas_qsb_array

def read_pdsi(pdsi_file_path: str, start_date: str, end_date: str) -> np.ndarray:
    """
    Returns
    -------
    pdsi_array : np.ndarray
        PDSI array for the simulation period
    """

    df = pd.read_csv(pdsi_file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    pdsi_array = df.loc[start_date:end_date, 'PDSI'].values

    return pdsi_array

def initialize_grid(
        nrows: int,    # number of rows (latitude) of the basin grid
        ncols: int,    # number of columns (longitude) of the basin grid
        reservoir_grid_id_array: np.ndarray,    # grid id array where reservoirs are located, CONUS-wide
        reservoir_id_array: np.ndarray,    # reservoir gid array, CONUS-wide
        reservoir_max_storage_array: np.ndarray,    # reservoir max storage array, CONUS-wide
        grid_id_array: np.ndarray,    # grid id array of the huc4 basin
        grid_length: float,    # grid length (km)
        flow_dir_array_huc4: np.ndarray    # flow direction array of the huc4 basin
    ) -> np.ndarray:    # structured array of the grid
    """
    Initialize grid of the simulation HUC4 basin, for the first time step, i.e., first model state.
    """

    grid = np.zeros((nrows, ncols), dtype=[
        ('grid_id', int),
        ('flow_direction', int),    # flow direction code based on D8
        ('flow_distance', float),    # flow distance to the next grid cell [km]: grid_length * sqrt(2) for diagonal, grid_length for horizontal/vertical
        ('reservoir_id', int),
        ('reservoir_storage_start', float),    # reservoir storage at the start of the time step [acft]
        ('reservoir_storage_end', float),    # reservoir storage at the end of the time step [acft]
        ('reservoir_max_storage', float),    # reservoir max storage [acft]
        ('qs', float),    # [mm/day]
        ('qsb', float),    # [mm/day]
        ('inflow_from_upstream', float),    # [m3/s]
        ('inflow_from_grid_runoff', float),    # [m3/s]
        ('inflow_total', float),    # total inflow to the grid cell [m3/s]
        ('outflow_before_operation', float),    # [m3/s]
        ('outflow_after_operation', float),    # [m3/s]
        ('grid_storage_start', float),    # grid storage at the start of the time step [m3]
        ('grid_storage_end', float),    # grid storage at the end of the time step [m3]
        ('grid_demand_agriculture', float),
        ('grid_demand_mi', float)
    ])

    # assign values for variables with starting values
    grid[:, :]['grid_id'] = grid_id_array
    grid[:, :]['flow_direction'] = flow_dir_array_huc4
    
    for i in range(nrows):
        for j in range(ncols):
            # Assign flow distance by flow direction, using switch case
            if flow_dir_array_huc4[i, j] == 1:
                grid[i, j]['flow_distance'] = grid_length
            elif flow_dir_array_huc4[i, j] == 2:
                grid[i, j]['flow_distance'] = grid_length * math.sqrt(2)
            elif flow_dir_array_huc4[i, j] == 4:
                grid[i, j]['flow_distance'] = grid_length
            elif flow_dir_array_huc4[i, j] == 8:
                grid[i, j]['flow_distance'] = grid_length * math.sqrt(2)
            elif flow_dir_array_huc4[i, j] == 16:
                grid[i, j]['flow_distance'] = grid_length
            elif flow_dir_array_huc4[i, j] == 32:
                grid[i, j]['flow_distance'] = grid_length * math.sqrt(2)
            elif flow_dir_array_huc4[i, j] == 64:
                grid[i, j]['flow_distance'] = grid_length
            elif flow_dir_array_huc4[i, j] == 128:
                grid[i, j]['flow_distance'] = grid_length * math.sqrt(2)
            else:
                grid[i, j]['flow_distance'] = 0.00000000001    # else a sink, set flow distance to a very small number rather than 0

            # Assign reservoir id and max storage
            grid_id = grid_id_array[i, j]
            if np.where(reservoir_grid_id_array == grid_id)[0].size == 0:    # this means this grid doesn't have a reservoir
                continue
            reservoir_gid = reservoir_id_array[np.where(reservoir_grid_id_array == grid_id)[0][0]]
            reservoir_max_storage = reservoir_max_storage_array[np.where(reservoir_grid_id_array == grid_id)[0][0]]
            grid[i, j]['reservoir_id'] = reservoir_gid
            grid[i, j]['reservoir_max_storage'] = reservoir_max_storage
            grid[i, j]['reservoir_storage_start'] = reservoir_max_storage / 2    # initialize reservoir storage to half of the max storage

                    
    return grid

def update_grid_storage(
        grid: np.ndarray,    # structured array of the grid at the time step
        i: int,    # grid row index
        j: int,    # grid col index
        grid_length: float,    # grid length (km)
        u_e: float    # effective velocity [m/s], as the model parameter to calibrate
    ) -> float:    # grid storage at the end of the time step [m3]
    """
    Update grid storage.
    """

    d_in = grid[i, j]['inflow_total']    # m3/s

    d = grid[i, j]['flow_distance']    # km
    d = d * 1000    # km to m

    delta_t = 3600 * 24    # daily time step converting to seconds

    c = u_e / d
    c_t = math.exp(-c * delta_t)    # 30 days time constant

    grid_storage_end = c_t * grid[i, j]['grid_storage_start'] + (1 - c_t) * d_in / c

    return grid_storage_end

def calculate_grid_outflow_before_operation(
        grid_storage_end: float,    # grid storage at the end of the time step [m3]  
        grid_length: float,    # grid length (km)
        d: float,    # grid flow distance [km]
        u_e: float    # effective velocity [m/s], as the model parameter to calibrate
    ) -> float:
    """
    Calculate outflow before reservoir operation.
    """

    delta_z = grid_length * 1000    # grid length (km) to m
    d = d * 1000    # grid length (km) to m

    c = u_e / d
    
    d_out = c * grid_storage_end    # outflow to downstream grid [m3/s]

    return d_out

def gdrom_release(
    reservoir_gid: int,     
    max_storage: float, 
    inflow: float,    # inflow to the reservoir [acft/day] 
    storage: float,    # reservoir storage at the start of the time step [acft] 
    pdsi: float,     
    doy: int,    # day of year
    min_storage: float = 0    # currently, min storage is set to 0 
    ) -> float:    # simulated release from GDROM [acft]
    """
    Calculate reservoir release based on the GDROM method.

    Returns:
    --------
    sim_release: simulated release from GDROM. Unit is acft
    reservoir_storage_end: reservoir storage at the end of the time step [acft]
    """
    
    # First, determine which module to use
    try:
        sim_module_func = getattr(gdrom_rules, f'determine_module_{reservoir_gid}')
        sim_module = sim_module_func(inflow, storage, pdsi, doy)
    
        # Then, determine release based on module
        sim_release_func = getattr(gdrom_rules, f'determine_module_release_{reservoir_gid}_{sim_module}')
        sim_release = sim_release_func(inflow, storage)
        
    except AttributeError:
        # This error comes probably from the single-module reservoir
        sim_release_func = getattr(gdrom_rules, f'determine_module_release_{reservoir_gid}_0')
        sim_release = sim_release_func(inflow, storage)
    
    # Postprocess the GDROM-simulated release. e.g., to see if any physical constraint is violated
    # check if storage > max storage
    if inflow + storage - sim_release > max_storage:
        sim_release = inflow + storage - max_storage
    # check if storage < min storage. Currently, min storage is set to 0
    elif inflow + storage - sim_release < min_storage:
        sim_release = inflow + storage - min_storage

    return sim_release

def update_reservoir_storage(
        reservoir_storage_start: float,    # reservoir storage at the start of the time step [acft]
        inflow_volume: float,    # inflow volume to the reservoir, i.e., grid_outflow_before_operation [m3/s]
        release_volume: float,    # outflow volume from the reservoir, calculated from gdrom [acft/day]    
    ) -> float:    # reservoir storage at the end of the time step [acft]
    """
    Update reservoir storage.
    """
    
    inflow_volume = inflow_volume * 3600 * 24 * 0.000810714    # [acft/day]
    release_volume = release_volume * 3600 * 24 * 0.000810714    # [acft/day]
    reservoir_storage_end = reservoir_storage_start + inflow_volume - release_volume
    
    return reservoir_storage_end

def calculate_grid_outflow_after_operation(
        grid_gid: int,    # reservoir gid of the grid
        max_storage: float,    # reservoir max storage [acft]
        reservoir_storage_start: float,    # reservoir storage at the start of the time step [m3]
        pdsi: float,    # PDSI of the grid
        doy: int,    # day of year
        d_out_before_operation: float,    # outflow before reservoir operation [m3/s]
        min_storage: float = 0    # currently, min storage is set to 0
    ) -> float:    # outflow after reservoir operation [m3/s]
    """
    Calculate outflow after reservoir operation.
        Inflow to the reservoir is the outflow before reservoir operation.
        If grid has no reservoir, outflow after reservoir operation is the same as outflow before reservoir operation.
    """

    if grid_gid > 0:    # if grid_gid not 0, i.e., the grid has a reservoir
        reservoir_inflow = d_out_before_operation
        reservoir_inflow_volume = reservoir_inflow * 3600 * 24 * 0.000810714    # [acft/day]

        # calculate reservoir release
        reservoir_release = gdrom_release(grid_gid, max_storage, reservoir_inflow_volume, reservoir_storage_start, pdsi, doy)    # [acft/day]
        # calculate outflow after reservoir operation
        d_out_after_operation = reservoir_release
        # convert back to m3/s
        d_out_after_operation = d_out_after_operation / (3600*24) * 1233.48    # [m3/s] 
    else:
        d_out_after_operation = d_out_before_operation


    return d_out_after_operation

def channel_routing(
        grid_length: float,    # grid length (km)
        grid: np.ndarray,    # structured array of the grid at the time step
        upstream_grid_dict: Dict,    # {(grid_i, grid_j): [(grid_i, grid_j)] list of connected upstream grids}. The key has been sorted by topological sorting.  
        qs_array_t: np.ndarray,    # NLDAS surface runoff array for the time step t (mm/day)
        qsb_array_t: np.ndarray,    # NLDAS baseflow array for the time step t (mm/day)
        pdsi_t: float,    # PDSI array for the time step t
        doy_t: int,    # day of year for the time step t
        u_e: float,   # effective velocity [m/s], as the model parameter to calibrate
    ):
    """
    Channel routing at a given time step.
    """

    grid_area = grid_length ** 2    # grid area (km2)

    # Loop through the sorted grids
    for (i, j), upstream_grids in upstream_grid_dict.items():

        # Read new grid cell runoff for time step t
        grid[i, j]['qs'] = qs_array_t[i, j]
        grid[i, j]['qsb'] = qsb_array_t[i, j]

        # Calculate inflow to the grid cell
        # inflow from upstream grids
        grid[i, j]['inflow_from_upstream'] = np.nansum([grid[upstream_grid]['outflow_after_operation'] for upstream_grid in upstream_grids])    # [m3/s]. using np.nansum() to treat nan as 0, if there is not upstream grid
        # inflow from grid runoff
        grid[i, j]['inflow_from_grid_runoff'] = (grid[i, j]['qs'] + grid[i, j]['qsb']) / 1000 * grid_area * 1000 * 1000 / (3600*24)    # mm/day * km2 to m3/s
        # total inflow to the grid cell
        grid[i, j]['inflow_total'] = grid[i, j]['inflow_from_upstream'] + grid[i, j]['inflow_from_grid_runoff']

        # Update grid storage
        grid[i, j]['grid_storage_end'] = update_grid_storage(grid, i, j, grid_length, u_e)

        # Calculate outflow
        # outflow before reservoir operation
        grid[i, j]['outflow_before_operation'] = calculate_grid_outflow_before_operation(grid[i, j]['grid_storage_end'], grid_length, grid[i, j]['flow_distance'], u_e)
        # outflow after reservoir operation [m3/s]
        grid[i, j]['outflow_after_operation'] = calculate_grid_outflow_after_operation(
            grid[i, j]['reservoir_id'], grid[i, j]['reservoir_max_storage'], grid[i, j]['reservoir_storage_start'], pdsi_t, doy_t, grid[i, j]['outflow_before_operation'])

        # Update reservoir storage
        if grid[i, j]['reservoir_id'] > 0:    # if grid_gid not 0, i.e., the grid has a reservoir
            grid[i, j]['reservoir_storage_end'] = update_reservoir_storage(grid[i, j]['reservoir_storage_start'], grid[i, j]['outflow_before_operation'], grid[i, j]['outflow_after_operation'])            
    return grid

def save_model_state(
        grid: np.ndarray,    # structured array of the grid at the time step
        save_var_list: List[str],    # list of variables to save  
        save_nc_path: str,    # path to save the nc file
    ):
    """
    Save a given model state (grid) to nc file.
    """

    # if list is empty, don't save
    if len(save_var_list) == 0:
        return

    with nc.Dataset(save_nc_path, 'w', format='NETCDF4') as ds:
        # create dimensions
        nrows, ncols = grid.shape
        ds.createDimension('lat', nrows)
        ds.createDimension('lon', ncols)

        # create variables
        for var_name in save_var_list:
            ds.createVariable(var_name, 'f8', ('lat', 'lon'))
            ds.variables[var_name][:, :] = grid[:, :][var_name]

def run_simulation(
        run_dir: str,    # directory to save the simulation results
        grid_length: float,    # grid length (km)
        upstream_grid_dict: Dict,    # {(grid_i, grid_j): [(grid_i, grid_j)] list of connected upstream grids}. The key has been sorted by topological sorting.
        grid: np.ndarray,    # structured array of the grid at the time step
        start_date: str,    # start date of the simulation period yyyy-mm-dd
        end_date: str,    # end date of the simulation period yyyy-mm-dd
        qs_array_huc4: np.ndarray,    # NLDAS surface runoff array for the simulation period (mm/day)
        qsb_array_huc4: np.ndarray,    # NLDAS baseflow array for the simulation period (mm/day)
        pdsi_array_huc4: np.ndarray,    # PDSI array for the simulation period    
        doy_array: np.ndarray,    # day of year array for the simulation period
        save_var_list: List[str],    # list of variables to save
        u_e: float    # effective velocity [m/s], as the model parameter to calibrate
    ) -> List[np.ndarray]:    # list of model states (grids) for each time step    
    """
    Run simulation over the simulation period.
    """

    # initialize all model states to store
    states = []

    # loop through all time steps
    n_time_steps = pd.date_range(start_date, end_date).shape[0]
    for t in range(n_time_steps):

        # Update grid_storage_start for each cell based on the previous time step's grid_storage_end
        if t > 0:    # skip the first time step
            grid[:, :]['grid_storage_start'] = grid[:, :]['grid_storage_end']
            grid[:, :]['reservoir_storage_start'] = grid[:, :]['reservoir_storage_end']

        # get the model inputs for the time step
        qs_array_t = qs_array_huc4[t, :, :]
        qsb_array_t = qsb_array_huc4[t, :, :]
        pdsi_t = pdsi_array_huc4[t]
        doy_t = doy_array[t]

        # channel routing
        grid = channel_routing(grid_length, grid, upstream_grid_dict, qs_array_t, qsb_array_t, pdsi_t, doy_t, u_e)

        # append the model states
        states.append(grid)

        # save the model state to nc file
        date = pd.date_range(start_date, end_date)[t].strftime('%Y-%m-%d')
        # save_nc_path = f'/Users/donghui/Box Sync/Research/PhD/Projects/Water_Supply_Drought/data/results/lrr_output/{huc4}_model_state_{date}.nc'
        save_nc_path = os.path.join(run_dir, f'model_state_{date}.nc')
        save_model_state(grid, save_var_list, save_nc_path)

    return states

# Test
if __name__ == '__main__':

    # parameters to calibrate
    u_e = 0.6    # effective velocity [m/s]

    # constant parameters
    grid_length = 111 / 8    # grid length (km)

    # simulation period
    start_date = '1980-10-01'
    end_date = '2000-10-01'

    # read conus grid nc
    conus_grid_nc_path = '/Users/donghui/Box Sync/Research/PhD/Projects/Water_Supply_Drought/data/processed/LRR/input/conus_nldas_grid.nc'
    lat_array_conus, lon_array_conus, grid_id_array_conus, flow_dir_array_conus = read_conus_grid_nc(conus_grid_nc_path)

    # read conus reservoir nc
    conus_res_nc_path = '/Users/donghui/Box Sync/Research/PhD/Projects/Water_Supply_Drought/data/processed/LRR/input/reservoirs.nc'
    res_gid_array, res_grid_id_array, res_max_storage_array, res_lat_array, res_lon_array = read_conus_reservoir_nc(conus_res_nc_path)

    # read huc4 basin
    huc4 = '0601'
    nhd_data_dir = '/Users/donghui/Box Sync/Research/PhD/Projects/Drought_Cycle_Analysis/Data'
    gdf_huc2_conus, gdf_huc4_conus, gdf_huc4 = read_huc4_basin(huc4, nhd_data_dir)

    # get grids in huc4
    gdf_huc4_points, lon_index_array, lat_index_array = get_grids_in_hu(lon_array_conus, lat_array_conus, gdf_huc4)

    # sort grids by flow direction
    upstream_grid_dict, upstream_grid_id_dict, G, flow_dir_array_huc4 = sort_grids_by_flow_dir(flow_dir_array_conus, gdf_huc4_points, grid_id_array_conus, lat_array_conus, lon_array_conus)

    # ######## plot flow network for the huc4 ########
    # plot_flow_network(G, gdf_huc4)
    # ################################################

    # read nldas runoff
    nldas_runoff_nc_path = '/Users/donghui/Box Sync/Research/PhD/Projects/Water_Supply_Drought/data/processed/LRR/input/nldas_runoff.nc'
    nldas_qs_array, nldas_qsb_array = read_nldas_runoff(nldas_runoff_nc_path, lat_index_array, lon_index_array, start_date, end_date)    # [time, lat, lon]

    # read pdsi
    pdsi_file_path = f'/Users/donghui/Box Sync/Research/PhD/Projects/Water_Supply_Drought/data/processed/LRR/input/pdsi_{huc4}.csv'
    pdsi_array = read_pdsi(pdsi_file_path, start_date, end_date)

    # prepare doy array
    doy_array = pd.date_range(start_date, end_date).dayofyear.values

    # initialize grid
    nrows, ncols = flow_dir_array_huc4.shape
    grid_id_array_huc4 = grid_id_array_conus[np.ix_(lat_index_array, lon_index_array)]
    grid = initialize_grid(nrows, ncols, res_grid_id_array, res_gid_array, res_max_storage_array, grid_id_array_huc4, grid_length, flow_dir_array_huc4)

    # run simulation
    save_var_list = [
        'grid_id', 'reservoir_id', 'reservoir_storage_start', 'reservoir_storage_end', 'outflow_before_operation', 'outflow_after_operation', 
        'grid_storage_start', 'grid_storage_end', 'flow_direction']
    run_simulation(grid_length, grid, start_date, end_date, nldas_qs_array, nldas_qsb_array, pdsi_array, doy_array, save_var_list, u_e)

