import os
import numpy as np
import xarray as xr
import pyvista as pv
from pathlib import Path
from typing import List


def openfoam_to_netcdf(
    case_dir: str,
    output_dir: str,
    variables: List[str],
    include_boundaries: bool = False
) -> dict:
    """
    Reads 3D OpenFOAM data and converts it into CF and UGRID compliant NetCDF files.
    
    This function reads native OpenFOAM unstructured polyhedral meshes and converts 
    them into the standard UGRID-1.0 topology structure. It saves each requested
    variable into a separate NetCDF file named {variable_name}.nc.
    
    Args:
        case_dir (str): Path to the OpenFOAM case.
        output_dir (str): Directory where the output NetCDF files will be saved.
        variables (list): Variables to extract (e.g., ['U', 'alpha.water']).
        include_boundaries (bool): If True, also extract and store boundary patch data.
        
    Returns:
        dict: A dictionary of generated xarray Datasets keyed by variable name.
    """
    case_path = Path(case_dir)
    foam_file = case_path / "openfoam.foam"
    
    if not foam_file.exists():
        foam_file.touch()
            
    print(f"Reading OpenFOAM case: {case_dir}")
    reader = pv.OpenFOAMReader(str(foam_file))
    time_values = reader.time_values
    
    if not time_values:
        raise ValueError("No time directories found in the OpenFOAM case.")
        
    reader.set_active_time_value(time_values[0])
    mesh = reader.read()
    internal_mesh = mesh["internalMesh"]
    
    # =========================================================
    # INTERNAL MESH TOPOLOGY
    # =========================================================
    nodes = internal_mesh.points
    n_nodes = len(nodes)
    n_cells = internal_mesh.n_cells
    
    print(f"Internal Mesh loaded: {n_cells} cells, {n_nodes} nodes.")
    print("Building UGRID internal topology...")
    
    try:
        offsets = internal_mesh.offset
        conn_flat = internal_mesh.cell_connectivity
        nodes_per_cell = np.diff(offsets)
        max_nodes = nodes_per_cell.max()
        connectivity = np.full((n_cells, max_nodes), -1, dtype=np.int32)
        mask = np.arange(max_nodes) < nodes_per_cell[:, None]
        connectivity[mask] = conn_flat
    except AttributeError:
        max_nodes = 0
        cell_nodes_list = []
        for i in range(n_cells):
            pt_ids = internal_mesh.get_cell(i).point_ids
            cell_nodes_list.append(pt_ids)
            if len(pt_ids) > max_nodes:
                max_nodes = len(pt_ids)
        connectivity = np.full((n_cells, max_nodes), -1, dtype=np.int32)
        for i, pt_ids in enumerate(cell_nodes_list):
            connectivity[i, :len(pt_ids)] = pt_ids
        
    # =========================================================
    # BOUNDARY MESH TOPOLOGY
    # =========================================================
    bnd_connectivity = None
    if include_boundaries and "boundary" in mesh.keys():
        print("Building UGRID boundary topology...")
        boundary_blocks = mesh["boundary"]
        
        bnd_nodes_list, bnd_face_nodes_list, bnd_face_patch_idx, bnd_centers_list, patch_names = [], [], [], [], []
        node_offset = 0
        
        for patch_idx, patch_name in enumerate(boundary_blocks.keys()):
            patch_names.append(patch_name)
            patch_mesh = boundary_blocks[patch_name]
            p_nodes = patch_mesh.points
            bnd_nodes_list.append(p_nodes)
            bnd_centers_list.append(patch_mesh.cell_centers().points)
            
            p_n_faces = patch_mesh.n_cells
            for i in range(p_n_faces):
                pt_ids = patch_mesh.get_cell(i).point_ids
                global_pt_ids = [pid + node_offset for pid in pt_ids]
                bnd_face_nodes_list.append(global_pt_ids)
                bnd_face_patch_idx.append(patch_idx)
            node_offset += len(p_nodes)
            
        if len(bnd_nodes_list) > 0:
            bnd_nodes_array = np.vstack(bnd_nodes_list)
            bnd_centers_array = np.vstack(bnd_centers_list)
            bnd_faces = len(bnd_face_nodes_list)
            bnd_max_nodes = max(len(pts) for pts in bnd_face_nodes_list)
            bnd_connectivity = np.full((bnd_faces, bnd_max_nodes), -1, dtype=np.int32)
            for i, pt_ids in enumerate(bnd_face_nodes_list):
                bnd_connectivity[i, :len(pt_ids)] = pt_ids
            print(f"Boundary Mesh loaded: {bnd_faces} faces across {len(patch_names)} patches.")
        
    # =========================================================
    # CREATE BASE XARRAY DATASET
    # =========================================================
    base_ds = xr.Dataset()
    base_ds.coords['time'] = ('time', time_values, {'standard_name': 'time', 'units': 's'})
    
    base_ds['node_x'] = ('node', nodes[:, 0], {'standard_name': 'projection_x_coordinate', 'units': 'm'})
    base_ds['node_y'] = ('node', nodes[:, 1], {'standard_name': 'projection_y_coordinate', 'units': 'm'})
    base_ds['node_z'] = ('node', nodes[:, 2], {'standard_name': 'projection_z_coordinate', 'units': 'm'})
    
    cell_centers = internal_mesh.cell_centers().points
    base_ds['volume_x'] = ('volume', cell_centers[:, 0], {'standard_name': 'projection_x_coordinate', 'units': 'm'})
    base_ds['volume_y'] = ('volume', cell_centers[:, 1], {'standard_name': 'projection_y_coordinate', 'units': 'm'})
    base_ds['volume_z'] = ('volume', cell_centers[:, 2], {'standard_name': 'projection_z_coordinate', 'units': 'm'})
    
    base_ds['mesh_topology'] = ((), 0, {
        'cf_role': 'mesh_topology',
        'long_name': 'Topology data of 3D unstructured mesh',
        'topology_dimension': 3,
        'node_coordinates': 'node_x node_y node_z',
        'volume_node_connectivity': 'volume_node_connectivity'
    })
    base_ds['volume_node_connectivity'] = (
        ('volume', 'max_node_per_volume'), connectivity, 
        {'cf_role': 'volume_node_connectivity', 'start_index': 0, '_FillValue': -1}
    )
    
    if bnd_connectivity is not None:
        base_ds['bnd_node_x'] = ('bnd_node', bnd_nodes_array[:, 0], {'units': 'm'})
        base_ds['bnd_node_y'] = ('bnd_node', bnd_nodes_array[:, 1], {'units': 'm'})
        base_ds['bnd_node_z'] = ('bnd_node', bnd_nodes_array[:, 2], {'units': 'm'})
        
        base_ds['bnd_face_x'] = ('bnd_face', bnd_centers_array[:, 0], {'units': 'm'})
        base_ds['bnd_face_y'] = ('bnd_face', bnd_centers_array[:, 1], {'units': 'm'})
        base_ds['bnd_face_z'] = ('bnd_face', bnd_centers_array[:, 2], {'units': 'm'})
        
        base_ds['bnd_topology'] = ((), 0, {
            'cf_role': 'mesh_topology',
            'long_name': 'Topology data of 2D unstructured boundary mesh',
            'topology_dimension': 2,
            'node_coordinates': 'bnd_node_x bnd_node_y bnd_node_z',
            'face_node_connectivity': 'bnd_face_node_connectivity'
        })
        base_ds['bnd_face_node_connectivity'] = (
            ('bnd_face', 'max_node_per_bnd_face'), bnd_connectivity,
            {'cf_role': 'face_node_connectivity', 'start_index': 0, '_FillValue': -1}
        )
        base_ds['bnd_patch_name'] = ('bnd_patch', patch_names)
        base_ds['bnd_face_patch_id'] = ('bnd_face', bnd_face_patch_idx, {'long_name': 'Index mapping face to bnd_patch_name'})
    
    # =========================================================
    # EXTRACT DATA VARIABLES OVER TIME
    # =========================================================
    print("Extracting time-varying fields...")
    import json
    openfoam_variables_file = Path(__file__).parent / "openfoam_variables.json"
    if openfoam_variables_file.exists():
        with open(openfoam_variables_file, "r") as f:
            KNOWN_METADATA = json.load(f)
    else:
        KNOWN_METADATA = {}

    type_metadata = KNOWN_METADATA.get('__types__', {})

    var_shapes = {}
    for var in variables:
        if var in KNOWN_METADATA and 'openfoam_type' in KNOWN_METADATA[var]:
            oftype = KNOWN_METADATA[var]['openfoam_type']
            var_shapes[var] = type_metadata.get(oftype, {}).get('components', 1)
        else:
            print(f"Warning: Variable '{var}' not found in openfoam_variables.json. Defaulting to scalar.")
            var_shapes[var] = 1
            
    def get_suffixes(var, n):
        if var in KNOWN_METADATA and 'openfoam_type' in KNOWN_METADATA[var]:
            oftype = KNOWN_METADATA[var]['openfoam_type']
            if oftype in type_metadata:
                suffixes = type_metadata[oftype].get('suffixes', [])
                if len(suffixes) == n or (len(suffixes) == 0 and n == 1):
                    return suffixes
        return [] if n == 1 else [str(i) for i in range(n)]

    # Create separate datasets for each variable
    datasets = {var: base_ds.copy() for var in var_shapes.keys()}

    for var, n_comp in var_shapes.items():
        ds = datasets[var]
        attrs = {'mesh': 'mesh_topology', 'location': 'volume'}
        bnd_attrs = {'mesh': 'bnd_topology', 'location': 'face'}
        
        if var in KNOWN_METADATA:
            for k, v in KNOWN_METADATA[var].items():
                if k not in ['components', 'openfoam_type']:
                    attrs[k] = v
                    bnd_attrs[k] = v

        suffixes = get_suffixes(var, n_comp)

        if not suffixes:
            ds[var] = (('time', 'volume'), np.zeros((len(time_values), n_cells), dtype=np.float32), attrs)
            if bnd_connectivity is not None:
                ds[f'{var}_bnd'] = (('time', 'bnd_face'), np.zeros((len(time_values), bnd_faces), dtype=np.float32), bnd_attrs)
        else:
            for suf in suffixes:
                suf_attrs = attrs.copy()
                suf_bnd_attrs = bnd_attrs.copy()
                if var in KNOWN_METADATA and 'components' in KNOWN_METADATA[var]:
                    comp_metadata = KNOWN_METADATA[var]['components'].get(suf, {})
                    suf_attrs.update(comp_metadata)
                    suf_bnd_attrs.update(comp_metadata)
                ds[f'{var}_{suf}'] = (('time', 'volume'), np.zeros((len(time_values), n_cells), dtype=np.float32), suf_attrs)
                if bnd_connectivity is not None:
                    ds[f'{var}_bnd_{suf}'] = (('time', 'bnd_face'), np.zeros((len(time_values), bnd_faces), dtype=np.float32), suf_bnd_attrs)

    for t_idx, t_val in enumerate(time_values):
        reader.set_active_time_value(t_val)
        mesh = reader.read()
        internal_mesh = mesh["internalMesh"]
        
        for var, n_comp in var_shapes.items():
            ds = datasets[var]
            if var in internal_mesh.cell_data:
                field = internal_mesh.cell_data[var]
                suffixes = get_suffixes(var, n_comp)
                if not suffixes:
                    if len(field.shape) > 1 and field.shape[1] == 1:
                        ds[var][t_idx, :] = field[:, 0]
                    else:
                        ds[var][t_idx, :] = field
                else:
                    for i, suf in enumerate(suffixes):
                        if len(field.shape) == 1:
                            ds[f'{var}_{suf}'][t_idx, :] = field
                        else:
                            ds[f'{var}_{suf}'][t_idx, :] = field[:, i]
                    
        if bnd_connectivity is not None and "boundary" in mesh.keys():
            boundary_blocks = mesh["boundary"]
            for var, n_comp in var_shapes.items():
                ds = datasets[var]
                bnd_values = []
                for patch_name in patch_names:
                    patch_mesh = boundary_blocks[patch_name]
                    if var in patch_mesh.cell_data:
                        bnd_values.append(patch_mesh.cell_data[var])
                    else:
                        if n_comp == 1:
                            bnd_values.append(np.full(patch_mesh.n_cells, np.nan))
                        else:
                            bnd_values.append(np.full((patch_mesh.n_cells, n_comp), np.nan))
                            
                bnd_values_array = np.vstack(bnd_values) if n_comp > 1 else np.concatenate(bnd_values)
                
                suffixes = get_suffixes(var, n_comp)
                if not suffixes:
                    ds[f'{var}_bnd'][t_idx, :] = bnd_values_array
                else:
                    for i, suf in enumerate(suffixes):
                        if bnd_values_array.ndim == 1:
                            ds[f'{var}_bnd_{suf}'][t_idx, :] = bnd_values_array
                        else:
                            ds[f'{var}_bnd_{suf}'][t_idx, :] = bnd_values_array[:, i]
                    
        if total_to_process := len(time_values):
            milestone = max(1, total_to_process // 10)
            if (t_idx + 1) % milestone == 0:
                percent = int(((t_idx + 1) / total_to_process) * 100)
                print(f"  -> Progress: {percent}%")
            
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for var, ds in datasets.items():
        ds.attrs['Conventions'] = 'CF-1.8 UGRID-1.0'
        ds.attrs['title'] = f'OpenFOAM 3D Volume and Boundary Data: {var}'
        ds.attrs['source'] = 'Generated by toolbox4modeler'
        
        var_nc = out_path / f"{var}.nc"
        print(f"Saving {var} to {var_nc}...")
        ds.to_netcdf(str(var_nc), engine='netcdf4')
        
    print("Success!")
    return datasets
