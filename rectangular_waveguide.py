import macromax
import numpy as np
import pyvista as pv
import matplotlib.colors as mcolors # Import matplotlib.colors
import imageio

def run_waveguide_simulation_and_gif(mode_config, output_filename, field_component_to_visualize):
    """Runs the macromax simulation and generates a 3D animated GIF.

    Args:
        mode_config (dict): Dictionary containing 'mode_type', 'n_mode', 'm_mode', 'sp_cutoff_ratio'.
        output_filename (str): Name of the GIF file to save.
        field_component_to_visualize (str): 'Ey', or 'Hx'.
    """

    mode_type = mode_config['mode_type']
    n_mode = mode_config['n_mode']
    m_mode = mode_config['m_mode']
    sp_cutoff_ratio = mode_config['sp_cutoff_ratio']

    # Waveguide Parameters
    grid_spacing = 0.05
    shape = (100, 40, 20)
    extent = np.array(shape) * grid_spacing

    axis_z = np.linspace(0, extent[0], shape[0])
    axis_x = np.linspace(0, extent[1], shape[1])
    axis_y = np.linspace(0, extent[2], shape[2])

    # Define the Material (Metal Box with Air Core)
    permittivity = np.ones(shape, dtype=complex) * (1.0 + 50j)
    a = 4
    b = 5
    permittivity[:, a:-a, b:-b] = 1.0

    # Calculate the physical dimensions of the waveguide core
    core_x_dimension = (shape[1] - 2 * a) * grid_spacing
    core_y_dimension = (shape[2] - 2 * b) * grid_spacing

    # Calculate cutoff wavelength
    lambda_c = 0.0
    if mode_type == 'TE':
        if n_mode == 0 and m_mode == 0:
            lambda_c = 1e9
        else:
            lambda_c = 2 / np.sqrt((n_mode / core_x_dimension)**2 + (m_mode / core_y_dimension)**2)
    elif mode_type == 'TM':
        if n_mode == 0 or m_mode == 0:
            lambda_c = 1e9
        else:
            lambda_c = 2 / np.sqrt((n_mode / core_x_dimension)**2 + (m_mode / core_y_dimension)**2)

    wavelength = sp_cutoff_ratio * lambda_c
    print(f"Calculated vacuum wavelength for {mode_type}{n_mode}{m_mode} mode: {wavelength:.3f} units")

    # Define the Source
    source_density = np.zeros((3,) + shape, dtype=complex)
    #    if mode_type == 'TE':
        # For TE modes, excite with a Y-polarized current source (creates Ey, Hx, Hz)
    #   source_density[1, 5, a:-a, b:-b] = 1.0
    #elif mode_type == 'TM':
        # For TM modes, excite with a Z-polarized current source (creates Ez, Hx, Hy)
    #   source_density[2, 5, a:-a, b:-b] = 1.0

    source_z_slice = 5

    # Get coordinates and dimensions for the core region
    x_values_in_core = axis_x[a:shape[1]-a]
    y_values_in_core = axis_y[b:shape[2]-b]

    # Shifted coordinates (from 0 to W or H) for sinusoidal profiles
    x_prime = x_values_in_core - x_values_in_core[0]
    y_prime = y_values_in_core - y_values_in_core[0]

    # Generate 2D source profile based on mode type and numbers
    source_2d_profile = np.zeros((len(x_values_in_core), len(y_values_in_core)), dtype=complex)

    if mode_type == 'TE':
        # For TE modes, use a Y-polarized current source (couples to Ey)
        # Ey field profile for TE_nm is typically sin(n*pi*x'/W) * cos(m*pi*y'/H)
        # Handle n_mode = 0 or m_mode = 0 cases appropriately for the profiles
        if n_mode == 0:
            source_profile_x = np.ones_like(x_prime)
        else:
            source_profile_x = np.sin(n_mode * np.pi * x_prime / core_x_dimension)

        if m_mode == 0:
            source_profile_y = np.ones_like(y_prime)
        else:
            source_profile_y = np.cos(m_mode * np.pi * y_prime / core_y_dimension)

        source_2d_profile = np.outer(source_profile_x, source_profile_y)
        # Apply to the Y-component of source_density
        for i_x, global_x_idx in enumerate(np.arange(a, shape[1] - a)):
            for i_y, global_y_idx in enumerate(np.arange(b, shape[2] - b)):
                source_density[1, source_z_slice, global_x_idx, global_y_idx] = source_2d_profile[i_x, i_y]

    elif mode_type == 'TM':
        # For TM modes, use a Z-polarized current source (couples to Ez)
        # Ez field profile for TM_nm is typically sin(n*pi*x'/W) * sin(m*pi*y'/H)
        # TM modes require n,m >= 1, so no n_mode=0 or m_mode=0 check needed for non-zero profiles
        source_profile_x = np.sin(n_mode * np.pi * x_prime / core_x_dimension)
        source_profile_y = np.sin(m_mode * np.pi * y_prime / core_y_dimension)

        source_2d_profile = np.outer(source_profile_x, source_profile_y)
        # Apply to the Z-component of source_density
        for i_x, global_x_idx in enumerate(np.arange(a, shape[1] - a)):
            for i_y, global_y_idx in enumerate(np.arange(b, shape[2] - b)):
                source_density[2, source_z_slice, global_x_idx, global_y_idx] = source_2d_profile[i_x, i_y]
    
    # Normalize the source profile to prevent very large values in the solver
    max_abs_source = np.max(np.abs(source_density))
    if max_abs_source == 0:
        max_abs_source = 1.0 # Avoid division by zero if source is identically zero
    source_density /= max_abs_source

    # Run the Physics Solver
    print("Solving Maxwell Equations... please wait.")
    solution = macromax.solve(
        (axis_z, axis_x, axis_y),
        epsilon=permittivity,
        current_density=source_density,
        vacuum_wavelength=wavelength
    )

    # Generate 3D Animated GIF
    print(f"Generating 3D animation for {field_component_to_visualize}...")
    field_data_complex = None
    if field_component_to_visualize == 'Ey': # New transverse component for E-field
        field_data_complex = solution.E[1, :, :, :].astype(np.complex64) # Extract Ey
        print("Visualizing Ey component.")
    elif field_component_to_visualize == 'Hx': # New transverse component for H-field
        field_data_complex = solution.H[0, :, :, :].astype(np.complex64) # Extract Hx
        print("Visualizing Hx component.")
    else:
        raise ValueError("Invalid field_component_to_visualize. Must be 'Ey', or 'Hx'.")

    grid = pv.ImageData(dimensions=(shape[1], shape[2], shape[0]), spacing=(grid_spacing, grid_spacing, grid_spacing))

    plotter = pv.Plotter(off_screen=True, window_size=[800, 500])
    plotter.open_gif(output_filename)
    plotter.camera_position = [(6, 6, 12), (1, 0.5, 2.5), (0, 1, 0)]
    plotter.add_axes ()

    # Define a custom 3-color colormap for negative, zero, and positive values
    colors = ["blue", "white", "red"]
    nodes = [0.0, 0.5, 1.0]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_bluewhite_red", list(zip(nodes, colors)))

    frames = 50
    # Calculate max absolute field once for normalization
    max_abs_field = np.max(np.abs(np.real(field_data_complex)))
    if max_abs_field == 0: # Avoid division by zero if field is identically zero
        max_abs_field = 1.0

    # Prepare the label text for the GIF
    label_text = f"Mode: {mode_type}{n_mode}{m_mode} - Field: {field_component_to_visualize}"

    for phase in np.linspace(0, 2 * np.pi, frames, endpoint=False):
        # Normalize the snapshot by the maximum absolute field value
        snapshot = np.real(field_data_complex * np.exp(-1j * phase)) / max_abs_field
        grid.point_data["Field Component"] = snapshot.transpose((1, 2, 0)).flatten(order="F")
        plotter.clear()
        plotter.add_mesh(grid.outline(), color="black")
        plotter.add_mesh(
            grid.contour(isosurfaces=10, scalars="Field Component"),
            cmap=custom_cmap, # Use the custom colormap
            clim=[-1.0, 1.0], # Set clim to -1 to 1 for normalized values
            opacity=0.7,
            smooth_shading=True
        )
        # Add text label to the GIF frame
        plotter.add_text(label_text, position='upper_left', color='black', font_size=10)

        plotter.write_frame()

    plotter.close()
    print(f"Done! Animation saved as '{output_filename}'.")

# --- Configuration for Ey (TE) GIF ---
te_config_ey = {
    'mode_type': 'TE',
    'n_mode': 1,
    'm_mode': 1,
    'sp_cutoff_ratio': 0.8
}
run_waveguide_simulation_and_gif(te_config_ey, f"TE{te_config_ey['n_mode']}{te_config_ey['m_mode']}_Ey.gif", 'Ey')

# --- Configuration for Hx (TM) GIF ---
tm_config_hx = {
    'mode_type': 'TM',
    'n_mode': 1,
    'm_mode': 1,
    'sp_cutoff_ratio': 0.8
}
run_waveguide_simulation_and_gif(tm_config_hx, f"TM{tm_config_hx['n_mode']}{tm_config_hx['m_mode']}_Hx.gif", 'Hx')
