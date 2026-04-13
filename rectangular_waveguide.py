import macromax
import numpy as np
import pyvista as pv
import matplotlib.colors as mcolors # Import matplotlib.colors
import imageio

def run_waveguide_simulation_and_gif(mode_config, output_filename, field_component_to_visualize):

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
    # Core is air, and the subtraction of 2* a or b implies the removal of material from a metal block.
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
    if mode_type == 'TE':
        # For TE modes, excite with a Y-polarized current source (creates Ey, Hx, Hz)
        source_density[1, 5, a:-a, b:-b] = 1.0
    elif mode_type == 'TM':
        # For TM modes, excite with a Z-polarized current source (creates Ez, Hx, Hy)
        source_density[2, 5, a:-a, b:-b] = 1.0

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
    if field_component_to_visualize == 'Ey': 
        field_data_complex = solution.E[1, :, :, :].astype(np.complex64) # Extract Ey, 0 for x; 1 for y and 2 for z
        print("Visualizing Ey component.")
    elif field_component_to_visualize == 'Hx': 
        field_data_complex = solution.H[0, :, :, :].astype(np.complex64) # Extract Hx, 0 for x; 1 for y and 2 for z
        print("Visualizing Hx component.")
    else:
        raise ValueError("Invalid field_component_to_visualize. Must be 'Ey', or 'Hx'.")

    grid = pv.ImageData(dimensions=shape, spacing=(grid_spacing, grid_spacing, grid_spacing))

    plotter = pv.Plotter(off_screen=True, window_size=[800, 500])
    plotter.open_gif(output_filename)
    plotter.camera_position = [(12, 6, 6), (2.5, 1, 0.5), (0, 0, 1)]

    # 3-color colormap for negative, zero, and positive values
    colors = ["blue", "lightgray", "red"]
    nodes = [0.0, 0.5, 1.0]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_bluewhite_red", list(zip(nodes, colors)))

    frames = 50
    # Calculate max absolute field once for normalization
    max_abs_field = np.max(np.abs(field_data_complex))
    if max_abs_field == 0: # Avoid division by zero if field is identically zero
        max_abs_field = 1.0

    # Prepare the label text for the GIF
    label_text = f"Mode: {mode_type}{n_mode}{m_mode} - Field: {field_component_to_visualize}"

    for phase in np.linspace(0, 2 * np.pi, frames, endpoint=False):
        # Normalize the snapshot by the maximum absolute field value
        snapshot = np.real(field_data_complex * np.exp(-1j * phase)) / max_abs_field
        grid.point_data["Field Component"] = snapshot.flatten(order="F")
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
run_waveguide_simulation_and_gif(te_config_ey, f"waveguide_3d_dynamic_TE{te_config_ey['n_mode']}{te_config_ey['m_mode']}_Ey.gif", 'Ey')

# --- Configuration for Hx (TM) GIF ---
tm_config_hx = {
    'mode_type': 'TM',
    'n_mode': 1,
    'm_mode': 1,
    'sp_cutoff_ratio': 0.8
}
run_waveguide_simulation_and_gif(tm_config_hx, f"waveguide_3d_dynamic_TM{tm_config_hx['n_mode']}{tm_config_hx['m_mode']}_Hx.gif", 'Hx')
