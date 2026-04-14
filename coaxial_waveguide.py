import macromax
import numpy as np
import pyvista as pv
import matplotlib.colors as mcolors

def run_coaxial_waveguide_simulation_and_gif(inner_radius, outer_radius, dielectric_epsilon, operating_wavelength, output_filename, field_component_to_visualize):
    """Runs the macromax simulation for a coaxial waveguide and generates a 3D animated GIF.

    Args:
        inner_radius (float): Radius of the inner conductor.
        outer_radius (float): Radius of the outer conductor (inner surface).
        dielectric_epsilon (float): Relative permittivity of the dielectric between conductors.
        operating_wavelength (float): The wavelength at which the simulation is run.
        output_filename (str): Name of the GIF file to save.
        field_component_to_visualize (str): 'Ez', 'Hz', 'Er', 'Ephi', 'Hr', 'Hphi'.
    """

    print(f"Setting up simulation for Coaxial Waveguide (inner_r={inner_radius}, outer_r={outer_radius})...")

    # Waveguide Parameters
    grid_spacing = 0.05
    # Dimensions: (Length, Width, Height) corresponding to (Z, X, Y) in macromax
    # X and Y dimensions must accommodate the outer radius
    shape_z = 100
    shape_xy = int(2 * outer_radius / grid_spacing) + 2 # Ensure enough space for outer conductor and odd points
    if shape_xy % 2 == 0:
        shape_xy += 1
    shape = (shape_z, shape_xy, shape_xy)

    extent = np.array(shape) * grid_spacing

    axis_z = np.linspace(0, extent[0], shape_z)
    axis_x = np.linspace(-extent[1]/2, extent[1]/2, shape_xy)
    axis_y = np.linspace(-extent[2]/2, extent[2]/2, shape_xy)

    # Define the Material (Coaxial Structure)
    permittivity = np.ones(shape, dtype=complex) * (1.0 + 50j) # Default is metal

    # Create a meshgrid for x and y coordinates relative to the center
    X, Y = np.meshgrid(axis_x, axis_y, indexing='ij')
    R_coords = np.sqrt(X**2 + Y**2)

    # Apply dielectric permittivity in the region between inner and outer conductors
    for z_idx in range(shape_z):
        permittivity[z_idx, (R_coords >= inner_radius) & (R_coords < outer_radius)] = dielectric_epsilon

    # Source definition for TEM mode approximation
    source_density = np.zeros((3,) + shape, dtype=complex)
    source_z_slice = 5

    # Approximate TEM mode excitation: a radial E-field.
    # We'll use a current source component (e.g., Ey) placed between the conductors
    # at the source plane, aiming to create a radial E-field.
    intermediate_radius = (inner_radius + outer_radius) / 2
    idx_y_source = np.argmin(np.abs(axis_y - intermediate_radius))
    idx_x_center = np.argmin(np.abs(axis_x - 0))

    # This attempts to create an Ey field. It's a simplification.
    source_density[1, source_z_slice, idx_x_center, idx_y_source] = 1.0 # Ey component current

    print(f"Exciting with an approximate Ey current source at (X={axis_x[idx_x_center]:.2f}, Y={axis_y[idx_y_source]:.2f}) for TEM-like mode.")


    # Run the Physics Solver
    print("Solving Maxwell Equations... please wait.")
    solution = macromax.solve(
        (axis_z, axis_x, axis_y),  # Pass tuple of coordinate arrays as 'grid'
        epsilon=permittivity,
        current_density=source_density,
        vacuum_wavelength=operating_wavelength # Use the given operating wavelength
    )

    # --- Generate 3D Animated GIF ---
    print(f"Generating 3D animation for {field_component_to_visualize}...")
    field_data_complex = None
    field_label = "Field Component"

    Ex_complex = solution.E[0, :, :, :].astype(np.complex64)
    Ey_complex = solution.E[1, :, :, :].astype(np.complex64)
    Ez_complex = solution.E[2, :, :, :].astype(np.complex64)
    Hx_complex = solution.H[0, :, :, :].astype(np.complex64)
    Hy_complex = solution.H[1, :, :, :].astype(np.complex64)
    Hz_complex = solution.H[2, :, :, :].astype(np.complex64)

    # Pre-calculate cylindrical coordinates for visualization if needed
    angles = np.arctan2(Y, X) # Angle phi

    if field_component_to_visualize == 'Ez':
        field_data_complex = Ez_complex
        print("Visualizing Ez component.")
    elif field_component_to_visualize == 'Hz':
        field_data_complex = Hz_complex
        print("Visualizing Hz component.")
    elif field_component_to_visualize == 'Er': # Radial Electric field
        field_data_complex = Ex_complex * np.cos(angles) + Ey_complex * np.sin(angles)
        print("Visualizing Er component.")
    elif field_component_to_visualize == 'Ephi': # Azimuthal Electric field
        field_data_complex = -Ex_complex * np.sin(angles) + Ey_complex * np.cos(angles)
        print("Visualizing Ephi component.")
    elif field_component_to_visualize == 'Hr': # Radial Magnetic field
        field_data_complex = Hx_complex * np.cos(angles) + Hy_complex * np.sin(angles)
        print("Visualizing Hr component.")
    elif field_component_to_visualize == 'Hphi': # Azimuthal Magnetic field
        field_data_complex = -Hx_complex * np.sin(angles) + Hy_complex * np.cos(angles)
        print("Visualizing Hphi component.")
    else:
        raise ValueError("Invalid field_component_to_visualize. Must be 'Ez', 'Hz', 'Er', 'Ephi', 'Hr', or 'Hphi'.")


    grid = pv.ImageData(dimensions=shape, spacing=(grid_spacing, grid_spacing, grid_spacing))

    plotter = pv.Plotter(off_screen=True, window_size=[800, 500])
    plotter.open_gif(output_filename)
    plotter.camera_position = [(12, 6, 6), (2.5, 1, 0.5), (0, 0, 1)]

    colors = ["blue", "white", "red"]
    nodes = [0.0, 0.5, 1.0]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_bluewhite_red", list(zip(nodes, colors)))

    frames = 50
    max_abs_field = np.max(np.abs(field_data_complex))
    if max_abs_field == 0:
        max_abs_field = 1.0

    label_text = f"Coaxial Field: {field_component_to_visualize} - Wavelength: {operating_wavelength:.3f}"

    for phase in np.linspace(0, 2 * np.pi, frames, endpoint=False):
        snapshot = np.real(field_data_complex * np.exp(-1j * phase)) / max_abs_field
        grid.point_data[field_label] = snapshot.flatten(order="F")
        plotter.clear()
        plotter.add_mesh(grid.outline(), color="black")
        plotter.add_mesh(
            grid.contour(isosurfaces=10, scalars=field_label),
            cmap=custom_cmap,
            clim=[-1.0, 1.0],
            opacity=0.7,
            smooth_shading=True
        )
        plotter.add_text(label_text, position='upper_left', color='black', font_size=10)

        plotter.write_frame()

    plotter.close()
    print(f"Done! Animation saved as '{output_filename}'.")

# --- Example Configuration for Coaxial TEM GIF ---
coaxial_tem_config = {
    'inner_radius': 0.2, # Radius of the inner conductor
    'outer_radius': 0.8, # Radius of the outer conductor (inner surface)
    'dielectric_epsilon': 1.0, # Air dielectric
    'operating_wavelength': 0.5 # A wavelength that would propagate as TEM
}

# Visualize radial E-field (Er)
run_coaxial_waveguide_simulation_and_gif(
    coaxial_tem_config['inner_radius'],
    coaxial_tem_config['outer_radius'],
    coaxial_tem_config['dielectric_epsilon'],
    coaxial_tem_config['operating_wavelength'],
    f"waveguide_3d_dynamic_Coaxial_TEM_Er.gif",
    'Er'
)

# Visualize azimuthal H-field (Hphi)
run_coaxial_waveguide_simulation_and_gif(
    coaxial_tem_config['inner_radius'],
    coaxial_tem_config['outer_radius'],
    coaxial_tem_config['dielectric_epsilon'],
    coaxial_tem_config['operating_wavelength'],
    f"waveguide_3d_dynamic_Coaxial_TEM_Hphi.gif",
    'Hphi'
)
