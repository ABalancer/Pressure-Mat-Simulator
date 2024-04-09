import tkinter as tk
import numpy as np
from scipy import constants
from tkinter import ttk

'''
TODO:
Disable real time simulation. Create Run simulation button / keybind.

Implement mesh analysis into simulator

Create test scenarios (use ovals for feet)

get data to fill table from simulations


later:

make objects configurable
make test scenarios
display numerical CoP values as well as error

prevent overlaps and moving objects out of the border - also reset load positions if they leave the screen
pause program when moving window to get rid of lag
check for memory leaks
'''

GRID_SIZE = 16
GRID_HEIGHT = 500
GRID_WIDTH = 500
TRACK_WIDTH = 10
PITCH_WIDTH = 10

APPROXIMATE_FORCE = False

colour_interpolation_values = [
    (13, 22, 135), (45, 25, 148), (66, 29, 158), (90, 32, 165), (112, 34, 168),
    (130, 35, 167), (148, 35, 161), (167, 36, 151), (182, 48, 139), (196, 63, 127),
    (208, 77, 115), (220, 93, 102), (231, 109, 92), (239, 126, 79), (247, 143, 68),
    (250, 160, 58), (254, 181, 44), (253, 202, 40), (247, 226, 37), (240, 249, 32)
]


def interpolate_colours(value):
    if value < 4095:
        colour_steps = len(colour_interpolation_values) - 1
        step = 4095 / colour_steps
        start_step = int(value // step)
        end_step = min(start_step + 1, colour_steps)

        start_color = colour_interpolation_values[start_step]
        end_color = colour_interpolation_values[end_step]

        start_r, start_g, start_b = start_color
        end_r, end_g, end_b = end_color

        start_value = start_step * step
        end_value = end_step * step

        ratio = (value - start_value) / (end_value - start_value)
        red = int(start_r + (end_r - start_r) * ratio)
        green = int(start_g + (end_g - start_g) * ratio)
        blue = int(start_b + (end_b - start_b) * ratio)
    else:
        red, green, blue = colour_interpolation_values[-1]
    return f'#{red:02x}{green:02x}{blue:02x}'  # Convert RGB values to hexadecimal color code


def create_colourmap():
    colour_array = []
    for i in range(0, 4096):
        colour_array.append(interpolate_colours(i))
    return colour_array


def create_widget(parent, widget_type, *args, **kwargs):
    widget = widget_type(parent, *args, **kwargs)
    widget.config(background="#2b2b2b", borderwidth=0, relief=tk.FLAT)
    # Apply the styling based on the current mode (light/dark)
    if widget_type is tk.Canvas:
        widget.config(highlightthickness=0)
    if widget_type is tk.Label or widget_type is tk.Listbox or widget_type is tk.Button:
        if "foreground" in kwargs:
            foreground = kwargs['foreground']
        else:
            foreground = "#a8b5c4"
        widget.config(foreground=foreground, font=("Helvetica", 12))
    if widget_type is tk.Button:
        widget.config(highlightbackground="#2b2b2b", activebackground="#485254",
                      activeforeground="#a8b5c4", background="#3c3f41", width=17, padx=2, pady=2)
    if widget_type is tk.Listbox:
        widget.config(exportselection=False, background="#3c3f41")
    if widget_type is tk.Entry:
        widget.config(background="#3c3f41", foreground="#a8b5c4", insertbackground="#a8b5c4")
    return widget


class Load:
    def __init__(self, reference, centre_x, centre_y, mass, area):
        self.reference = reference
        self._centre_x = centre_x
        self._centre_y = centre_y
        self._mass = mass
        self._area = area

    def update_location(self, centre_x, centre_y):
        self._centre_x = centre_x
        self._centre_y = centre_y

    def update_load(self, mass, area):
        self._mass = mass
        self._area = area

    def get_location(self):
        return self._centre_x, self._centre_y

    def compute_pressure(self):
        return constants.g*self._mass/self._area

    def compute_x_moment(self):
        return self._mass * self._centre_x

    def compute_y_moment(self):
        return self._mass * self._centre_y

    def get_mass(self):
        return self._mass


class Sensor:
    def __init__(self, reference, sensor_area, pdr, r0, k):
        self.reference = reference
        self._sensor_area = sensor_area
        self._pdr = pdr
        self._r0 = r0
        self._k = k

    def update_sensor(self, sensor_area, pdr, r0, k):
        self._sensor_area = sensor_area
        self._pdr = pdr
        self._r0 = r0
        self._k = k

    def compute_resistance(self, pressure):
        resistance = (self._r0 / self._sensor_area) * np.exp(-self._k * pressure)
        return resistance

    # Through the potential divider the voltage has a somewhat linear relationship with force up to about 100kN
    # This means these ADC values are an approximation for force and are not always accurate. This is the default.
    # If approximate=False, then the correct force values will be mapped to the 12bit ADC output.
    def compute_adc_value(self, pressure, approximate=True):
        z1 = self.compute_resistance(pressure)
        if approximate:  # uses the potential divider to estimate the force rather than the equation.
            z2 = self._pdr
            voltage = z2/(z1+z2)
            adc_value = round(4095*voltage)
        else:  # returns the real value of the force using the equation.
            force = np.log(z1/80000)/-self._k
            if force > 200000:
                force = 200000
            adc_value = round(4095*force/200000)
        return adc_value


# Draws the left grid and the configurable loads for creating a simulation
class SimulationSetup:
    def __init__(self, canvas, rows, columns):
        self.canvas = canvas
        self.canvas_width = canvas.winfo_reqwidth()
        self.canvas_height = canvas.winfo_reqheight()

        # Default parameters
        self._sensors = []
        self._loads = []
        self._num_rows = rows
        self._num_columns = columns
        self._track_width_mm = 10  # in mm
        self._gap_width_mm = 10
        self._spacing_ratio = 0
        self._track_width_pixel = 0
        self._spacing_width_pixel = 0
        self._pixel_ratio = 0
        # Multiply real width by pixel ratio to get a pixel width.
        # Divide a pixel width by pixel ratio to get a real width
        self._sensor_area = 0
        self.update_mat_parameters(self._num_rows, self._num_columns, self._track_width_mm, self._gap_width_mm)

    def update_mat_parameters(self, rows, cols, track_width, spacing_width):
        self._clear_sensors()
        self._clear_loads()
        self._num_rows = rows
        self._num_columns = cols
        self._track_width_mm = track_width
        self._sensor_area = (track_width / 1000) ** 2  # in metres squared
        self._gap_width_mm = spacing_width
        self._spacing_ratio = spacing_width / track_width
        if self._num_rows > self._num_columns:
            num = self._num_rows
        else:
            num = self._num_columns
        self._track_width_pixel = (self.canvas_width - 1) / (self._spacing_ratio * num + num)
        self._spacing_width_pixel = self._spacing_ratio * self._track_width_pixel
        self._pixel_ratio = self._track_width_pixel / self._track_width_mm
        self._draw_sensors()
        self._draw_load(5, 70, 70, 'green')
        self._draw_load(5, 40, 40, 'blue')

    def _draw_sensors(self):
        x = self._spacing_width_pixel / 2
        y = self._spacing_width_pixel / 2

        for row in range(self._num_rows):
            for col in range(self._num_columns):
                sensor_reference = self.canvas.create_rectangle(x,
                                                                y,
                                                                x + self._track_width_pixel,
                                                                y + self._track_width_pixel,
                                                                fill='black', outline='', tags='pressure_sensor')
                current_sensor = Sensor(sensor_reference,
                                        sensor_area=self._sensor_area, pdr=10000, r0=0.91618, k=1.356e-5)
                self._sensors.append(current_sensor)
                x += self._track_width_pixel + self._spacing_width_pixel
            x = self._spacing_width_pixel / 2
            y += self._track_width_pixel + self._spacing_width_pixel

    def _draw_load(self, weight, width, height, colour):  # kg, mm, mm
        pixel_width = round(self._pixel_ratio * width)
        pixel_height = round(self._pixel_ratio * height)
        load_reference = self.canvas.create_rectangle(0, 0, pixel_width, pixel_height,
                                                      outline='', fill=colour, tags='load')
        load = Load(load_reference,
                    centre_x=width/2, centre_y=height/2,
                    mass=weight, area=width * height / 1000000)
        self.canvas.tag_bind(load_reference, '<Button-1>', self._on_drag_start)
        self.canvas.tag_bind(load_reference, '<B1-Motion>', self._on_drag_motion)
        self._loads.append(load)

    # Checks if the loads and sensors overlap. If they do, computes and returns the pressure reading.
    def _get_sensor_pressure(self, sensor, load, approximate):
        # Get the coordinates of the rectangles
        sensor_coordinates = self.canvas.coords(sensor.reference)
        load_coordinates = self.canvas.coords(load.reference)

        # Coordinates: [x1, y1, x2, y2]
        x1 = max(sensor_coordinates[0], load_coordinates[0])
        y1 = max(sensor_coordinates[1], load_coordinates[1])
        x2 = min(sensor_coordinates[2], load_coordinates[2])
        y2 = min(sensor_coordinates[3], load_coordinates[3])

        # Calculate overlap area
        overlap_width = max(0, x2 - x1)
        overlap_height = max(0, y2 - y1)
        pixel_overlap_area = overlap_width * overlap_height

        if pixel_overlap_area > 0:
            pressure = load.compute_pressure()
        else:
            pixel_overlap_area = 0
            pressure = 0
        real_overlap_area = pixel_overlap_area/((self._pixel_ratio * 1000) ** 2)
        adc_result = sensor.compute_adc_value(pressure, approximate=approximate)

        return adc_result

    def check_sensors(self, approximate=True):
        matrix_adc_results = np.zeros((self._num_rows, self._num_columns), dtype=np.int16)
        adc_results = np.zeros(len(self._loads), dtype=np.int16)
        no_load = self._sensors[0].compute_adc_value(0, approximate=approximate)
        scaling_factor = 4095/(4095 - no_load)
        if matrix_adc_results.size != len(self._sensors):
            exit("Mismatch between number of sensors and matrix size")
        sensor = 0
        for row in range(0, self._num_rows):
            for col in range(0, self._num_columns):
                for load_number in range(0, len(self._loads)):
                    adc_results[load_number] = self._get_sensor_pressure(self._sensors[sensor],
                                                                         self._loads[load_number],
                                                                         approximate=approximate)
                sensor += 1
                rescaled_adc_result = max(adc_results) - no_load
                if rescaled_adc_result < 0:
                    rescaled_adc_result = 0
                matrix_adc_results[row][col] = scaling_factor * rescaled_adc_result
        return matrix_adc_results

    def get_loads(self):
        return self._loads

    def get_pixel_ratio(self):
        return self._pixel_ratio

    # returns in mm
    def get_grid_spacing(self):
        return self._gap_width_mm, self._track_width_mm

    def _on_drag_start(self, event, *args):
        event.widget.start_x = event.x
        event.widget.start_y = event.y

    def _on_drag_motion(self, event, *args):
        delta_x = event.x - event.widget.start_x
        delta_y = event.y - event.widget.start_y
        reference = event.widget.find_withtag(tk.CURRENT)[0]
        self.canvas.move(reference, delta_x, delta_y)
        for load in self._loads:
            if load.reference == reference:
                previous_centre_x, previous_centre_y = load.get_location()
                new_centre_x = previous_centre_x + delta_x / self._pixel_ratio
                new_centre_y = previous_centre_y + delta_y / self._pixel_ratio
                load.update_location(new_centre_x, new_centre_y)
        event.widget.start_x = event.x
        event.widget.start_y = event.y

    def _clear_sensors(self):
        self.canvas.delete('pressure_sensor')  # Clear previous tracks
        self._sensors = []

    def _clear_loads(self):
        self.canvas.delete('load')
        self._loads = []


# Draws the right grid which displays the results of the simulation
class SimulationResult:
    def __init__(self, canvas, rows, columns):
        # Will be assuming that the width and height of the canvas are the same
        self.canvas = canvas
        self._num_rows = rows
        self._num_columns = columns
        self._canvas_width = canvas.winfo_reqwidth()
        self._canvas_height = canvas.winfo_reqheight()
        self._pixel_grid_spacing_width = self._get_grid_spacing()
        self._real_pressure_circle_x_position = self._canvas_width / 2
        self._real_pressure_circle_y_position = self._canvas_height / 2
        self._estimated_pressure_circle_x_position = self._canvas_width / 2
        self._estimated_pressure_circle_y_position = self._canvas_height / 2
        self._cell_width = self._canvas_width // columns
        self._cell_height = self._canvas_height // rows
        self.rectangles = []
        self.colour_map = create_colourmap()
        self.draw()
        self.pressure_circle = 0

    def draw(self):
        x = 0
        y = 0

        for row in range(self._num_rows):
            for col in range(self._num_columns):
                rectangle = self.canvas.create_rectangle(x,
                                                         y,
                                                         x + self._pixel_grid_spacing_width,
                                                         y + self._pixel_grid_spacing_width,
                                                         outline="#777777", tags='sensor_result')
                self.rectangles.append(rectangle)
                x += self._pixel_grid_spacing_width
            x = 0
            y += self._pixel_grid_spacing_width

        self.canvas.create_oval(self._canvas_width / 2 - 5, self._canvas_height / 2 - 5,
                                self._canvas_width / 2 + 5, self._canvas_height / 2 + 5,
                                fill='#E3242B', outline='', state='hidden', tag='real_pressure_circle')

        self.canvas.create_oval(self._canvas_width / 2 - 5, self._canvas_height / 2 - 5,
                                self._canvas_width / 2 + 5, self._canvas_height / 2 + 5,
                                fill='white', outline='', state='hidden', tag='estimated_pressure_circle')

    def edit_rectangle(self, row, col, color):
        index = row * self._num_columns + col
        if 0 <= index < len(self.rectangles):
            self.canvas.itemconfig(self.rectangles[index], fill=color)

    def match_colours(self, matrix_data):
        # Map each value in the matrix to a color
        if self._check_matrix_size(matrix_data):
            colour_matrix = [[self.colour_map[value] for value in row] for row in matrix_data]
            return colour_matrix
        else:
            return None

    def update_matrix_parameters(self, row, col):
        self._num_rows = row
        self._num_columns = col
        self._pixel_grid_spacing_width = self._get_grid_spacing()
        self.clear()
        self._cell_width = self._canvas_width // self._num_columns
        self._cell_height = self._canvas_height // self._num_rows
        self.draw()

    def update_matrix_colours(self, colour_matrix):
        if colour_matrix:
            for row in range(0, self._num_rows):
                for column in range(0, self._num_columns):
                    self.edit_rectangle(row, column, colour_matrix[row][column])

    def _check_matrix_size(self, matrix):
        if len(matrix) == self._num_rows:
            if len(matrix[-1]) == self._num_columns:
                return True
        print('Matrix data did not match with the expected size')
        return False

    def plot_estimated_centre_of_pressure(self, matrix_data, gap_spacing, track_spacing):
        # Create coordinate matrices for X and Y
        x, y = np.meshgrid(np.arange(matrix_data.shape[1]), np.arange(matrix_data.shape[0]))
        # Calculate total pressure and centroid coordinates
        total_pressure = np.sum(matrix_data)
        if total_pressure > 0:
            centre_x = np.sum(x * matrix_data) / total_pressure
            centre_y = np.sum(y * matrix_data) / total_pressure
            # print(centre_x, centre_y)
            # print("X: {}, Y: {}".format(centre_x, centre_y))
            new_centre_x = self._pixel_grid_spacing_width * centre_x + self._pixel_grid_spacing_width / 2
            new_centre_y = self._pixel_grid_spacing_width * centre_y + self._pixel_grid_spacing_width / 2
            centre_dx = new_centre_x - self._estimated_pressure_circle_x_position
            centre_dy = new_centre_y - self._estimated_pressure_circle_y_position
            self._estimated_pressure_circle_x_position = new_centre_x
            self._estimated_pressure_circle_y_position = new_centre_y
            self.canvas.move('estimated_pressure_circle', centre_dx, centre_dy)
            self.canvas.itemconfigure('estimated_pressure_circle', state='normal')
            total_spacing = gap_spacing + track_spacing
            centre_x_mm = (centre_x + 0.5) * total_spacing
            centre_y_mm = (centre_y + 0.5) * total_spacing
        else:
            self.canvas.itemconfigure('estimated_pressure_circle', state='hidden')
            centre_x_mm = -1.0
            centre_y_mm = -1.0
        return centre_x_mm, centre_y_mm

    # Computes the centre of pressure using moments of uniform loads
    def plot_real_centre_of_pressure(self, loads, pixel_ratio):
        if loads:
            centre_x_mm = 0
            centre_y_mm = 0
            total_mass = 0
            for load in loads:
                centre_x_mm += load.compute_x_moment()
                centre_y_mm += load.compute_y_moment()
                total_mass += load.get_mass()
            # Measurements in mm
            centre_x_mm /= total_mass
            centre_y_mm /= total_mass
            # Measurements in terms of pixels on the canvas
            new_centre_x_pixel = pixel_ratio * centre_x_mm
            new_centre_y_pixel = pixel_ratio * centre_y_mm
            centre_dx = new_centre_x_pixel - self._real_pressure_circle_x_position
            centre_dy = new_centre_y_pixel - self._real_pressure_circle_y_position
            self._real_pressure_circle_x_position = new_centre_x_pixel
            self._real_pressure_circle_y_position = new_centre_y_pixel
            self.canvas.move('real_pressure_circle', centre_dx, centre_dy)
            self.canvas.itemconfigure('real_pressure_circle', state='normal')
        else:
            self.canvas.itemconfigure('real_pressure_circle', state='hidden')
            centre_x_mm = -1.0
            centre_y_mm = -1.0
        return centre_x_mm, centre_y_mm

    def clear(self):
        self.canvas.delete('real_pressure_circle')
        self.canvas.delete('estimated_pressure_circle')
        self.canvas.delete('sensor_result')
        self.rectangles = []
        self._real_pressure_circle_x_position = self._canvas_width / 2
        self._real_pressure_circle_y_position = self._canvas_height / 2
        self._estimated_pressure_circle_x_position = self._canvas_width / 2
        self._estimated_pressure_circle_y_position = self._canvas_height / 2
        self.pressure_circle = 0

    def _get_grid_spacing(self):
        if self._num_rows > self._num_columns:
            num = self._num_rows
        else:
            num = self._num_columns
        grid_spacing = (self._canvas_width - 1) / num
        return grid_spacing


class App:
    def __init__(self, name):
        # All of this stuff sets up and organises the GUI
        self.timer = [0, 0]

        self.selected_device = None
        self.devices = []

        self.root = tk.Tk()
        self.root.config(background="#2b2b2b")
        self.root.title(name)
        self.root.resizable(False, False)
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.root.iconbitmap("icon.ico")
        self.root.protocol("WM_DELETE_WINDOW", self._exit)
        # Row 0
        # Canvas Labels
        self.simulation_header_label = create_widget(self.root, tk.Label, text="Simulation Setup")
        self.simulation_header_label.grid(row=0, column=0, columnspan=1)
        self.results_header_label = create_widget(self.root, tk.Label, text="Results")
        self.results_header_label.grid(row=0, column=1, columnspan=1)

        # Row 1
        # Canvas simulation input grid
        self.simulation_canvas = create_widget(self.root, tk.Canvas, width=GRID_WIDTH, height=GRID_HEIGHT,
                                               borderwidth=0)
        self.simulation_canvas.grid(row=1, column=0, columnspan=1)
        self.setup_grid = SimulationSetup(self.simulation_canvas, rows=GRID_SIZE, columns=GRID_SIZE)

        # Canvas result grid
        self.result_canvas = create_widget(self.root, tk.Canvas, width=GRID_WIDTH, height=GRID_HEIGHT, borderwidth=0)
        self.result_canvas.grid(row=1, column=1, columnspan=1)

        self.results_grid = SimulationResult(self.result_canvas, rows=GRID_SIZE, columns=GRID_SIZE)
        self.results_grid.draw()

        # Row 2
        # Scale of heat colours
        self.heat_canvas = create_widget(self.root, tk.Canvas, width=GRID_WIDTH, height=25)
        self.heat_canvas.grid(row=2, column=1, columnspan=1, pady=2, sticky=tk.N)
        self._create_heatmap_scale(GRID_WIDTH, 25, self.results_grid.colour_map)

        # Row 3
        # Frame for Pressure Mat Design Configuration
        frame_configurations = create_widget(self.root, tk.Frame)
        frame_configurations.grid(row=3, column=0, rowspan=2, columnspan=1, sticky="ew")
        frame_pressure_mat = create_widget(frame_configurations, tk.Frame)
        frame_pressure_mat.grid(row=0, column=0)

        create_widget(frame_pressure_mat, tk.Label, text="Pressure Mat Design").grid(row=0, columnspan=2, pady=5)

        create_widget(frame_pressure_mat, tk.Label, text="Rows:").grid(row=1, column=0, sticky=tk.W)
        self.entry_row_number = create_widget(frame_pressure_mat, tk.Entry, width=4)
        self.entry_row_number.insert(0, GRID_SIZE)
        self.entry_row_number.grid(row=1, column=1, sticky=tk.W)

        create_widget(frame_pressure_mat, tk.Label, text="Columns:").grid(row=2, column=0, sticky=tk.W)
        self.entry_col_number = create_widget(frame_pressure_mat, tk.Entry, width=4)
        self.entry_col_number.insert(0, GRID_SIZE)
        self.entry_col_number.grid(row=2, column=1, sticky=tk.W)

        create_widget(frame_pressure_mat, tk.Label, text="Track Width (mm):").grid(row=3, column=0, sticky=tk.W)
        self.entry_track_width = create_widget(frame_pressure_mat, tk.Entry, width=4)
        self.entry_track_width.insert(0, TRACK_WIDTH)
        self.entry_track_width.grid(row=3, column=1, sticky=tk.W)

        create_widget(frame_pressure_mat, tk.Label, text="Spacing (mm):").grid(row=4, column=0, sticky=tk.W)
        self.entry_spacing = create_widget(frame_pressure_mat, tk.Entry, width=4)
        self.entry_spacing.insert(0, PITCH_WIDTH)
        self.entry_spacing.grid(row=4, column=1, sticky=tk.W)

        self.update_mat_button = create_widget(frame_pressure_mat, tk.Button, text="Update Mat Design",
                                               command=self._update_mat).grid(row=5, columnspan=2, pady=5)

        # Frame for Pressure Mat settings
        frame_load_config = create_widget(frame_configurations, tk.Frame)
        frame_load_config.grid(row=0, column=1)

        create_widget(frame_load_config, tk.Label, text="Load Configuration", pady=10).grid(row=0, columnspan=2)
        '''
        create_widget(frame_load_config, tk.Label, text="Rows:").grid(row=1, column=0, sticky=tk.W)
        self.entry_resolution = tk.Entry(frame_pressure_mat, width=4)
        self.entry_resolution.grid(row=1, column=1, sticky=tk.W)

        create_widget(frame_load_config, tk.Label, text="Columns:").grid(row=2, column=0, sticky=tk.W)
        self.entry_resolution = tk.Entry(frame_pressure_mat, width=4)
        self.entry_resolution.grid(row=2, column=1, sticky=tk.W)

        create_widget(frame_load_config, tk.Label, text="Track Width (mm):").grid(row=3, column=0, sticky=tk.W)
        self.entry_track_width = tk.Entry(frame_pressure_mat, width=4)
        self.entry_track_width.grid(row=3, column=1, sticky=tk.W)

        create_widget(frame_load_config, tk.Label, text="Spacing (mm):").grid(row=4, column=0, sticky=tk.W)
        self.entry_spacing = tk.Entry(frame_pressure_mat, width=4)
        self.entry_spacing.grid(row=4, column=1, sticky=tk.W)
        '''
        self.update_load_button = create_widget(frame_load_config, tk.Button, text="Update Load",
                                                command=self._update_load).grid(row=5, columnspan=2, pady=5)

        # Centre of Pressure Readouts
        cop_readouts = create_widget(self.root, tk.Frame)
        cop_readouts.grid(row=3, column=1, columnspan=1, sticky="new")

        create_widget(cop_readouts, tk.Label, text="Centre of Pressure").grid(row=0, columnspan=2, pady=5, sticky="we")
        create_widget(cop_readouts, tk.Label, text="Real: ").grid(row=1, column=0, sticky="w")
        self.real_CoP_label = create_widget(cop_readouts, tk.Label, text="0", foreground="#E3242B")
        self.real_CoP_label.grid(row=1, column=1, sticky="w")
        create_widget(cop_readouts, tk.Label, text="Estimated: ").grid(row=2, column=0, sticky="w")
        self.estimated_CoP_label = create_widget(cop_readouts, tk.Label, text="0", foreground="white")
        self.estimated_CoP_label.grid(row=2, column=1, sticky="w")
        create_widget(cop_readouts, tk.Label, text="Error:").grid(row=3, column=0, sticky="w")
        self.error_CoP_label = create_widget(cop_readouts, tk.Label, text="0")
        self.error_CoP_label.grid(row=3, column=1, sticky="w")

        # Row 5
        # Buttons
        # self.connect_button = create_widget(self.root, tk.Button, text="Start", command=self.start_simulation)
        # self.connect_button.grid(row=5, column=0)

        # self.search_button = create_widget(self.root, tk.Button, text="Stop", command=self.stop_simulation)
        # self.search_button.grid(row=5, column=1)

        # Test Scenarios

        self.root.columnconfigure((0, 1), weight=1)

        # Starts the simulation loop
        self.root.after(20, self._simulate)

    def _update_mat(self):
        row_number = self.entry_row_number.get()
        col_number = self.entry_col_number.get()
        track_width = self.entry_track_width.get()
        spacing_width = self.entry_spacing.get()
        if row_number.isdigit() and col_number.isdigit() and track_width.isdigit() and spacing_width.isdigit():
            self.setup_grid.update_mat_parameters(rows=int(row_number), cols=int(col_number),
                                                  track_width=int(track_width), spacing_width=int(spacing_width))
            self.results_grid.update_matrix_parameters(row=int(row_number), col=int(col_number))

    def _update_load(self):
        print("TODO")

    # Recursive function that acts as an infinite loop, constantly checking and updating the grids.
    def _simulate(self):
        matrix = self.setup_grid.check_sensors(approximate=APPROXIMATE_FORCE)
        self._update_heatmap(matrix)
        self.root.after(20, self._simulate)

    def _exit(self):
        self.root.destroy()

    def _create_heatmap_scale(self, width, height, colour_map):
        for x in range(width):
            increment = 4095 * x / width
            colour = colour_map[round(increment)]
            self.heat_canvas.create_line(x, 0, x, height, fill=colour, width=1)

    def _update_heatmap(self, matrix_data):
        matrix_colours = self.results_grid.match_colours(matrix_data)
        self.results_grid.update_matrix_colours(matrix_colours)
        real_x, real_y = self.results_grid.plot_real_centre_of_pressure(self.setup_grid.get_loads(),
                                                                        self.setup_grid.get_pixel_ratio())
        gap_spacing, track_spacing = self.setup_grid.get_grid_spacing()
        estimated_x, estimated_y = self.results_grid.plot_estimated_centre_of_pressure(matrix_data,
                                                                                       gap_spacing, track_spacing)
        error_x = 100 * (estimated_x - real_x) / real_x
        error_y = 100 * (estimated_y - real_y) / real_y
        self.real_CoP_label.config(text="x={:7.2f}mm, y={:7.2f}mm".format(real_x, real_y))
        self.estimated_CoP_label.config(text="x={:7.2f}mm, y={:7.2f}mm".format(estimated_x, estimated_y))
        self.error_CoP_label.config(text="x={:7.2f}%, y={:7.2f}%".format(error_x, error_y))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    program = App("BLE Pressure Mat")
    program.run()