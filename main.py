import tkinter as tk
import numpy as np
import random
from scipy import constants
from tkinter import ttk

'''
TODO:
fix slight error in real cop measurement

make objects configurable

prevent overlaps and moving objects out of the border - also reset load positions if they leave the screen
pause program when moving window to get rid of lag
check for memory leaks
'''

GRID_SIZE = 16
GRID_LENGTH = 500
GRID_WIDTH = 500
TRACK_WIDTH = 15
PITCH_WIDTH = 15

R0 = 9.1618  # 0.91618
K = 1.356e-5

APPROXIMATE_FORCE = True

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
        self._reference = reference
        self._centre_x = centre_x
        self._centre_y = centre_y
        self._mass = mass
        self._area = area  # area in m^2

    def update_location(self, centre_x, centre_y):
        self._centre_x = centre_x
        self._centre_y = centre_y

    def update_mass(self, mass):
        self._mass = mass

    def update_area(self, area):
        self._area = area

    def get_reference(self):
        return self._reference

    def get_mass(self):
        return self._mass

    def get_location(self):
        return self._centre_x, self._centre_y

    def compute_pressure(self):
        return constants.g*self._mass/self._area

    def compute_x_moment(self):
        return self._mass * self._centre_x

    def compute_y_moment(self):
        return self._mass * self._centre_y


class Sensor:
    def __init__(self, reference, sensor_area, r0, k):
        self._reference = reference
        self._sensor_area = sensor_area
        self._r0 = r0
        self._pdr = self._r0 / self._sensor_area
        self._k = k

    def get_reference(self):
        return self._reference

    def update_sensor(self, sensor_area, pdr, r0, k):
        self._sensor_area = sensor_area
        self._pdr = pdr
        self._r0 = r0
        self._k = k

    def compute_resistance(self, pressure, loaded_area):
        if 0 < loaded_area < self._sensor_area:
            resistance_loaded = (self._r0 / loaded_area) * np.exp(-self._k * pressure)
            resistance_unloaded = (self._r0 / (self._sensor_area - loaded_area))
            resistance = 1/(1/resistance_loaded + 1/resistance_unloaded)
        else:
            resistance = (self._r0 / self._sensor_area) * np.exp(-self._k * pressure)
        return resistance

    # Through the potential divider the voltage has a somewhat linear relationship with force up to about 100kN
    # This means these ADC values are an approximation for force and are not always accurate. This is the default.
    # If approximate=False, then the correct force values will be mapped to the 12bit ADC output.
    def compute_adc_value(self, pressure, loaded_area=None, approximate=True):
        z1 = self.compute_resistance(pressure, loaded_area)
        if approximate:  # uses the potential divider to estimate the force rather than the equation.
            z2 = self._pdr
            voltage = z2 / (z1 + z2)
            adc_value = round(4095 * voltage)
        else:  # returns the real value of the force using the equation.
            force = np.log(z1/(self._r0 / self._sensor_area))/-self._k
            if force > 200000:
                force = 200000
            adc_value = round(4095 * force / 200000)
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
        self._track_width_mm = TRACK_WIDTH  # in mm
        self._pitch_width_mm = PITCH_WIDTH
        self._mat_width = self._num_columns * (self._track_width_mm + self._pitch_width_mm)
        self._mat_length = self._num_rows * (self._track_width_mm + self._pitch_width_mm)
        self._spacing_ratio = 0
        self._track_width_pixel = 0
        self._spacing_width_pixel = 0
        self._pixel_ratio = 0
        # Multiply real width by pixel ratio to get a pixel width.
        # Divide a pixel width by pixel ratio to get a real width
        self._sensor_area = 0
        self.update_mat_parameters(self._num_rows, self._num_columns, self._track_width_mm, self._pitch_width_mm)

    def update_mat_parameters(self, rows, cols, track_width, spacing_width):
        self._clear_sensors()
        self.clear_loads()
        self._num_rows = rows
        self._num_columns = cols
        self._track_width_mm = track_width
        self._sensor_area = (track_width / 1000) ** 2  # in metres squared
        self._pitch_width_mm = spacing_width
        self._spacing_ratio = spacing_width / track_width
        self._mat_width = self._num_columns * (self._track_width_mm + self._pitch_width_mm)
        self._mat_length = self._num_rows * (self._track_width_mm + self._pitch_width_mm)
        if self._num_rows > self._num_columns:
            num = self._num_rows
        else:
            num = self._num_columns
        self._track_width_pixel = (self.canvas_width - 1) / (self._spacing_ratio * num + num)
        self._spacing_width_pixel = self._spacing_ratio * self._track_width_pixel
        self._pixel_ratio = self._track_width_pixel / self._track_width_mm
        self._draw_sensors()
        # temporary dummy loads until a setting that allows users to add loads is created
        self.draw_load(20, 70, 70, 'green')
        self.draw_load(20, 40, 40, 'blue')

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
                                        sensor_area=self._sensor_area, r0=R0, k=K)
                self._sensors.append(current_sensor)
                x += self._track_width_pixel + self._spacing_width_pixel
            x = self._spacing_width_pixel / 2
            y += self._track_width_pixel + self._spacing_width_pixel

    def draw_load(self, weight, width, height, colour, x=0, y=0, drag=True):  # kg, mm, mm, canvas x pos, canvas y pos
        pixel_width = round(self._pixel_ratio * width)
        pixel_height = round(self._pixel_ratio * height)
        real_x_position = x / self._pixel_ratio
        real_y_position = y / self._pixel_ratio
        load_reference = self.canvas.create_rectangle(x, y, x + pixel_width, y + pixel_height,
                                                      outline='', fill=colour, tags='load')

        # update this to fix real cop
        load = Load(load_reference,
                    centre_x=real_x_position + width/2, centre_y=real_y_position + height/2,
                    mass=weight, area=width * height / 1000000)
        if drag is True:
            self.canvas.tag_bind(load_reference, '<Button-1>', self._on_drag_start)
            self.canvas.tag_bind(load_reference, '<B1-Motion>', self._on_drag_motion)
        self._loads.append(load)

    # Checks if the loads and sensors overlap. If they do, computes and returns the pressure reading.
    def _get_sensor_pressure(self, sensor, load, approximate):
        # Get the coordinates of the rectangles
        sensor_coordinates = self.canvas.coords(sensor.get_reference())
        load_coordinates = self.canvas.coords(load.get_reference())

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
        adc_result = sensor.compute_adc_value(pressure, real_overlap_area, approximate=approximate)

        return adc_result

    def check_sensors(self, approximate=True):
        matrix_adc_results = np.zeros((self._num_rows, self._num_columns), dtype=np.int16)
        adc_results = np.zeros(len(self._loads), dtype=np.int16)
        no_load = self._sensors[0].compute_adc_value(0, loaded_area=0, approximate=approximate)
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
                if adc_results.any():
                    rescaled_adc_result = max(adc_results) - no_load
                else:
                    rescaled_adc_result = 0
                if rescaled_adc_result < 0:
                    rescaled_adc_result = 0
                matrix_adc_results[row][col] = scaling_factor * rescaled_adc_result
        return matrix_adc_results

    def get_loads(self):
        return self._loads

    def get_mat_dimensions(self):
        return self._mat_width, self._mat_length

    def update_load_mass(self, index, mass):
        self._loads[index].update_mass(mass)

    def update_load_location(self, index, x, y):  # x and y are the absolute centre coordinates in real distance (mm)
        start_x, start_y = self._loads[index].get_location()
        delta_x_pixels = round((x - start_x) * self._pixel_ratio)
        delta_y_pixels = round((y - start_y) * self._pixel_ratio)
        self._loads[index].update_location(x, y)
        self.canvas.move(self._loads[index].get_reference(), delta_x_pixels, delta_y_pixels)

    def get_load_size(self, index):
        x1, y1, x2, y2 = self.canvas.coords(self._loads[index].get_reference())
        x1 /= self._pixel_ratio
        y1 /= self._pixel_ratio
        x2 /= self._pixel_ratio
        y2 /= self._pixel_ratio
        return x1, y1, x2, y2

    def update_load_size(self, index, x1, y1, x2, y2):  # top left corner, bottom right corner, size is real distance mm
        x1_pixels = round(x1 * self._pixel_ratio)
        y1_pixels = round(y1 * self._pixel_ratio)
        x2_pixels = round(x2 * self._pixel_ratio)
        y2_pixels = round(y2 * self._pixel_ratio)
        centre_x = x1 + ((x2 - x1) / 2)
        centre_y = y1 + ((y2 - y1) / 2)
        self._loads[index].update_location(centre_x, centre_y)
        self.canvas.coords(self._loads[index].get_reference(), x1_pixels, y1_pixels, x2_pixels, y2_pixels)

    def update_load_area(self, index, area):
        self._loads[index].update_area(area)

    def get_pixel_ratio(self):
        return self._pixel_ratio

    # returns in mm
    def get_grid_spacing(self):
        return self._pitch_width_mm, self._track_width_mm

    def _on_drag_start(self, event, *args):
        event.widget.start_x = event.x
        event.widget.start_y = event.y

    def _on_drag_motion(self, event, *args):
        delta_x = event.x - event.widget.start_x
        delta_y = event.y - event.widget.start_y
        reference = event.widget.find_withtag(tk.CURRENT)[0]
        self.canvas.move(reference, delta_x, delta_y)
        for load in self._loads:
            if load.get_reference() == reference:
                previous_centre_x, previous_centre_y = load.get_location()
                new_centre_x = previous_centre_x + delta_x / self._pixel_ratio
                new_centre_y = previous_centre_y + delta_y / self._pixel_ratio
                load.update_location(new_centre_x, new_centre_y)
        event.widget.start_x = event.x
        event.widget.start_y = event.y

    def _clear_sensors(self):
        self.canvas.delete('pressure_sensor')  # Clear previous tracks
        self._sensors = []

    def clear_loads(self):
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

        self.foot_length = 250  # mm
        self.foot_width = 105  # mm
        self._scenario_running = False
        self._scenario_update_task = None
        self._left_centre_x = None
        self._left_centre_y = None
        self._right_centre_x = None
        self._right_centre_y = None
        self._error_x = 0
        self._error_y = 0
        self._average_error_x = None
        self._average_error_y = None
        self._average_error_x_sum = 0
        self._average_error_y_sum = 0
        self._average_error_count = 0

        # Row 0
        # Canvas Labels
        self.simulation_header_label = create_widget(self.root, tk.Label, text="Simulation Setup")
        self.simulation_header_label.grid(row=0, column=0, columnspan=1)
        self.results_header_label = create_widget(self.root, tk.Label, text="Results")
        self.results_header_label.grid(row=0, column=1, columnspan=1)

        # Row 1
        # Canvas simulation input grid
        self.simulation_canvas = create_widget(self.root, tk.Canvas, width=GRID_WIDTH, height=GRID_LENGTH,
                                               borderwidth=0)
        self.simulation_canvas.grid(row=1, column=0, columnspan=1)
        self.setup_grid = SimulationSetup(self.simulation_canvas, rows=GRID_SIZE, columns=GRID_SIZE)

        # Canvas result grid
        self.result_canvas = create_widget(self.root, tk.Canvas, width=GRID_WIDTH, height=GRID_LENGTH, borderwidth=0)
        self.result_canvas.grid(row=1, column=1, columnspan=1)

        self.results_grid = SimulationResult(self.result_canvas, rows=GRID_SIZE, columns=GRID_SIZE)
        self.results_grid.draw()

        # Row 2
        # Scale of heat colours
        self.heat_canvas = create_widget(self.root, tk.Canvas, width=GRID_WIDTH, height=25)
        self.heat_canvas.grid(row=2, column=1, columnspan=1, pady=2, sticky=tk.N)
        self._create_heatmap_scale(GRID_WIDTH, 25, self.results_grid.colour_map)

        # Frame for Pressure Mat Design Configuration
        frame_configurations = create_widget(self.root, tk.Frame)
        frame_configurations.grid(row=2, column=0, rowspan=2, columnspan=1, sticky=tk.NSEW)
        frame_configurations.grid_columnconfigure((0, 1, 2), weight=1)
        frame_configurations.grid_rowconfigure(0, weight=1)

        frame_pressure_mat = create_widget(frame_configurations, tk.Frame)
        frame_pressure_mat.grid(row=0, column=0, sticky=tk.NSEW)
        frame_pressure_mat.grid_columnconfigure(0, weight=1)

        create_widget(frame_pressure_mat, tk.Label, text="Pressure Mat Design").grid(row=0, columnspan=2, pady=1,
                                                                                     sticky=tk.N)
        create_widget(frame_pressure_mat, tk.Label, text="Rows:").grid(row=1, column=0, sticky=tk.W)
        self.entry_row_number = create_widget(frame_pressure_mat, tk.Entry, width=3)
        self.entry_row_number.insert(0, GRID_SIZE)
        self.entry_row_number.grid(row=1, column=1, sticky=tk.W, padx=(0, 3))

        create_widget(frame_pressure_mat, tk.Label, text="Columns:").grid(row=2, column=0, sticky=tk.W)
        self.entry_col_number = create_widget(frame_pressure_mat, tk.Entry, width=3)
        self.entry_col_number.insert(0, GRID_SIZE)
        self.entry_col_number.grid(row=2, column=1, sticky=tk.W, padx=(0, 3))

        create_widget(frame_pressure_mat, tk.Label, text="Track Width (mm):").grid(row=3, column=0, sticky=tk.W)
        self.entry_track_width = create_widget(frame_pressure_mat, tk.Entry, width=3)
        self.entry_track_width.insert(0, TRACK_WIDTH)
        self.entry_track_width.grid(row=3, column=1, sticky=tk.W, padx=(0, 3))

        create_widget(frame_pressure_mat, tk.Label, text="Spacing (mm):").grid(row=4, column=0, sticky=tk.W)
        self.entry_spacing = create_widget(frame_pressure_mat, tk.Entry, width=3)
        self.entry_spacing.insert(0, PITCH_WIDTH)
        self.entry_spacing.grid(row=4, column=1, sticky=tk.W, padx=(0, 3))

        self.update_mat_button = create_widget(frame_pressure_mat, tk.Button, text="Update Mat Design",
                                               command=self._update_mat).grid(row=5, columnspan=2, pady=1)

        # Frame for Pressure Mat load settings
        frame_load_config = create_widget(frame_configurations, tk.Frame)
        frame_load_config.grid(row=0, column=1, sticky=tk.NSEW)
        frame_load_config.grid_columnconfigure(0, weight=1)

        create_widget(frame_load_config, tk.Label, text="Load Configuration").grid(row=0, columnspan=2,
                                                                                   sticky=tk.N, pady=1)
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
                                                command=self._update_load).grid(row=5, columnspan=2, pady=1)
        # Frame for test scenarios
        frame_test_scenarios = create_widget(frame_configurations, tk.Frame)
        frame_test_scenarios.grid(row=0, column=2, sticky=tk.NSEW)
        frame_test_scenarios.grid_columnconfigure(0, weight=1)

        create_widget(frame_test_scenarios, tk.Label, text="Test Scenarios").grid(row=0, columnspan=2,
                                                                                  sticky=tk.N, pady=1)
        self.side_weight_shift_button = create_widget(frame_test_scenarios, tk.Button, text="Side Weight Shift",
                                                      command=lambda: self.start_scenario(
                                                          self._side_weight_shift_scenario))\
            .grid(row=1, columnspan=2, pady=1)
        self.front_weight_shift_button = create_widget(frame_test_scenarios, tk.Button, text="Front Weight Shift",
                                                       command=lambda: self.start_scenario(
                                                           self._front_weight_shift_scenario))\
            .grid(row=2, columnspan=2, pady=1)
        self.foot_slide_button = create_widget(frame_test_scenarios, tk.Button, text="Foot Sliding",
                                               command=lambda: self.start_scenario(
                                                   self._foot_slide_shift_scenario))\
            .grid(row=3, columnspan=2, pady=1)
        self.foot_slide_button = create_widget(frame_test_scenarios, tk.Button, text="Random Placements",
                                               command=lambda: self.start_scenario(
                                                   self._random_foot_placement_scenario))\
            .grid(row=4, columnspan=2, pady=1)
        self.stop_scenario_button = create_widget(frame_test_scenarios, tk.Button, text="Stop Scenario",
                                                  command=self.stop_scenario).grid(row=5, columnspan=2, pady=1)

        # Row 3
        # Centre of Pressure Readouts
        cop_readouts = create_widget(self.root, tk.Frame)
        cop_readouts.grid(row=3, column=1, columnspan=1, sticky="new")
        cop_readouts.grid_columnconfigure(0, weight=1)

        create_widget(cop_readouts, tk.Label, text="Centre of Pressure").grid(row=0, columnspan=2, pady=1, sticky="nwe")
        create_widget(cop_readouts, tk.Label, text="Real: ").grid(row=1, column=0, sticky="w")
        self.real_CoP_label = create_widget(cop_readouts, tk.Label, text="0", foreground="#E3242B")
        self.real_CoP_label.grid(row=1, column=1, sticky="w")
        create_widget(cop_readouts, tk.Label, text="Estimated: ").grid(row=2, column=0, sticky="w")
        self.estimated_CoP_label = create_widget(cop_readouts, tk.Label, text="0", foreground="white")
        self.estimated_CoP_label.grid(row=2, column=1, sticky="w")
        create_widget(cop_readouts, tk.Label, text="Error:").grid(row=3, column=0, sticky="w")
        self.error_CoP_label = create_widget(cop_readouts, tk.Label, text="0")
        self.error_CoP_label.grid(row=3, column=1, sticky="w")
        create_widget(cop_readouts, tk.Label, text="Average Error:").grid(row=4, column=0, sticky="w")
        self.average_error_CoP_label = create_widget(cop_readouts, tk.Label, text="0")
        self.average_error_CoP_label.grid(row=4, column=1, sticky="w")

        # Row 5
        # Buttons
        # self.connect_button = create_widget(self.root, tk.Button, text="Start", command=self.start_simulation)
        # self.connect_button.grid(row=5, column=0)

        # self.search_button = create_widget(self.root, tk.Button, text="Stop", command=self.stop_simulation)
        # self.search_button.grid(row=5, column=1)

        # Test Scenarios

        self.root.columnconfigure((0, 1), weight=1)
        self.root.rowconfigure((0, 1, 3), weight=1)

        # Starts the simulation loop
        self.root.after(20, self._simulate)

    def _update_mat(self):
        self.stop_scenario()
        row_number = self.entry_row_number.get()
        col_number = self.entry_col_number.get()
        track_width = self.entry_track_width.get()
        pitch_width = self.entry_spacing.get()
        if row_number.isdigit() and col_number.isdigit() and track_width.replace(".", "").isdigit() \
                and pitch_width.replace(".", "").isdigit():
            self.setup_grid.update_mat_parameters(rows=int(row_number), cols=int(col_number),
                                                  track_width=float(track_width), spacing_width=float(pitch_width))
            self.results_grid.update_matrix_parameters(row=int(row_number), col=int(col_number))

    def start_scenario(self, scenario_function):
        if self._scenario_running:
            self.stop_scenario()
        random.seed(0)
        np.random.seed(0)
        self._scenario_running = True
        self.setup_grid.clear_loads()
        canvas_width = self.setup_grid.canvas_width  # pixels
        canvas_height = self.setup_grid.canvas_height  # pixels
        pixel_ratio = self.setup_grid.get_pixel_ratio()
        foot_length_pixels = self.foot_length * pixel_ratio
        foot_width_pixels = self.foot_width * pixel_ratio
        y1 = round((canvas_height - foot_length_pixels)/2)
        x1 = y1
        y2 = y1
        x2 = canvas_width - foot_width_pixels - x1
        self.setup_grid.draw_load(weight=40, width=self.foot_width, height=self.foot_length, colour="yellow",
                                  x=x1, y=y1, drag=False)
        self.setup_grid.draw_load(weight=40, width=self.foot_width, height=self.foot_length, colour="yellow",
                                  x=x2, y=y2, drag=False)
        loads = self.setup_grid.get_loads()
        self._left_centre_x, self._left_centre_y = loads[0].get_location()
        self._right_centre_x, self._right_centre_y = loads[1].get_location()
        self.root.after(100, scenario_function, 0)

    def _side_weight_shift_scenario(self, time):
        left_foot_weight = 70 * np.square((np.sin(2 * np.pi * time / 10000)))
        right_foot_weight = 70 * np.square((np.cos(2 * np.pi * time / 10000)))
        self.setup_grid.update_load_mass(0, left_foot_weight)
        self.setup_grid.update_load_mass(1, right_foot_weight)
        self._calculate_average_error(time)
        time += 100
        if self._scenario_running:
            self._scenario_update_task = self.root.after(100, self._side_weight_shift_scenario, time)

    def _front_weight_shift_scenario(self, time):
        left_foot_x1, foot_y1, left_foot_x2, foot_y2 = self.setup_grid.get_load_size(0)
        right_foot_x1, right_foot_y1, right_foot_x2, right_foot_y2 = self.setup_grid.get_load_size(1)
        y1_max = self._left_centre_y - self.foot_length / 2
        y2_max = self.foot_length / 2 + self._left_centre_y

        sine_multiplier = np.sin(2 * np.pi * time / 10000)
        if sine_multiplier >= 0:
            foot_y2 = (y2_max - y1_max) * (1 - (2 * sine_multiplier / 3)) + y1_max
        else:
            foot_y1 = y2_max - (y2_max - y1_max) * (1 + (2 * sine_multiplier / 3))

        self.setup_grid.update_load_size(0, left_foot_x1, foot_y1, left_foot_x2, foot_y2)
        self.setup_grid.update_load_size(1, right_foot_x1, foot_y1, right_foot_x2, foot_y2)
        area = (left_foot_x2 - left_foot_x1) * (foot_y2 - foot_y1) / 1000000
        self.setup_grid.update_load_area(0, area)
        self.setup_grid.update_load_area(1, area)
        self._calculate_average_error(time)

        time += 100
        if self._scenario_running:
            self._scenario_update_task = self.root.after(100, self._front_weight_shift_scenario, time)

    def _foot_slide_shift_scenario(self, time):
        left_foot_multiplier = np.sin(2 * np.pi * time / 5000) if np.sin(2 * np.pi * time / 5000) >= 0 else 0
        right_foot_multiplier = -np.sin(2 * np.pi * time / 5000) if -np.sin(2 * np.pi * time / 5000) >= 0 else 0
        left_foot_x = (self._left_centre_x - (self.foot_width / self.setup_grid.get_pixel_ratio())
                       * left_foot_multiplier)
        right_foot_x = (self._right_centre_x + (self.foot_width / self.setup_grid.get_pixel_ratio())
                        * right_foot_multiplier)
        self.setup_grid.update_load_location(0, left_foot_x, self._left_centre_y)
        self.setup_grid.update_load_location(1, right_foot_x, self._right_centre_y)
        self._calculate_average_error(time)
        time += 100
        if self._scenario_running:
            self._scenario_update_task = self.root.after(100, self._foot_slide_shift_scenario, time)

    def _random_foot_placement_scenario(self, time):
        def generate_positions():
            mat_width, mat_length = self.setup_grid.get_mat_dimensions()
            x1 = round(random.randint(0, mat_width - self.foot_width) + self.foot_width / 2)
            y1 = round(random.randint(0, mat_length - self.foot_length) + self.foot_length / 2)
            x2 = round(random.randint(0, mat_width - self.foot_width) + self.foot_width / 2)
            y2 = round(random.randint(0, mat_length - self.foot_length) + self.foot_length / 2)
            return x1, y1, x2, y2

        def update_locations(x1, y1, x2, y2):
            self.setup_grid.update_load_location(0, x1, y1)
            self.setup_grid.update_load_location(1, x2, y2)

        def generate_and_update_masses():
            m1 = np.random.normal(35, 10)
            m1 = 0 if m1 < 0 else m1
            m1 = 70 if m1 > 70 else m1
            m2 = 70 - m1
            self.setup_grid.update_load_mass(0, m1)
            self.setup_grid.update_load_mass(0, m2)
        def check_overlap(x1, y1, x2, y2):
            return abs(x2 - x1) < self.foot_width and abs(y2 - y1) < self.foot_length

        if time % 30000 == 0:
            random.seed(0)
            np.random.seed(0)
            self.setup_grid.update_load_mass(0, 35)
            self.setup_grid.update_load_mass(1, 35)
            self.setup_grid.update_load_location(0, self._left_centre_x, self._left_centre_y)
            self.setup_grid.update_load_location(1, self._right_centre_x, self._right_centre_y)
        elif time % 600 == 0:
            while True:
                x1, y1, x2, y2 = generate_positions()
                if not check_overlap(x1, y1, x2, y2):
                    break
            update_locations(x1, y1, x2, y2)
            generate_and_update_masses()

        self._calculate_average_error(time / 6)
        time += 100
        if self._scenario_running:
            self._scenario_update_task = self.root.after(100, self._random_foot_placement_scenario, time)

    def _calculate_average_error(self, time):
        if self._average_error_y is None or self._average_error_x is None:
            self._average_error_x_sum += abs(self._error_x)
            self._average_error_y_sum += abs(self._error_y)
            self._average_error_count += 1
            if time >= 5000:
                self._average_error_x = self._average_error_x_sum / self._average_error_count
                self._average_error_y = self._average_error_y_sum / self._average_error_count

    def stop_scenario(self):
        if self._scenario_running:
            self._scenario_running = False
            self._average_error_y = None
            self._average_error_x = None
            self._average_error_x_sum = 0
            self._average_error_y_sum = 0
            self._average_error_count = 0
            if self._scenario_update_task is not None:
                self.root.after_cancel(self._scenario_update_task)
                self._scenario_update_task = None

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
        self._error_x = 100 * (estimated_x - real_x) / real_x
        self._error_y = 100 * (estimated_y - real_y) / real_y
        self.real_CoP_label.config(text="x={:7.2f}mm, y={:7.2f}mm".format(real_x, real_y))
        self.estimated_CoP_label.config(text="x={:7.2f}mm, y={:7.2f}mm".format(estimated_x, estimated_y))
        self.error_CoP_label.config(text="x={:7.2f}%,     y={:7.2f}%".format(self._error_x, self._error_y))
        self.average_error_CoP_label.config(text="x={:7.2f}%,     y={:7.2f}%"
                                            .format(self._average_error_x if self._average_error_x is not None else 0,
                                                    self._average_error_y if self._average_error_y is not None else 0))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    program = App("BLE Pressure Mat")
    program.run()
