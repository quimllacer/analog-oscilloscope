import numpy as np
import time

import config
from file_processing import process_files, load_data, interpolate_path

def time(frequency, sample_rate):
    return np.linspace(0, 1/frequency, sample_rate, False).repeat(3).reshape(-1, 3).astype(np.float32)

def update():
    fp = '/Users/joaquinllacerwintle/Google Drive/mis proyectos/oscilloscope_render/files_to_render'
    process_files(fp)
    pass

# update()

def saw_wave(t, frequency):
    waveform = (t * frequency) % 1.0
    return waveform

def roses(array, k):
    out = np.column_stack((np.sin(k*2*np.pi*array[:,0])*np.sin(2*np.pi*array[:,0]),
                           np.sin(k*2*np.pi*array[:,1])*np.cos(2*np.pi*array[:,1]),
                           array[:,2]*0))
    return out

def mic_input(array, audio_input):
    audio_input = np.frombuffer(audio_input, dtype=np.float32)[:, np.newaxis].repeat(3).reshape(-1, 3)
    array = (audio_input*array+array)/2
    return array

def waves(array, n, wf, rate):

    wave = (0.5*(n/(2*np.pi))*(np.sin(n-np.pi/2).clip(0, 1))**2*np.sin(np.linspace(0,  2*np.pi*wf*(np.sin(n).clip(0, 1))**2, len(array))+n)).repeat(3).reshape(-1, 3)
    array = (wave*array+array)
    n += 2*np.pi*rate

    if n >= 2*np.pi:
        n = 0
    functions['waves']['n'] = n
    return array

def circle(array):
    out = np.column_stack((np.cos(2*np.pi*array[:,0]),
                           np.sin(2*np.pi*array[:,1]),
                           array[:,2]*0))
    return out

def mesh(array, name):
    fp = '/Users/joaquinllacerwintle/Google Drive/mis proyectos/oscilloscope_render/files_to_render'
    vertices, path = load_data(fp, name +'.obj')
    out = interpolate_path(vertices, path, array[:,0])
    return out


def apply_current_effect(array):
    if config.current_function in mode_functions:
        params = list(config.current_function_params.values())
        func = mode_functions[config.current_function]
        return func(array, params)
    return array


def rotx(array, n, angle, rate):
    c, s = np.cos(angle+n), np.sin(angle+n)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    n += 2*np.pi*rate
    if n >= 2*np.pi:
        n = 0
    functions['rotx']['n'] = n
    return np.dot(array, rotation_matrix.T)

def roty(array, n, angle, rate):
    """Rotate 3D vectors around the Y-axis by a given angle (in radians)."""
    c, s = np.cos(angle+n), np.sin(angle+n)
    rotation_matrix = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    n += 2*np.pi*rate
    if n >= 2*np.pi:
        n = 0
    functions['roty']['n'] = n
    return np.dot(array, rotation_matrix.T)

def rotz(array, n, angle, rate):
    """Rotate 3D vectors around the Z-axis by a given angle (in radians)."""
    c, s = np.cos(angle+n), np.sin(angle+n)
    rotation_matrix = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    n += 2*np.pi*rate
    if n >= 2*np.pi:
        n = 0
    functions['rotz']['n'] = n
    return np.dot(array, rotation_matrix.T)

def trace(array, n, length, rate):
    slice_end = np.int_(np.floor(length * len(array))) + n
    slice_end = min(slice_end, array.shape[0])
    new_indices = np.linspace(n, slice_end - 1, num=array.shape[0])

    for i in range(array.shape[1]):
        array[:, i] = np.interp(new_indices, np.arange(n, slice_end), array[n:slice_end, i])

    n = (n + rate) % array.shape[0]  # Ensure n wraps around array length
    functions['trace']['n'] = n
    return array


functions = {'time':      {'f': time, 'order': 0, 'status': 'on', 'frequency': 1, 'sample_rate': 96000//60},
             'saw_wave':  {'f': saw_wave, 'order': 1, 'status': 'on', 'frequency': 1},
             'update':    {'f': update, 'order': 2, 'status': 'off', },
             'mesh':      {'f': mesh, 'order': 3, 'status': 'on', 'name': 'octahedron'},
             'circle':    {'f': circle, 'order': 4, 'status': 'off'},
             'roses':     {'f': roses, 'order': 5, 'status': 'off', 'k': 3},
             'trace':     {'f': trace, 'order': 6, 'status': 'off', 'n': 0, 'length': 0.5, 'rate': 10},
             'mic':       {'f': mic_input, 'order': 7, 'status': 'on'},
             'waves':     {'f': waves, 'order': 8, 'status': 'on', 'n': 0, 'wf': 50, 'rate': 0.001},
             'rotx':      {'f': rotx, 'order': 9, 'status': 'on', 'n': 0, 'angle': 0, 'rate': 0.0015*2, },
             'roty':      {'f': roty, 'order': 10, 'status': 'on', 'n': 0, 'angle': 0, 'rate': 0.0010*2, },
             'rotz':      {'f': rotz, 'order': 11, 'status': 'on', 'n': 0, 'angle': 0, 'rate':0.0005*2, }}

def wave_former(input_chunk):
    array = time(*list(functions['time'].values())[3:])
    active_functions = [value['order'] for value in functions.values() if value['status'] == 'on'][1:]
    for idx in active_functions:
        f = [tuple(value.values()) for value in functions.values() if value['status'] == 'on' and value['order'] ==idx][0][0]
        if f is mic_input:
            params = [input_chunk]
        else:
            params = [tuple(value.values()) for value in functions.values() if value['status'] == 'on' and value['order'] ==idx][0][3:]
        array = f(array, *params)

    return array

