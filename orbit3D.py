"""
A script to simulate the orbits of multiple satellites around the Earth 
using Runge-Kutta 4th order method.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date: Oct 03, 2020
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# %% MAIN
# Constants
G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
M_earth = 5.972e24  # Earth's mass in kg
R_earth = 6.371e6  # Earth's radius in m

# Function to calculate initial conditions
def calculate_initial_conditions(a, e, i, m):
    v_perigee = np.sqrt(G * (M_earth + m) * (1 + e) / (a * (1 - e)))  # Velocity at perigee
    return [a * (1 - e), 0, a * np.sin(np.radians(i)), 0, v_perigee * np.cos(np.radians(i)), v_perigee * np.sin(np.radians(i))]

def create_satellites(params):
    satellites = []
    for idx, (a, e, i, m, label) in enumerate(params):
        satellites.append({
            'name': f'sat{idx+1}',
            'a': a,
            'e': e,
            'i': i,
            'm': m,
            'label': label,
            'state0': calculate_initial_conditions(a, e, i, m)
        })
    return satellites

# List of (a, e, i, m, label) parameters for each satellite
params = [
    #(7.0e6, 0.1, 20.0, 500.0, 'Satellite 1'),
    #(9.0e6, 0.2, 40.0, 1000.0, 'Satellite 2'),
    (7.0e6, 0.0, 51.6, 420.0, 'LEO'),
    (26.56e6, 0.01, 55, 2000.0, 'MEO'),
    (42.164e6, 0.0, 0, 2000.0, 'GEO')
    # add more satellites as needed
]

satellites = create_satellites(params)

# Time setup
dt = 200  # time step in seconds
t_max = 10 * np.pi * np.sqrt(max(sat['a'] for sat in satellites)**3 / (G * M_earth))  # Period of the furthest satellite
t = np.arange(0, t_max, dt)

# RK4 step
def rk4_step(func, state, t, dt, m):
    k1 = func(state, t, m)
    k2 = func(state + dt/2 * k1, t + dt/2, m)
    k3 = func(state + dt/2 * k2, t + dt/2, m)
    k4 = func(state + dt * k3, t + dt, m)
    return state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

# System of differential equations
def func(state, t, m):
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)
    ax = - G * (M_earth + m) * x / r**3
    ay = - G * (M_earth + m) * y / r**3
    az = - G * (M_earth + m) * z / r**3
    return np.array([vx, vy, vz, ax, ay, az])

# Solve system of equations
for sat in satellites:
    path = [sat['state0']]
    for t_i in t[1:]:
        path.append(rk4_step(func, path[-1], t_i, dt, sat['m']))
    sat['path'] = np.array(path)

# Convert units to thousand km
for sat in satellites:
    sat['path'][:,:3] /= 1e6
    sat['a'] /= 1e6
R_earth /= 1e6

# Calculate the maximum extent of any satellite's path
max_extent = max(np.max(np.sqrt(np.sum(sat['path'][:,:3]**2, axis=1))) for sat in satellites)/2

# Animation
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# Draw Earth
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = R_earth * np.outer(np.cos(u), np.sin(v))
y = R_earth * np.outer(np.sin(u), np.sin(v))
z = R_earth * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=0.25)

# Create lines and points for each satellite
lines = []
points = []
for sat in satellites:
    sat_line, = ax.plot([], [], [], label=sat['label'])
    lines.append(sat_line)
    sat_point, = ax.plot([], [], [], 'o')
    points.append(sat_point)
time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)
ax.legend()

def init():
    ax.set_xlim(-1.2*max_extent, 1.2*max_extent)
    ax.set_ylim(-1.2*max_extent, 1.2*max_extent)
    ax.set_zlim(-1.2*max_extent, 1.2*max_extent)
    plt.tight_layout()
    for line, point in zip(lines, points):
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
    time_text.set_text('')
    return lines + points + [time_text]

def update(frame):
    for sat, line, point in zip(satellites, lines, points):
        line.set_data(sat['path'][:frame,0], sat['path'][:frame,1])
        line.set_3d_properties(sat['path'][:frame,2])
        point.set_data(sat['path'][frame,0], sat['path'][frame,1])
        point.set_3d_properties(sat['path'][frame,2])
    time_text.set_text(f'Time = {frame*dt/3600:.1f} h')
    plt.tight_layout()
    return lines + points + [time_text]

ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=1)
#ani = FuncAnimation(fig, update, frames=np.linspace(0, len(t)-1, 200))
ani.save('satellites_orbit.gif', writer='imagemagick', fps=10)
plt.show()
# %% END