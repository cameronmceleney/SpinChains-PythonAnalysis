import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Time step for the animation (s), max time to animate for (s).
dt, tmax = 0.01, 4.015
# Signal frequency (s-1), decay constant (s-1).
f, alpha = 1.5, 1
# These lists will hold the data to plot.
t, M, N = [], [], []
tvb = True

# Draw an empty plot, but preset the plot x- and y-limits.
fig, ax = plt.subplots()
line1, = ax.plot([], [], color= '#64bb6a', lw=3, zorder=1.1)
line2, = ax.plot([], [], color='#37782c', lw=2, zorder=1.05)
ax.set_xlim(-0.2, 4.2)
ax.set_ylim(-0.51, 1.2)
#ax.axis('off')
#ax.set_xlabel('t /s')
#ax.set_ylabel('M (arb. units)')

ax.vlines(x=[-0.03, 4.03], ymin=-0.5, ymax=0.5,
           colors='black', lw=5,
           label='vline_multiple - full height')

ax.text(-0.3, 0.42, 'A', fontsize=16)
ax.text(4.15, 0.42, 'B', fontsize=16)
ax.text(-0.3, 0.0, 'OFF', va='center', ha='center', fontsize=12)
ax.text(-0.3, 1.0, 'ON', va='center', ha='center', fontsize=12)

def animate1(i):
    """Draw the frame i of the animation."""

    global t, M, N
    # Append this time point and its data and set the plotted line data.
    _t = i*dt
    A = 0.5
    t.append(_t)
    sinusoid = A*np.sin(2 * np.pi * f * _t)
    if sinusoid < 0:
       tv = 0
    else:
       tv = 1

    M.append(sinusoid)
    N.append(tv)
    line1.set_data(t, M)
    line2.set_data(t, N)
def animate(i):
    """Draw the frame i of the animation."""

    global t, M, N
    # Append this time point and its data and set the plotted line data.
    _t = i*dt
    A = 0.5
    lim = 0.4
    t.append(_t)
    tv = A*np.sin(2 * np.pi * f * _t)
    if tv < lim:
       tv = 0
    elif tv >= lim :
       tv = 1

    M.append(A*np.sin(2*np.pi*f*_t))
    N.append(tv)
    line1.set_data(t, M)
    line2.set_data(t, N)

# Interval between frames in ms, total number of frames to use.
interval, nframes = 1000 * dt, int(tmax / dt)
# Animate once (set repeat=False so the animation doesn't loop).
ani = animation.FuncAnimation(fig, animate1, frames=nframes, repeat=False,
                              interval=interval)
plt.show()
