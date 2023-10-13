import numpy as np
import matplotlib.pyplot as plt

# Parameters
f = 5  # Frequency, in cycles per second, or Hertz
f_s = 100  # Sampling rate, or number of measurements per second

# Generate the time values for one second
t = np.linspace(0, 1, 2 * f_s, False)

# Generate the sinusoidal signal
x = np.sin(2 * np.pi * f * t)

# Generate the square wave
square_wave = np.where(x>=0, 1, 0)

# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Plot the sinusoidal signal
ax1.plot(t, x)
ax1.set_title('Sinusoidal Signal')

# Plot the square wave
ax2.plot(t, square_wave, color='orange')
ax2.set_title('Square Wave')

# Display the plot
plt.show()
