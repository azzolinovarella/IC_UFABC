import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

fig1 = plt.figure(1)
camera = Camera(fig1)
x = np.linspace(0, 2 * np.pi, 361)
y = np.sin(x)

for cont in range(0, len(x)):
    x_t = x[:cont]
    y_t = y[:cont]
    plt.plot(x_t, y_t, color='blue')
    camera.snap()
animation = camera.animate()
animation.save('teste.gif', fps=15)
