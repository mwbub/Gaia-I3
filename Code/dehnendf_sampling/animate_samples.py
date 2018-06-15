import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import animation
from sample_dehnendf import get_3d_samples

norbits = 1000
t = np.linspace(0, 500, 1000) * u.Myr

orbits = get_3d_samples(n=norbits, integration_steps=1000)
Rs = np.stack([o.R(t) for o in orbits], axis=1)
zs = np.stack([o.z(t) for o in orbits], axis=1)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set(xlim=(0,2), ylim=(-0.2, 0.2), xlabel='$R/R_0$', ylabel='$z/z_0$')
scatter = ax.scatter(Rs[0], zs[0], s=5)

def animate(i):
    scatter.set_offsets(np.c_[Rs[i], zs[i]])
    ax.set(title='t = {} Myr'.format(int(t[i].value)))
    
anim = animation.FuncAnimation(fig, animate, interval=33, frames=len(Rs)-1)
plt.draw()
plt.show()

if not os.path.exists('animations'):
    os.mkdir('animations')

anim.save('animations/{}_samples.gif'.format(norbits), writer='imagemagick')
