"""
Plot power curves. Rik Henson.
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('power_analysis_results.csv', delimiter=',', skiprows=1)

snr = data[:, 0]
pow = 100.0*data[:, 2] # K>=2 (not K==2) 

fig, axes = plt.subplots(1, 1, figsize=(5, 4.5))
ax = axes[0] if isinstance(axes, np.ndarray) else axes

ax.plot(snr, pow, 'k-', lw=1.5, zorder=3)
ax.set_xlabel('SNR', fontsize=12)
ax.set_ylabel('Power (%)')
ax.set_title('Recovery of multi-state model (BIC)')
plt.tight_layout()

plt.savefig('power_curve.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved power_curve.png')
