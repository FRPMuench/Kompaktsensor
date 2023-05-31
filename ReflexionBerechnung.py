import numpy as np
import math
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

A1 = 10
A2= 6
d = np.linspace(0, 100, 100)
absorption=-((1-0.92)/100)*d+1

#plt.plot(d, absorption)
#plt.show()

Amplitude_res_peak=A1-A2*absorption
Amplitude_res_ref=A2

Intensität_res_peak=Amplitude_res_peak**2
Intensität_res_ref=Amplitude_res_ref**2
Intensität=Amplitude_res_peak/Amplitude_res_ref

plt.plot(d, Intensität_res_peak)
plt.xlabel('Schichtdicke')
plt.ylabel('Intensität')
plt.title(A2)
plt.grid(True)
plt.show()



S=(1.00-0.92)/100
d=10
Steigung=2*(A1/A2)*S+2*(S*d+1)*S
Teil1=2*(A1/A2)*S
Teil2=2*(S*d+1)*S

print(f'Steigung für d={d}')
print(Steigung)
print(f'teil1 ={Teil1} ')
print(f'teil2 ={Teil2} ')