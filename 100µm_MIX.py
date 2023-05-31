# 12 Âµm PET mit AlOx beschichtet zwischen 9nm(A05) bis 11.6nm(A01) 


import xlrd
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import scipy.fftpack
from scipy.interpolate import interp1d

#from scipy import pi
#from scipy.fftpack import fft, ifft


PET_MIX_100 = "100µm_MIX.xls"

workbook = xlrd.open_workbook(PET_MIX_100)
first_sheet = workbook.sheet_by_index(0)

k_A = [first_sheet.cell_value(i, 1) for i in range(first_sheet.nrows)]    # Wellenzahl
A01 = [first_sheet.cell_value(i, 2) for i in range(first_sheet.nrows)]    # Unbeschichtete Folie mit Gold als Referenzg
A02 = [first_sheet.cell_value(i, 3) for i in range(first_sheet.nrows)]     # Beschichtete Folie (30nm AlOx) mit Gold als Referenz
A03 = [first_sheet.cell_value(i, 4) for i in range(first_sheet.nrows)]    # Unbeschichtete Folie mit unbeschichteter Folie als Referenz
A04 = [first_sheet.cell_value(i, 5) for i in range(first_sheet.nrows)]    #Beschichtete Folie (30nm AlOx) mit unbeschichteter Folie als Referenz
A05 = [first_sheet.cell_value(i, 6) for i in range(first_sheet.nrows)]
A06 = [first_sheet.cell_value(i, 7) for i in range(first_sheet.nrows)]
A07 = [first_sheet.cell_value(i, 8) for i in range(first_sheet.nrows)]
A08 = [first_sheet.cell_value(i, 9) for i in range(first_sheet.nrows)]
A09 = [first_sheet.cell_value(i, 10) for i in range(first_sheet.nrows)]
A10 = [first_sheet.cell_value(i, 11) for i in range(first_sheet.nrows)]

k_A = k_A[2:]
A01 = A01[2:]
A02 = A02[2:]
A03 = A03[2:]
A04 = A04[2:]
A05 = A05[2:]
A06 = A06[2:]
A07 = A07[2:]
A08 = A08[2:]
A09 = A09[2:]
A10 = A10[2:]

plt.figure()
plt.plot(k_A, A01)
plt.plot(k_A, A02)

plt.figure()
plt.plot(k_A[6400:7100], A01[6400:7100])
plt.plot(k_A[6400:7100], A02[6400:7100])


plt.figure()
plt.plot(k_A, A03)
plt.plot(k_A, A04)

plt.figure()
plt.plot(k_A[6400:7100], A03[6400:7100])
plt.plot(k_A[6400:7100], A04[6400:7100])

plt.figure()
plt.plot(k_A, A05)
plt.plot(k_A, A06)

plt.figure()
plt.plot(k_A[6400:7100], A05[6400:7100])
plt.plot(k_A[6400:7100], A06[6400:7100])
plt.title('Hauptpeak')

plt.figure()
plt.plot(k_A[5200:6200], A05[5200:6200])
plt.plot(k_A[5200:6200], A06[5200:6200])
plt.title('Substratpeak 1 ')

plt.figure()
plt.plot(k_A[4600:5000], A05[4600:5000])
plt.plot(k_A[4600:5000], A06[4600:5000])
plt.title('Substratpeak 2')

plt.figure()
plt.plot(k_A, A07)
plt.plot(k_A, A08)

plt.figure()
plt.plot(k_A[6400:7100], A07[6400:7100])
plt.plot(k_A[6400:7100], A08[6400:7100])
plt.title('Hauptpeak')

plt.figure()
plt.plot(k_A[5200:6200], A07[5200:6200])
plt.plot(k_A[5200:6200], A08[5200:6200])
plt.title('Substratpeak 1')

plt.figure()
plt.plot(k_A[4200:5200], A07[4200:5200])
plt.plot(k_A[4200:5200], A08[4200:5200])
plt.title('Substratpeak 2')


plt.figure()
plt.plot(k_A, A09)
plt.plot(k_A, A10)

plt.figure()
plt.plot(k_A[6400:7100], A09[6400:7100])
plt.plot(k_A[6400:7100], A10[6400:7100])



