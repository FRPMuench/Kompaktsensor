import xlrd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.integrate as integrate
from numpy import pi, linspace, inf, array, genfromtxt
from tmm import (coh_tmm, inc_tmm, unpolarized_RT, ellips, position_resolved, find_in_structure_with_inf, absorp_in_each_layer)
from scipy.interpolate import interp1d
import math
from scipy.stats import linregress


Zeonor_FTIR = "Zeonor_Noppen_FTIR.xls"


workbook2 = xlrd.open_workbook(Zeonor_FTIR)
first_sheet2 = workbook2.sheet_by_index(0)




k = [first_sheet2.cell_value(i, 1) for i in range(first_sheet2.nrows)]
zeonor_25 = [first_sheet2.cell_value(i, 2) for i in range(first_sheet2.nrows)]    # Unbeschichtete Folie mit Gold als Referenz
zeonor_25_rück = [first_sheet2.cell_value(i, 3) for i in range(first_sheet2.nrows)] 
zeonor_30 = [first_sheet2.cell_value(i, 4) for i in range(first_sheet2.nrows)] 
zeonor_30_rück = [first_sheet2.cell_value(i, 5) for i in range(first_sheet2.nrows)] 
zeonor_35 = [first_sheet2.cell_value(i, 6) for i in range(first_sheet2.nrows)] 
zeonor_35_rück = [first_sheet2.cell_value(i, 7) for i in range(first_sheet2.nrows)] 
zeonor_38 = [first_sheet2.cell_value(i, 8) for i in range(first_sheet2.nrows)] 
zeonor_38_rück = [first_sheet2.cell_value(i, 9) for i in range(first_sheet2.nrows)] 
zeonor_48 = [first_sheet2.cell_value(i, 10) for i in range(first_sheet2.nrows)] 
zeonor_48_rück = [first_sheet2.cell_value(i, 11) for i in range(first_sheet2.nrows)] 
zeonor_60 = [first_sheet2.cell_value(i, 12) for i in range(first_sheet2.nrows)] 
zeonor_60_rück = [first_sheet2.cell_value(i, 13) for i in range(first_sheet2.nrows)] 

k = k[1:]

zeonor_25 = zeonor_25[1:]
zeonor_25_rück = zeonor_25_rück[1:]
zeonor_30 = zeonor_30[1:]
zeonor_30_rück = zeonor_30_rück[1:]
zeonor_35 = zeonor_35[1:]
zeonor_35_rück = zeonor_35_rück[1:]
zeonor_38 = zeonor_38[1:]
zeonor_38_rück = zeonor_38_rück[1:]
zeonor_48 = zeonor_48[1:]
zeonor_48_rück = zeonor_48_rück[1:]
zeonor_60 = zeonor_60[1:]
zeonor_60_rück = zeonor_60_rück[1:]


#print(zeonor_60.index(max(zeonor_60)))



plt.figure()
plt.rcParams.update({'font.size': 8})
plt.plot(k[5900:6600], zeonor_25_rück[5900:6600])
plt.plot(k[5900:6600], zeonor_25[5900:6600])
plt.plot(k[5900:6600], zeonor_30[5900:6600])
plt.plot(k[5900:6600], zeonor_35[5900:6600])
plt.plot(k[5900:6600], zeonor_38[5900:6600])
plt.plot(k[5900:6600], zeonor_48[5900:6600])
plt.plot(k[5900:6600], zeonor_60[5900:6600])
plt.xlabel('Wellenzahl')
plt.ylabel('Reflexion')
plt.legend(['Unbeschichtet', 'SiOx_25nm', 'SiOx 30nm', 'SiOx 35.6nm', 'SiOx 38.9nm', 'SiOx 48.3nm', 'SiOx 60.8nm'])
plt.title('FTIR SiOx beschichtete Plastikteile')


vh25_rück = (zeonor_25_rück[6300] / zeonor_25_rück[5908])
vh30_rück = (zeonor_30_rück[6300] / zeonor_30_rück[5908])
vh35_rück = (zeonor_35_rück[6300] / zeonor_35_rück[5908])
vh38_rück = (zeonor_38_rück[6300] / zeonor_38_rück[5908])
vh48_rück = (zeonor_48_rück[6300] / zeonor_48_rück[5908])
vh60_rück = (zeonor_60_rück[6300] / zeonor_60_rück[5908])


vh_rück = (vh25_rück + vh30_rück + vh35_rück + vh38_rück + vh48_rück + vh60_rück) / 6
vh25 = (max(zeonor_25) / zeonor_25[5908])
vh30 = (max(zeonor_30) / zeonor_30[5908])
vh35 = (max(zeonor_35) / zeonor_35[5908])
vh38 = (max(zeonor_38) / zeonor_38[5908])
vh48 = (max(zeonor_48) / zeonor_48[5908])
vh60 = (max(zeonor_60) / zeonor_60[5908])

vh = [vh_rück, vh25, vh30, vh35, vh38, vh48, vh60]
Schichtdicken = [0, 25, 30, 35.6, 38.9, 48.3, 60.8]

m, t, r, p, std = linregress(Schichtdicken, vh)


plt.figure()
plt.scatter(Schichtdicken, vh)
plt.plot([0,60],[t,t+60*m],c="red",alpha=0.5)
plt.xlabel('Schichtdicke')
plt.ylabel('Verhältnis Peak zu Referenz')
plt.title('FTIR Messung')

##########################




k = [10000/first_sheet2.cell_value(i, 1) for i in range(first_sheet2.nrows)]
zeonor_25 = [first_sheet2.cell_value(i, 2) for i in range(first_sheet2.nrows)]    # Unbeschichtete Folie mit Gold als Referenz
zeonor_25_rück = [first_sheet2.cell_value(i, 3) for i in range(first_sheet2.nrows)]
zeonor_30 = [first_sheet2.cell_value(i, 4) for i in range(first_sheet2.nrows)]
zeonor_30_rück = [first_sheet2.cell_value(i, 5) for i in range(first_sheet2.nrows)]
zeonor_35 = [first_sheet2.cell_value(i, 6) for i in range(first_sheet2.nrows)]
zeonor_35_rück = [first_sheet2.cell_value(i, 7) for i in range(first_sheet2.nrows)]
zeonor_38 = [first_sheet2.cell_value(i, 8) for i in range(first_sheet2.nrows)]
zeonor_38_rück = [first_sheet2.cell_value(i, 9) for i in range(first_sheet2.nrows)]
zeonor_48 = [first_sheet2.cell_value(i, 10) for i in range(first_sheet2.nrows)]
zeonor_48_rück = [first_sheet2.cell_value(i, 11) for i in range(first_sheet2.nrows)]
zeonor_60 = [first_sheet2.cell_value(i, 12) for i in range(first_sheet2.nrows)]
zeonor_60_rück = [first_sheet2.cell_value(i, 13) for i in range(first_sheet2.nrows)]

k = k[1:]

zeonor_25 = zeonor_25[1:]
zeonor_25_rück = zeonor_25_rück[1:]
zeonor_30 = zeonor_30[1:]
zeonor_30_rück = zeonor_30_rück[1:]
zeonor_35 = zeonor_35[1:]
zeonor_35_rück = zeonor_35_rück[1:]
zeonor_38 = zeonor_38[1:]
zeonor_38_rück = zeonor_38_rück[1:]
zeonor_48 = zeonor_48[1:]
zeonor_48_rück = zeonor_48_rück[1:]
zeonor_60 = zeonor_60[1:]
zeonor_60_rück = zeonor_60_rück[1:]


#print(zeonor_60.index(max(zeonor_60)))



plt.figure()
plt.rcParams.update({'font.size': 8})
plt.plot(k[1430:6600], zeonor_25_rück[1430:6600])
plt.plot(k[1430:6600], zeonor_25[1430:6600])
plt.plot(k[1430:6600], zeonor_30[1430:6600])
plt.plot(k[1430:6600], zeonor_35[1430:6600])
plt.plot(k[1430:6600], zeonor_38[1430:6600])
plt.plot(k[1430:6600], zeonor_48[1430:6600])
plt.plot(k[1430:6600], zeonor_60[1430:6600])
plt.xlabel('Wellenlänge')
plt.ylabel('Reflexion')
plt.legend(['Unbeschichtet', 'SiOx_25nm', 'SiOx 30nm', 'SiOx 35.6nm', 'SiOx 38.9nm', 'SiOx 48.3nm', 'SiOx 60.8nm'])
plt.title('FTIR SiOx beschichtete Plastikteile')




