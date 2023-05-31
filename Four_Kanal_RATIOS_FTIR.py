# Berechnung der Integrale von FTIR Spektren für verschiedenen Channel von 4-Kanal Detektor
# und 8-14µm Detektor mit Einbezug des Planckspektrums 


import xlrd
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import scipy.fftpack
from scipy.interpolate import interp1d
from scipy.integrate import trapz, simps
import math

c = 299792458              # speed of light in m/s
h = 6.62607004e-34         # Planck constant in Js
k = 1.38064852e-23         # Boltzmann constant J/K

# Infrasolid HIS2000R-BWC300
T_C = 1160                  # temperature in °C
area = 31                  # emitting area in mm2
emissivity = 0.8


# filter position
lambda_0 = 9.5 #7.775        # center Wavelength (µm)
FWHM = 0.45 #14.45           # (µm)


T_K = T_C + 273.15         # temperature in K
nu_min = 10000/(lambda_0 + FWHM/2)
nu_max = 10000/(lambda_0 - FWHM/2)

def M_nu(nu, T):           # spectral radiance in mW/(cm2 µm)
    return(
        #( h * c**2 * nu**3 ) / np.expm1( (h * c * nu)/(k * T) )
                           # Stöcker
        ( 2 * np.pi * h * c**2 * nu**3 ) / np.expm1( (h * c * nu)/(k * T) )
                           # replace lambda = 1/nu, add factor d lamda / d nu = 1/ nu**2
                           # expm1(.) = exp(.)-1, spectral radiance in W/(m2 / m)
        * 10               # in mW/(cm2 µm)
    )


def l(nu):                  # wavelength in m
    return( 1 /( 100 * nu )) # nu: wavenumber in 1/cm


def M_l(nu, T):            # spectral radiance in mW/(cm2 µm)
    return(
        ( 2 * np.pi * h * c**2 / l(nu)**5 ) / np.expm1( (h * c)/(l(nu) * k * T) )
                           # expm1(.) = exp(.)-1, spectral radiance in W/(m2 m)
        * 1e-7             # in mW/(cm2 µm)
    )

def M_l_l(l, T):           # spectral radiance in mW/(cm2 µm)
    return(
        ( 2 * np.pi * h * c**2 / l**5 ) / np.expm1( (h * c)/(l * k * T) )
                           # expm1(.) = exp(.)-1, spectral radiance in W/(m2 m)
        * 1e-7             # in mW/(cm2 µm)
    )

nu_list_integral, nu_step = np.linspace(nu_min, nu_max, num=100, endpoint=True, retstep=True)
M_nu_list = M_nu(nu_list_integral * 100, T_K)       # * 100: nu in 1/m
int_power = np.trapz(M_nu_list, dx = nu_step) * (area * 1e-2) * emissivity

print('Power = ', int_power, ' mW')



# plotting
l_list = np.linspace(0.1, 20, num=100, endpoint=True)
spectral_radiance_l = M_l_l(l_list * 1e-6, T_K)    # * 1e-6: l in µm
"""
#plt.plot(l_list, spectral_radiance_l * (area * 1e-2) / np.pi, color='black')
plt.plot(l_list, spectral_radiance_l, color='black')
plt.xlabel(u'Wellenlänge (µm)', fontsize=16)
plt.ylabel(u'Emission (mW cm$^{-2}$ µm$^{-1}$)', fontsize=16)
plt.tick_params(labelsize=16)
plt.title(u'%d°C' % T_C, fontsize=16);

plt.show()
"""

########
# Leistung der Lichtquelle in mW für die jeweiligen Wellenlängenbereiche der Filter
########

planck_8_14 = 280.14713572278447 
planck_9_5  = 4.3 * 28.78860095402817     # O-Filter 9.5µm ; FWHM 450nm
planck_ch4  = 12 * 26.232557038206995
planck_ch3  = 30.067160964626535
planck_ch2  = 30.297058659310824
planck_ch1  = 74.2430011408125



####################################################
# 4-Kanal DETEKTOR Berechnung Integrale von FTIR für die 4 Kanäle und Berechnung
# der Verhältnisse zwischen CH4 und CH3 
####################################################

PET_MIX_100 = "100µm_MIX_2.xls"
Round_test = "Round_test.xls"
Becher_AlOx_var_Schichtdicke = "Becher_AlOx_var_Schichtdicke.xls"

workbook_becher = xlrd.open_workbook(Becher_AlOx_var_Schichtdicke)
first_sheet_becher = workbook_becher.sheet_by_index(0)

Becher_5nm = [first_sheet_becher.cell_value(i, 2) for i in range(first_sheet_becher.nrows)]   
Becher_30nm = [first_sheet_becher.cell_value(i, 5) for i in range(first_sheet_becher.nrows)]
Becher_50nm = [first_sheet_becher.cell_value(i, 6) for i in range(first_sheet_becher.nrows)]

workbook = xlrd.open_workbook(PET_MIX_100)
first_sheet = workbook.sheet_by_index(0)

workbook_2 = xlrd.open_workbook(Round_test)
first_sheet_2 = workbook_2.sheet_by_index(0)

k_A = [first_sheet.cell_value(i, 1) for i in range(first_sheet.nrows)]    # Wellenzahl
oBesch = [first_sheet_2.cell_value(i, 2) for i in range(first_sheet_2.nrows)]  
oBesch_2 = [first_sheet_2.cell_value(i, 3) for i in range(first_sheet_2.nrows)]  
AlOx = [first_sheet_2.cell_value(i, 4) for i in range(first_sheet_2.nrows)]  
AlOx_2 = [first_sheet_2.cell_value(i, 5) for i in range(first_sheet_2.nrows)]  



k_A = k_A[2:]
oBesch = oBesch[2:]
oBesch_2 = oBesch_2[2:]
AlOx = AlOx[2:]
AlOx_2 = AlOx_2[2:]
Becher_5nm = Becher_5nm[2:]
Becher_30nm = Becher_30nm[2:]
Becher_50nm = Becher_50nm[2:]
normierung = np.full(7199, 100)


 
offset = 0.5
offset_2 = 0
offset_3 = 0.2
oBesch = [oBesch[i] + offset for i in range(len(oBesch))]
oBesch_2 = [oBesch_2[i] + offset_2 for i in range(len(oBesch_2))]
Becher_5nm = [Becher_5nm[i] + offset_3 for i in range(len(Becher_5nm))]

#AlOx = [AlOx[i] + offset for i in range(len(oBesch))]
#AlOx_2 = [AlOx_2[i] + offset_2 for i in range(len(oBesch_2))]



oBesch_int = interp1d(k_A, oBesch, kind='linear')
oBesch_int_2 = interp1d(k_A, oBesch_2, kind='linear')
AlOx_int = interp1d(k_A, AlOx, kind='linear')
AlOx_int_2 = interp1d(k_A, AlOx_2, kind='linear')
Becher_5nm_fit = interp1d(k_A, Becher_5nm, kind='linear')
Becher_30nm_fit = interp1d(k_A, Becher_30nm, kind='linear')
Becher_50nm_fit = interp1d(k_A, Becher_50nm, kind='linear')

normierung_fit = interp1d(k_A, normierung, kind='linear')


ch1_x = np.linspace(2503, 2561, 124)
ch2_x = np.linspace(1351, 1389, 80)
ch3_x = np.linspace(1034, 1086, 113)
ch4_x = np.linspace(782, 848, 141)
Det_O_9_5_x = np.linspace(1028,1078,107)
Det_8_14_x = np.linspace(714,1250, 1152)
#ganzes_spektrum = np.linspace(651, 3999, 7199)




plt.figure()
plt.plot(k_A, Becher_5nm, linewidth=0.75)
plt.plot(k_A, Becher_30nm, linewidth=0.75)
#plt.plot(k_A, Becher_50nm, linewidth=0.75)
plt.legend(['5nm', '30nm', '50nm'])


plt.figure()
plt.plot(k_A, oBesch, linewidth=0.75)
plt.plot(k_A, AlOx, linewidth=0.75)


#integral_normierung = simps(oBesch_int(ganzes_spektrum))

#normierung_Det_8_14 = 


#Channel 1 Wertebereich: 2503 - 2561 Wellenzahlen
#Channel 2 Wertebereich: 1351 - 1389 Wellenzahlen
#Channel 3 Wertebereich: 1034 - 1086 Wellenzahlen
#Channel 4 Wertebereich: 782 - 848 Wellenzahlen

CH4_unb = (simps(oBesch_int(ch4_x)) / simps(normierung_fit(ch4_x))) * planck_ch4
CH3_unb = (simps(oBesch_int(ch3_x)) / simps(normierung_fit(ch3_x))) * planck_ch3
CH2_unb = (simps(oBesch_int(ch2_x)) / simps(normierung_fit(ch2_x))) * planck_ch2
CH1_unb = (simps(oBesch_int(ch1_x)) / simps(normierung_fit(ch1_x))) * planck_ch1
Det_8_14_unb = (simps(oBesch_int(Det_8_14_x)) / simps(normierung_fit(Det_8_14_x))) * planck_8_14
Det_O_9_5_unb = (simps(oBesch_int(Det_O_9_5_x)) / simps(normierung_fit(Det_O_9_5_x))) * planck_9_5

Becher_5nm_8_14 = (simps(Becher_5nm_fit(Det_8_14_x)) / simps(normierung_fit(Det_8_14_x))) * planck_8_14
Becher_30nm_8_14 = (simps(Becher_30nm_fit(Det_8_14_x)) / simps(normierung_fit(Det_8_14_x))) * planck_8_14
Becher_50nm_8_14 = (simps(Becher_50nm_fit(Det_8_14_x)) / simps(normierung_fit(Det_8_14_x))) * planck_8_14


Becher_5nm_ch4 = (simps(Becher_5nm_fit(ch4_x)) / simps(normierung_fit(ch4_x))) * planck_ch4
Becher_30nm_ch4 = (simps(Becher_30nm_fit(ch4_x)) / simps(normierung_fit(ch4_x))) * planck_ch4
Becher_50nm_ch4 = (simps(Becher_50nm_fit(ch4_x)) / simps(normierung_fit(ch4_x))) * planck_ch4



Becher_5nm_O_9_5 = (simps(Becher_5nm_fit(Det_O_9_5_x)) / simps(normierung_fit(Det_O_9_5_x))) * planck_9_5
Becher_30nm_O_9_5 = (simps(Becher_30nm_fit(Det_O_9_5_x)) / simps(normierung_fit(Det_O_9_5_x))) * planck_9_5
Becher_50nm_O_9_5 = (simps(Becher_50nm_fit(Det_O_9_5_x)) / simps(normierung_fit(Det_O_9_5_x))) * planck_9_5


CH4_AlOx = (simps(AlOx_int(ch4_x)) / simps(normierung_fit(ch4_x))) * planck_ch4
CH3_AlOx = (simps(AlOx_int(ch3_x)) / simps(normierung_fit(ch3_x))) * planck_ch3
CH2_AlOx = (simps(AlOx_int(ch2_x)) / simps(normierung_fit(ch2_x))) * planck_ch2
CH1_AlOx = (simps(AlOx_int(ch1_x)) / simps(normierung_fit(ch1_x))) * planck_ch1
Det_8_14_AlOx = (simps(AlOx_int(Det_8_14_x)) / simps(normierung_fit(Det_8_14_x))) * planck_8_14
Det_O_9_5_AlOx = (simps(AlOx_int(Det_O_9_5_x)) / simps(normierung_fit(Det_O_9_5_x))) * planck_9_5



CH4_unb_2 = (simps(oBesch_int_2(ch4_x)) / simps(normierung_fit(ch4_x))) * planck_ch4
CH3_unb_2 = (simps(oBesch_int_2(ch3_x)) / simps(normierung_fit(ch3_x))) * planck_ch3
CH2_unb_2 = (simps(oBesch_int_2(ch2_x)) / simps(normierung_fit(ch2_x))) * planck_ch2
CH1_unb_2 = (simps(oBesch_int_2(ch1_x)) / simps(normierung_fit(ch1_x))) * planck_ch1
Det_8_14_unb_2 = (simps(oBesch_int_2(Det_8_14_x)) / simps(normierung_fit(Det_8_14_x))) * planck_8_14
Det_O_9_5_unb_2 = (simps(oBesch_int_2(Det_O_9_5_x)) / simps(normierung_fit(Det_O_9_5_x))) * planck_9_5



CH4_AlOx_2 = (simps(AlOx_int_2(ch4_x)) / simps(normierung_fit(ch4_x))) * planck_ch4
CH3_AlOx_2 = (simps(AlOx_int_2(ch3_x)) / simps(normierung_fit(ch3_x))) * planck_ch3
CH2_AlOx_2 = (simps(AlOx_int_2(ch2_x)) / simps(normierung_fit(ch2_x))) * planck_ch2
CH1_AlOx_2 = (simps(AlOx_int_2(ch1_x)) / simps(normierung_fit(ch1_x))) * planck_ch1
Det_8_14_AlOx_2 = (simps(AlOx_int_2(Det_8_14_x)) / simps(normierung_fit(Det_8_14_x))) * planck_8_14
Det_O_9_5_AlOx_2 = (simps(AlOx_int_2(Det_O_9_5_x)) / simps(normierung_fit(Det_O_9_5_x))) * planck_9_5

becher_ratio_5 = Becher_5nm_8_14 / Becher_5nm_O_9_5
becher_ratio_30 = Becher_30nm_8_14 / Becher_30nm_O_9_5
becher_ratio_50 = Becher_50nm_8_14 / Becher_50nm_O_9_5

becher_ratio_5_ch4 = Becher_5nm_ch4 / Becher_5nm_O_9_5
becher_ratio_30_ch4 = Becher_30nm_ch4 / Becher_30nm_O_9_5
becher_ratio_50_ch4 = Becher_50nm_ch4 / Becher_50nm_O_9_5

becher_ratio_ratio_30v5 = becher_ratio_30 / becher_ratio_5
becher_ratio_ratio_50v5 = becher_ratio_50 / becher_ratio_5

becher_ratio_ratio_30v5_ch4 = becher_ratio_30_ch4 / becher_ratio_5_ch4
becher_ratio_ratio_50v5_ch4 = becher_ratio_50_ch4 / becher_ratio_5_ch4



print('Becher ratio 30vs5nm 8-14µm: ', becher_ratio_ratio_30v5)
print('Becher ratio 50vs5nm 8-14µm: ', becher_ratio_ratio_50v5)

print('Becher ratio 30vs5nm CH4: ', becher_ratio_ratio_30v5_ch4)
print('Becher ratio 50vs5nm CH4: ', becher_ratio_ratio_50v5_ch4)



ratio_cage_unb = Det_8_14_unb / Det_O_9_5_unb
ratio_cage_AlOx = Det_8_14_AlOx / Det_O_9_5_AlOx
ratio_ratio_cage = ratio_cage_AlOx / ratio_cage_unb


ratio_ch4_O_unb = CH4_unb / Det_O_9_5_unb
ratio_ch4_O_AlOx = CH4_AlOx / Det_O_9_5_AlOx
ratio_ratio_ch4 = ratio_ch4_O_AlOx / ratio_ch4_O_unb

print('Ratio unbeschichtet: ', ratio_cage_unb)
print('Ratio AlOx: ', ratio_cage_AlOx)
print('Ratio Ratio cage: ', ratio_ratio_cage)

print('Ratio CH4 unbeschichtet: ', ratio_ch4_O_unb)
print('Ratio CH4 AlOx: ', ratio_ch4_O_AlOx)
print('Ratio Ratio ch4: ', ratio_ratio_ch4)



plt.figure()
plt.plot(k_A[3091:3215], oBesch[3091:3215], linewidth=0.75)
plt.plot(k_A[3091:3215], AlOx[3091:3215], linewidth=0.75)
plt.legend(['unbeschichtet', 'AlOx'])
plt.title('CH1')

plt.figure()
plt.plot(k_A[5610:5690], oBesch[5610:5690], linewidth=0.75)
plt.plot(k_A[5610:5690], AlOx[5610:5690], linewidth=0.75)
plt.legend(['unbeschichtet', 'AlOx'])
plt.title('CH2')

plt.figure()
plt.plot(k_A[6260:6370], oBesch[6260:6370], linewidth=0.75)
plt.plot(k_A[6260:6370], AlOx[6260:6370], linewidth=0.75)
plt.legend(['unbeschichtet', 'AlOx'])
plt.title('CH3')

plt.figure()
plt.plot(k_A[6770:6914], oBesch[6770:6914], linewidth=0.75)
plt.plot(k_A[6770:6914], AlOx[6770:6914], linewidth=0.75)
plt.legend(['unbeschichtet', 'AlOx'])
plt.title('CH4')

plt.figure()
plt.plot(k_A[5908:7060], oBesch[5908:7060], linewidth=0.75)
plt.plot(k_A[5908:7060], AlOx[5908:7060], linewidth=0.75)
plt.legend(['unbeschichtet', 'AlOx'])
plt.title('8-14µm Detektor')

plt.figure()
plt.plot(k_A[6278:6385], oBesch[6278:6385], linewidth=0.75)
plt.plot(k_A[6278:6385], AlOx[6278:6385], linewidth=0.75)
plt.legend(['unbeschichtet', 'AlOx'])
plt.title('Detektor O 9.5µm')



















"""
print('Channel 4-1 Unbeschichtet')
print('Leistung in mW CH4 unbeschichtet: ', CH4_unb)
print('Leistung in mW CH3 unbeschichtet: ', CH3_unb)
print('Leistung in mW CH2 unbeschichtet: ', CH2_unb)
print('Leistung in mW CH1 unbeschichtet: ', CH1_unb)
print('Leistung in mW 8-14µm Detektor unbeschichtet: ', Det_8_14_unb)

print('')
print('Channel 4-1 AlOx')
print('Leistung in mW CH4 AlOx: ', CH4_AlOx)
print('Leistung in mW CH3 AlOx: ', CH3_AlOx)
print('Leistung in mW CH2 AlOx: ', CH2_AlOx)
print('Leistung in mW CH1 AlOx: ', CH1_AlOx)
print('Leistung in mW 8-14µm Detektor AlOx: ', Det_8_14_AlOx)

print('')
print('RATIO CH4 AlOx zu unbeschichtet: ', CH4_AlOx / CH4_unb)
print('RATIO CH3 AlOx zu unbeschichtet: ', CH3_AlOx / CH3_unb)
print('RATIO CH2 AlOx zu unbeschichtet: ', CH2_AlOx / CH2_unb)
print('RATIO CH1 AlOx zu unbeschichtet: ', CH1_AlOx / CH1_unb)
print('RATIO 8-14µm AlOx zu unbeschichtet: ', Det_8_14_AlOx / Det_8_14_unb)

print('')
print('Differenz CH4 AlOx zu unbeschichtet: ', CH4_AlOx - CH4_unb)
print('Differenz CH3 AlOx zu unbeschichtet: ', CH3_AlOx - CH3_unb)
print('Differenz CH2 AlOx zu unbeschichtet: ', CH2_AlOx - CH2_unb)
print('Differenz CH1 AlOx zu unbeschichtet: ', CH1_AlOx - CH1_unb)
print('Differenz 8-14µm AlOx zu unbeschichtet: ', Det_8_14_AlOx - Det_8_14_unb)




plt.figure()
plt.plot(k_A[4800:7000], oBesch_2[4800:7000], linewidth=0.75)
plt.plot(k_A[4800:7000], AlOx_2[4800:7000], linewidth=0.75)
plt.show()

nu_list = np.linspace(1, 5000, num=100, endpoint=True)
spectral_radiance_l = M_l(nu_list, T_K)

plt.plot(nu_list, spectral_radiance_l, color='black')
plt.xlabel(u'Wellenzahl (cm$^{-1}$)', fontsize=16)
plt.ylabel(u'Emission (mW cm$^{-2}$ µm$^{-1}$)', fontsize=16)
plt.tick_params(labelsize=16)
plt.title(u'%d°C' % T_C, fontsize=16);

plt.show()


spectral_radiance_nu = M_nu(nu_list * 100, T_K)     # * 100: nu in 1/m

plt.plot(nu_list, spectral_radiance_nu, color='black')
plt.xlabel(u'Wellenzahl (cm$^{-1}$)', fontsize=16)
plt.ylabel(u'Emission (mW cm$^{-2}$ / cm$^{-1}$)', fontsize=16)
plt.tick_params(labelsize=16)
plt.title(u'%d°C' % T_C, fontsize=16);

plt.show()


def divergence(g, r_ursp, angle):        # Calculates the reduction of power through increasment of the beam area through divergence
    return( 
        (r_ursp**2) / (r_ursp + (np.tan(math.radians(angle)) * g))**2
        )

x = divergence(30, 2.5, 5)
power_in_divergence = 0.77
efficiency_optics_size = 0.9
eff_filter = 0.8
reflectance = 0.07
beam_split = 0.5

final_power = int_power * reflectance * efficiency_optics_size**2 * divergence(30,2.5,5) * eff_filter * power_in_divergence * beam_split

print('Leistung am Detektor: ', final_power, 'mW')



t_det = 3*1e-9      # Time constant 
D = 9e8             # Detectivity in (cm * sqrt(Hz) / W)
A = 0.01            # Detector Area in cm^2

NEP = (np.sqrt(A * 1 / (2*t_det)) / D)   # Noise equivalent Power

print('NEP: ', NEP * 1000, 'mW')
"""









