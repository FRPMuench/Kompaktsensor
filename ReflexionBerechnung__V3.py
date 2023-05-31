import numpy as np
import math
import matplotlib.pyplot as plt
import cmath

####
#Die Berechung gilt für senkrecht einfallendes Licht. Daher wird nicht in s und p polarisiert unterschieden
#Das Substrat wird als unendlich ausgedehnt betrachtet
#R ist die Reflektivität und bezieht sich auf die Intensität, r ist der reflexionskoeffizient und bezieht sich auf die Amplitude

lamda=13.6612*1000 #nanometer #hier die Wellenlänge eintragen. Der brechungsindex muss darauf angepasst sein

#d=60 #Schichtdicke in nanometer
# Initialisierung von d
d_values = np.arange(0, 4000, 1)  # in nm, Schichtdicken

n_strich_Luft=1.0003 #realteil von n1
k_Luft=0 #imaginärteil von n1
n_Luft = n_strich_Luft + k_Luft * 1j #Brechungsindex Beschichtung

n_strich_Schicht=1.9754 ##realteil von n2
k_Schicht= 0.88181 ##imaginärteil von n2
n_Schicht = n_strich_Schicht + k_Schicht * 1j #Brechungsindex Beschichtung

n_strich_Substrat=1.5449 #realteil von n3
k_Substrat= 0.35977 #imaginärteil von n3
n_Substrat = n_strich_Schicht + k_Schicht * 1j #Brechungsindex Substrat

einfallswinkel = np.radians(0) #Einfallswinkel in Grad zum Lot hin
#spezialfall -> Einfallswinkel =0

import cmath

def fresnel_coeffs_complex(n_strich_1, k1, n_strich_2, k2, n_strich_3, k3, theta1):

    # Berechne den komplexen Brechungsindex des Lichts im Medium 1
    n1 = complex(n_strich_1, k1)#k positiv oder negativ?
    print(n1)

    # Berechne den komplexen Brechungsindex des Lichts im Medium 2
    n2 = complex(n_strich_2, k2)
    print(n2)

    n3= complex(n_strich_3, k3)

    # Berechne den Einfallswinkel im Medium 2

    theta2 = np.arcsin((n1 / n2) * np.sin(theta1))
    print(np.degrees(theta2.real))

    # Berechne die Fresnel-Koeffizienten
    r_12 = (n2-n1)/(n2+n1)
    r_23 = (n3-n2)/(n3+n2)

    r_12 = (n1 - n2) / (n2 + n1)
    r_23 = (n2 - n3) / (n3 + n2)

    t_12 = (n2/n1)*(1-r_12) #wrong!!#wird eh nicht weiter in der Berchnung verwendet
    t_23 = (n3 / n2) * (1 - r_23)#wrong??#wird nicht wieter in der Berechnung verwendet

    return r_12, r_23, t_12, t_23,theta2



r_12, r_23, t_12, t_23, brechungswinkel = fresnel_coeffs_complex(n_strich_Luft, k_Luft, n_strich_Schicht, k_Schicht, n_strich_Substrat, k_Substrat, einfallswinkel)

# Ausgabe der Koeffizienten
print("r_12 =", r_12)
print("r_23 =", r_23)
print("t_12 =", t_12)
print("t_23 =", t_23)

r=r_12
#Amplituden-Reflexionskoeffizient beträgt folglich
lamda_Schicht=lamda/n_Schicht.real
k2=np.pi*2/lamda_Schicht

alpha=(2*np.pi*k_Schicht)/lamda_Schicht #Absorptionskonstante
print("alpha=", alpha)

#Initialisierung
r_ges_values=[]
R_values=[]
absorption_values=[]#lambert-beer
R_absorption_values=[]
#r_FP_values=[]
beta=k2-1j*alpha #für Schicht zwei (die Beschichtung)

# Berechnung von A_res für jeden Wert von d

#R_13=((n_Substrat.real-n_Luft.real)/(n_Substrat.real+n_Luft.real))**2
#print("r ohne Schicht =", ((n_Substrat.real-n_Luft.real)/(n_Substrat.real+n_Luft.real)))

for d in d_values:
    r_ges = r * (1 - np.exp(-2 * 1j * k2 * d)) / (1 - r ** 2 * np.exp(-2 * 1j * k2 * d))
    r_ges_values.append(r_ges.real)

    r_conj = r_ges.conjugate()
    R_squared = r_ges * r_conj


    absorption=np.exp(-2*alpha*d)#Abschwächung des Lichts nach lambert-beer

    #R=(4*r**2*np.sin(k2*d)**2)/((1-r**2)**2+4*r**2*np.sin(k2*d)**2)
    R_2=((r_12+r_23*absorption)**2-4*r_12*r_23*absorption*np.sin(k2*d)**2)/((1+r_12*r_23*absorption)**2-4*r_12*r_23*absorption*np.sin(k2*d)**2) #mit Absorption
    R_3=((r_12+r_23)**2-4*r_12*r_23*np.sin(k2*d)**2)/((1+r_12*r_23)**2-4*r_12*r_23*np.sin(k2*d)**2) #ohne Absorption nach lembert-beer
    absorption_values.append(absorption)
    R_values.append(R_3.real)
    R_absorption_values.append(R_2.real)
    #r_FP=(r_12+r_23*np.exp(-2*1j*beta*d))/(1+r_12*r_23*np.exp(-2*1j*d))
    #r_FP_values.append(r_FP.real)


'''# Plotten der Ergebnisse
plt.plot(d_values, R_absorption_values, label='R_absorption')
#plt.plot(d_values, R_values, label='R')
plt.xlabel('Schichtdicke d [nm]')
plt.ylabel('Reflektivität R oder auch Intensitätsreflexion')
#plt.ylabel('Amplitudenreflexionskoeffizient')
plt.legend()
plt.show()'''

#Phasenverschiebung berechnen, komplexer Reflexionskoeffizient:

#################
#Veranschaulichung für einen Wellenzug der auf das Schichtsystem trifft
#################

d=50 #nanometer # eine Dicke für die Beschichtung festlegen
absorption=np.exp(-2*alpha*d) #absorption nach lambert-beer

t_12_mal_t_21 = 1-r_12**2
r_21=-r_12

r_1=r_12#=E_r1/E_i
r_2= t_12_mal_t_21*r_23*np.exp(-2*1j*k2*d)#E_r2/E_i
r_3= r_21*r_23**2*t_12_mal_t_21*np.exp(-4*1j*k2*d)#E_r3/E_i

print("r_1 =", r_1)
print("r_2 =", r_2)
print("r_3 =", r_3)


phase_shift = cmath.phase(r_1)

print("Phasenverschiebung, Welle 1 zu Welle 0: ", np.degrees(r_1.real))#zu Welle 0
print("Phasenverschiebung, Welle 2 zu Welle 0: ", np.degrees(r_2.real))
print("Phasenverschiebung, Welle 3 zu Welle 0: ", np.degrees(r_3.real))

import numpy as np
import matplotlib.pyplot as plt

# Parameter für Welle 1
Amplitude_Welle0 = 1.0 #hier eine Zahl gewählt
Wellenlänge = lamda/1000  # Wellenlänge in Mikrometern
k = 2 * np.pi / Wellenlänge  # Wellenzahl berechnen

# Parameter für Reflexionskoeffizient
#r = -0.4 - 0.2j

# Anzahl der Perioden
Perioden = 1

# Ortarray für den Plot
Ort = np.linspace(0, Perioden * Wellenlänge, 1000)

# Welle 1
Welle0 = Amplitude_Welle0 * np.exp(1j * k * Ort)

# Welle 2
Welle1 = r_1 * Welle0
Welle2 = r_2 * Welle0
Welle2_absorption = r_2 * Welle0 * np.sqrt(absorption)
Welle3 = r_3 * Welle0
Welle3_absorption = r_3 * Welle0* absorption**2


#################plot


# Plot erstellen
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(Ort, np.real(Welle0), color='grey', label='Welle 0')
ax.plot(Ort, np.real(Welle1), color='darkblue', label='Welle 1')
ax.plot(Ort, np.real(Welle2), color='darkgreen', label='Welle 2')
ax.plot(Ort, np.real(Welle3), color='black', label='Welle 3')
ax.plot(Ort, np.real(Welle2_absorption), 'g:', label='Welle 2 mit Absorption über Schichtdicke')
ax.plot(Ort, np.real(Welle3_absorption), color='black', linestyle='dotted', label='Welle 3 mit Absorption über Schichtdicke')

ax.tick_params(axis='both', direction='in', length=5, labelleft=True, labelright=False, labelsize=16)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

#ax.set_xlabel(u'wavenumber (cm$^{-1}$)', fontsize=16)
#ax.set_ylabel(u'emission (mW cm$^{-1}$)', fontsize=16)
ax.set_xlabel(u'Ort (\u00B5m)', fontsize=16)
ax.set_ylabel(u'Amplitude', fontsize=16)
#plt.legend(loc='lower center')

plt.show()


######neuer Plot

# Plot erstellen
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(d_values, R_absorption_values, color='black', label='R_absorption')
#plt.plot(d_values, R_absorption_values, label='R_absorption')


ax.tick_params(axis='both', direction='in', length=5, labelleft=True, labelright=False, labelsize=16)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

#ax.set_xlabel(u'wavenumber (cm$^{-1}$)', fontsize=16)
#ax.set_ylabel(u'emission (mW cm$^{-1}$)', fontsize=16)
ax.set_xlabel(u'Beschichtungsdicke (nm)', fontsize=16)
ax.set_ylabel(u'Intensitätsreflexion', fontsize=16)
#plt.legend(loc='lower center')

plt.show()