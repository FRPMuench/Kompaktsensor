def auswerte(dicke, NummerMessung, Kommentar, file_name):
    import serial
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.signal import windows
    import scipy.signal as signal
    from numpy.fft import fft, ifft


    ser = serial.Serial('COM3', 115200, timeout=1)#, timeout=0.01)#None)#0.01)

    ##Einstellungen
    acquisitiontime = 4 #seconds
    messung1 = [] #Detektor 1, Messung1 Transparent
    messung2 = [] #Detektor 1, Messung2,Transparent
    messung3 = [] #Detektor 2, Messung 1, Filter
    messung4 = [] #Detektor 2, Messung 2, Filter
    ticker = []

    def makeFig():  # Create a function that makes our desired plot
        # plt.ylim(80, 90)  # Set y min and max values
        plt.xlim(0,0.6)
        #plt.ylim(2.35, 2.45)
        #plt.figure(figsize=(8, 6))  # 8 Zoll breit und 6 Zoll hoch
        plt.title('My Streaming Sensor Data')  # Plot the title
        plt.grid(True)  # Turn the grid on
        plt.ylabel('Spannung in Volt')  # Set ylabels
        plt.xlabel('Zeit in Sekunden')  # Set ylabels
        plt.plot(ticker, messung1, 'blue', label='Detektor1, Messung1')  # plot messung1 #Detektor 1 ist der mit dem transparenten Filter
        plt.plot(ticker, messung2, 'green', label='Detektor1, Messung2')  # plot messung2
        #plt.plot(ticker, (np.array(messung2) - np.array(messung1)))
        plt.plot(ticker, messung3, 'red', label='Detektor2, channel 1')  # plot messung2
        plt.plot(ticker, messung4, 'purple', label='Detektor2, channel 2')  # plot messung2
        # plt.plot(ticker, (np.array(messung4)-np.array(messung3)))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.savefig('Spannung_Zeit_Diagramm_Hintergrund_Luft')
        plt.show()

    ##Einlesen

    T0 = time.perf_counter()  # Zeit bis Messund startet (Kommunikationsdauer)

    ser.write(('1').encode('ascii'))

    t_start = time.perf_counter()
    print('Startzeit')
    print(t_start)
    t_end = t_start + acquisitiontime
    print('Endzeit')
    print(t_end)
    T = t_start

    while T < t_end:
        arduinoString = ser.readline()  # read the line of text from the serial port
        dataArray = arduinoString.split(b' ')  # Split it into an array called dataArray

        m1 = float(dataArray[0])
        m2 = float(dataArray[1])
        m3 = float(dataArray[2])
        m4 = float(dataArray[3])
        T = time.perf_counter()

        messung1.append(m1)
        messung2.append(m2)
        messung3.append(m3)
        messung4.append(m4)
        readings = T - t_start
        ticker.append(readings) #ticker entspricht run in Benedikts file

    acquisitiontime_measured = T - t_start #Erfassungszeit
    print('Messdauer:\t\t', "%.2f" % acquisitiontime_measured, ' s')
    ser.write(('2').encode('ascii'))

    #######
    # FFT #
    #######

    ticker=np.array(ticker)

    print('fft start')

    samples = ticker.shape[0] #Anzahl an Abtastwerten
    print(f'sample = {samples}')
    samplerate = acquisitiontime_measured / samples  # Dauer einer Messung in Sekunden

    ##Frequenzen
    freqs = np.fft.fftfreq(samples, samplerate) #(??, was passiert hier?), was ist der Unterschied zu timearray?? in sekunden
    idx = np.argsort(freqs)  # Frequenzen sortieren
    freqs_slice = freqs[idx][int((samples+3)/ 2):]  # nur positive Frequenzen verwenden -> (samples+3)/ 2)

    ##Amplituden & Apply Flat Top window to the signal

    messung1 = np.array(messung1)
    messung2 = np.array(messung2)
    messung3 = np.array(messung3)
    messung4 = np.array(messung4)

    window1 = windows.flattop(len(ticker))
    window2= signal.windows.nuttall(len(ticker))
    #window2 = 1

    #messungDet1 = (np.array(messung2) - np.array(messung1)) #Detektor 1, transparenter Filter, background
    #messungDet2 = (np.array(messung4) - np.array(messung3))
    #messungDet1 = messung3*window1*window2
    #messungDet2 = messung4*window1*window2


    #messungDet1_chan1 = messung1*window1
    #messungDet1_chan2 = messung2*window1

    messungDet1_chan1 = messung1*window1*window2
    messungDet1_chan2 = messung2*window1*window2
    messungDet2_chan1 = messung3*window1*window2
    messungDet2_chan2 = messung4*window1*window2

    makeFig()

    n = 8  # messarray wird abgeschnitten, damit maximum bei null rausfliegt
    freqs_slice = freqs_slice[n:]

    ######
    # Amplitude Detektor 1, Channel 1 (messung1), mit flattop window
    ######

    amplitudeDet1_chan1 = (1 / samples) * np.abs(np.fft.fft(messungDet1_chan1))  # *2 #Fast Fourier Transformation von Messung 3, /samplerate mitnehmen #was für eine Einheit kommt bei der fft raus. Spannung pro Zeitintervall
    amplitudeDet1_chan1_slice = amplitudeDet1_chan1[idx][int((samples + 3) / 2):]  # +11 (ungerade!), um Maximum bei f=0 auszuschließen
    amplitudeDet1_chan1_slice = amplitudeDet1_chan1_slice[n:]

    maxvalueDet1_chan1 = np.amax(amplitudeDet1_chan1_slice)
    maxindexDet1_chan1 = np.argmax(amplitudeDet1_chan1_slice)
    dominantfreqDet1_chan1 = freqs_slice[maxindexDet1_chan1]

    ######
    # Amplitude Detektor 1, Channel 2 (messung2), mit flattop window
    ######

    amplitudeDet1_chan2 = (1 / samples) * np.abs(np.fft.fft(messungDet1_chan2))  # *2 #Fast Fourier Transformation von Messung 3, /samplerate mitnehmen #was für eine Einheit kommt bei der fft raus. Spannung pro Zeitintervall
    amplitudeDet1_chan2_slice = amplitudeDet1_chan2[idx][int((samples + 3) / 2):]  # +11 (ungerade!), um Maximum bei f=0 auszuschließen
    amplitudeDet1_chan2_slice = amplitudeDet1_chan2_slice[n:]

    maxvalueDet1_chan2 = np.amax(amplitudeDet1_chan2_slice)
    maxindexDet1_chan2 = np.argmax(amplitudeDet1_chan2_slice)
    dominantfreqDet1_chan2 = freqs_slice[maxindexDet1_chan2]

    ######
    #Amplitude Detektor 2, Channel 1 (messung3), mit flattop window
    ######

    amplitudeDet2_chan1=(1/samples)* np.abs(np.fft.fft(messungDet2_chan1))#*2 #Fast Fourier Transformation von Messung 3, /samplerate mitnehmen #was für eine Einheit kommt bei der fft raus. Spannung pro Zeitintervall
    amplitudeDet2_chan1_slice = amplitudeDet2_chan1[idx][int((samples +3)/ 2):]  # +11 (ungerade!), um Maximum bei f=0 auszuschließen
    amplitudeDet2_chan1_slice=amplitudeDet2_chan1_slice[n:]


    maxvalueDet2_chan1 = np.amax(amplitudeDet2_chan1_slice)
    maxindexDet2_chan1 = np.argmax(amplitudeDet2_chan1_slice)
    dominantfreqDet2_chan1 = freqs_slice[maxindexDet2_chan1]

    ######
    # Amplitude Detektor 2, Channel 2 (messung4), mit flattop window
    ######

    amplitudeDet2_chan2 = (1 / samples) * np.abs(np.fft.fft(messungDet2_chan2))  # *2 #Fast Fourier Transformation von Messung 3, /samplerate mitnehmen #was für eine Einheit kommt bei der fft raus. Spannung pro Zeitintervall
    amplitudeDet2_chan2_slice = amplitudeDet2_chan2[idx][int((samples + 3) / 2):]  # +11 (ungerade!), um Maximum bei f=0 auszuschließen
    amplitudeDet2_chan2_slice = amplitudeDet2_chan2_slice[n:]

    maxvalueDet2_chan2 = np.amax(amplitudeDet2_chan2_slice)
    maxindexDet2_chan2 = np.argmax(amplitudeDet2_chan2_slice)
    dominantfreqDet2_chan2 = freqs_slice[maxindexDet2_chan2]

    ##Ausgabewerte###

    print(f'Maxvalue Detektor1, channel 1, (messung1) mit flattop window = {maxvalueDet1_chan1}')
    print(f'Maxvalue Detektor1, channel 2, (messung2) mit flattop window = {maxvalueDet1_chan2}')
    print(f'Maxvalue Detektor1, Channel 1 und 2= {maxvalueDet1_chan1 + maxvalueDet1_chan2}')

    print(f'Maxvalue Detektor2, channel 1, (messung3) mit flattop window = {maxvalueDet2_chan1}')
    print(f'Maxvalue Detektor2, channel 2, (messung4) mit flattop window = {maxvalueDet2_chan2}')
    print(f'Maxvalue Detektor2, Channel 1 und 2= {maxvalueDet2_chan1 + maxvalueDet2_chan2}')
    print(f'Verhältnis max Value Detektor 2, chan2  durch Detektor 1, chan2 = {(maxvalueDet2_chan2) / (maxvalueDet1_chan2)}')

    print(u'Frequenz Detektor1, chan1 (FFT):\t\t', "%.2f" % dominantfreqDet1_chan1, ' Hz')
    print(u'Frequenz Detektor2, chan2 (FFT):\t\t', "%.2f" % dominantfreqDet1_chan2, ' Hz')
    print(u'Frequenz Detektor2, chan1 (FFT):\t\t', "%.2f" % dominantfreqDet2_chan1, ' Hz')
    print(u'Frequenz Detektor2, chan2 (FFT):\t\t', "%.2f" % dominantfreqDet2_chan2, ' Hz')

    ##plot##

    plt.figure(figsize=(8.2, 6.2))
    plt.xlim(0, 25)

    plt.plot(freqs_slice, amplitudeDet1_chan1_slice, color='blue', label='Detektor 1, channel 1')
    plt.plot(dominantfreqDet1_chan1, maxvalueDet1_chan1, color='blue', marker='*')
    plt.plot(freqs_slice, amplitudeDet1_chan2_slice, color='green', label='Detektor 1, channel 2')
    plt.plot(dominantfreqDet1_chan2, maxvalueDet1_chan2, color='green', marker='*')

    plt.plot(freqs_slice, amplitudeDet2_chan1_slice, color='red', label='Detektor 2, channel 1')
    plt.plot(dominantfreqDet2_chan1, maxvalueDet2_chan1, color='red', marker='*')
    plt.plot(freqs_slice, amplitudeDet2_chan2_slice, color='purple', label='Detektor 2, channel 2')
    plt.plot(dominantfreqDet2_chan2, maxvalueDet2_chan2, color='purple', marker='*')

    plt.title(f' Messung mit KompaktsensorV1, {dicke} nm AlOx Beschichtung')
    plt.xlabel(u'Frequenz (Hz)')
    plt.ylabel(u'Amplitude');
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

    ################# EXCEL EINLESEN ###########################

    d = {'Messnummer': [NummerMessung],'dicke':[dicke], 'maxValueDet1_chan1': [maxvalueDet1_chan1], 'maxValueDet1_chan2': [maxvalueDet1_chan2], 'maxValueDet2_chan1': [maxvalueDet2_chan1], 'maxValueDet2_chan2': [maxvalueDet2_chan2], 'Verhaeltnis2zu1': [maxvalueDet2_chan1/maxvalueDet1_chan1], 'Frequenz': [dominantfreqDet2_chan1], 'Kommentar': [Kommentar]}

    df1 = pd.DataFrame(data=d)

    with pd.ExcelWriter(file_name, engine="openpyxl", mode='a', if_sheet_exists='overlay') as writer:
        df1.to_excel(writer, header=False, startrow=NummerMessung, startcol=0, merge_cells=True)

    print('excel export successful')
