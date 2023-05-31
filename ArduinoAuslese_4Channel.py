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
    acquisitiontime = 6 #seconds
    messung1 = [] #chan 1, Arduino A0
    messung2 = [] # chan 2, Arduino A1
    messung3 = [] # chan 3, Arduino A2
    messung4 = [] # chan 4, Arduino A3
    ticker = []

    def makeFig():  # Create a function that makes our desired plot
        # plt.ylim(80, 90)  # Set y min and max values
        #plt.xlim(0,0.6)
        #plt.ylim(2.35, 2.45)
        #plt.figure(figsize=(8, 6))  # 8 Zoll breit und 6 Zoll hoch
        plt.title('My Streaming Sensor Data')  # Plot the title
        plt.grid(True)  # Turn the grid on
        plt.ylabel('Spannung in Volt')  # Set ylabels
        plt.xlabel('Zeit in Sekunden')  # Set ylabels
        plt.plot(ticker, messung1, 'blue', label='Chan1, 3.95 mü/ 90 nm Ref')  # plot messung1 #Detektor 1 ist der mit dem transparenten Filter
        #plt.plot(ticker, messung2, 'green', label='Chan2, 7.30 mü/ 200 nm SO2')  # plot messung2
        #plt.plot(ticker, messung3, 'red', label='Chan3, 9.44 mü/ 460 nm')  # plot messung2
        plt.plot(ticker, messung4, 'purple', label='Chan4, 12.28 mü/ 1000 nm (for AlOx)')  # plot messung2
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
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

    multiplier = (3.3 / 1024)  # Der Wert, mit dem jedes Array-Element multipliziert werden soll#3.3V für nano und 5V für UNO

    messung1 *= multiplier
    messung2 *= multiplier
    messung3 *= multiplier
    messung4 *= multiplier

    window1 = windows.flattop(len(ticker))
    window2= signal.windows.nuttall(len(ticker))

    messung_chan1 = messung1*window1
    messung_chan2 = messung2*window1
    messung_chan3 = messung3*window1
    messung_chan4 = messung4*window1

    makeFig()

    n = 8  # messarray wird abgeschnitten, damit maximum bei null rausfliegt
    freqs_slice = freqs_slice[n:]

    ######
    # Amplitude Channel 1 (messung1), mit flattop window und nuttall
    ######

    amplitude_chan1 = (1 / samples) * np.abs(np.fft.fft(messung_chan1))  # *2 #Fast Fourier Transformation von Messung 3, /samplerate mitnehmen #was für eine Einheit kommt bei der fft raus. Spannung pro Zeitintervall
    amplitude_chan1_slice = amplitude_chan1[idx][int((samples + 3) / 2):]  # +11 (ungerade!), um Maximum bei f=0 auszuschließen
    amplitude_chan1_slice = amplitude_chan1_slice[n:]

    maxvalue_chan1 = np.amax(amplitude_chan1_slice)
    maxindex_chan1 = np.argmax(amplitude_chan1_slice)
    dominantfreq_chan1 = freqs_slice[maxindex_chan1]

    ######
    # Amplitude Channel 2 (messung2), mit flattop window und nuttall
    ######

    amplitude_chan2 = (1 / samples) * np.abs(np.fft.fft(messung_chan2))  # *2 #Fast Fourier Transformation von Messung 3, /samplerate mitnehmen #was für eine Einheit kommt bei der fft raus. Spannung pro Zeitintervall
    amplitude_chan2_slice = amplitude_chan2[idx][int((samples + 3) / 2):]  # +11 (ungerade!), um Maximum bei f=0 auszuschließen
    amplitude_chan2_slice = amplitude_chan2_slice[n:]

    maxvalue_chan2 = np.amax(amplitude_chan2_slice)
    maxindex_chan2 = np.argmax(amplitude_chan2_slice)
    dominantfreq_chan2 = freqs_slice[maxindex_chan2]

    ######
    #Amplitude Channel 3 (messung3), mit flattop window und nuttall
    ######

    amplitude_chan3=(1/samples)* np.abs(np.fft.fft(messung_chan3))#*2 #Fast Fourier Transformation von Messung 3, /samplerate mitnehmen #was für eine Einheit kommt bei der fft raus. Spannung pro Zeitintervall
    amplitude_chan3_slice = amplitude_chan3[idx][int((samples +3)/ 2):]  # +11 (ungerade!), um Maximum bei f=0 auszuschließen
    amplitude_chan3_slice=amplitude_chan3_slice[n:]


    maxvalue_chan3 = np.amax(amplitude_chan3_slice)
    maxindex_chan3 = np.argmax(amplitude_chan3_slice)
    dominantfreq_chan3 = freqs_slice[maxindex_chan3]

    ######
    # Amplitude Channel 4 (messung4), mit flattop window und nuttall
    ######

    amplitude_chan4 = (1 / samples) * np.abs(np.fft.fft(messung_chan4))  # *2 #Fast Fourier Transformation von Messung 3, /samplerate mitnehmen #was für eine Einheit kommt bei der fft raus. Spannung pro Zeitintervall
    amplitude_chan4_slice = amplitude_chan4[idx][int((samples + 3) / 2):]  # +11 (ungerade!), um Maximum bei f=0 auszuschließen
    amplitude_chan4_slice = amplitude_chan4_slice[n:]

    maxvalue_chan4 = np.amax(amplitude_chan4_slice)
    maxindex_chan4 = np.argmax(amplitude_chan4_slice)
    dominantfreq_chan4 = freqs_slice[maxindex_chan4]

    ##Ausgabewerte###

    print(f'Maxvalue channel 1, (messung1) mit flattop und nuttall window = {maxvalue_chan1}')
    print(f'Maxvalue channel 2, (messung2) mit flattop und nuttall window = {maxvalue_chan2}')
    print(f'Maxvalue channel 3, (messung3) mit flattop und nutall window = {maxvalue_chan3}')
    print(f'Maxvalue channel 4, (messung4) mit flattop und nutall window = {maxvalue_chan4}')
    print(f'Maxvalue channel 4 durch maxValue chan1 (Ref) = {maxvalue_chan4/maxvalue_chan1}')

    print(u'Frequenz chan1 (FFT):\t\t', "%.2f" % dominantfreq_chan1, ' Hz')
    print(u'Frequenz chan2 (FFT):\t\t', "%.2f" % dominantfreq_chan2, ' Hz')
    print(u'Frequenz chan3 (FFT):\t\t', "%.2f" % dominantfreq_chan3, ' Hz')
    print(u'Frequenz chan4 (FFT):\t\t', "%.2f" % dominantfreq_chan4, ' Hz')

    ##plot##

    plt.figure(figsize=(8.2, 6.2))
    plt.xlim(0, 25)

    plt.plot(freqs_slice, amplitude_chan1_slice, color='blue', label='FFT Chan1, 3.95 mü/ 90 nm Ref')
    plt.plot(dominantfreq_chan1, maxvalue_chan1, color='blue', marker='*')
    plt.plot(freqs_slice, amplitude_chan2_slice, color='green', label='FFT Chan2, 7.30 mü/ 200 nm SO2')
    plt.plot(dominantfreq_chan2, maxvalue_chan2, color='green', marker='*')

    plt.plot(freqs_slice, amplitude_chan3_slice, color='red', label='FFT Chan3, 9.44 mü/ 460 nm')
    plt.plot(dominantfreq_chan3, maxvalue_chan3, color='red', marker='*')
    plt.plot(freqs_slice, amplitude_chan4_slice, color='purple', label='FFT Chan4, 12.28 mü/ 1000 nm (AlOx)')
    plt.plot(dominantfreq_chan4, maxvalue_chan4, color='purple', marker='*')

    plt.title(f' Messung mit 4 Channel Sensor, {dicke} nm AlOx Beschichtung')
    plt.xlabel(u'Frequenz (Hz)')
    plt.ylabel(u'Amplitude');
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

    ################# EXCEL EINLESEN ###########################

    d = {'Messnummer': [NummerMessung],'dicke':[dicke], 'maxValue_chan1': [maxvalue_chan1], 'maxValue_chan2': [maxvalue_chan2], 'maxValue_chan3': [maxvalue_chan3], 'maxValue_chan4': [maxvalue_chan4], 'Frequenz': [dominantfreq_chan4], 'Kommentar': [Kommentar]}

    df1 = pd.DataFrame(data=d)

    with pd.ExcelWriter(file_name, engine="openpyxl", mode='a', if_sheet_exists='overlay') as writer:
        df1.to_excel(writer, header=False, startrow=NummerMessung, startcol=0, merge_cells=True)

    print('excel export successful')
