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
    acquisitiontime = 15 #.5 # seconds
    #messung1 = [] #Detektor 1, Messung1 Transparent
    #messung2 = [] #Detektor 1, Messung2,Transparent
    messung3 = [] #Detektor 2, Messung 1, Filter
    messung4 = [] #Detektor 2, Messung 2, Filter
    ticker = []

    def makeFig():  # Create a function that makes our desired plot
        # plt.ylim(80, 90)  # Set y min and max values
        #plt.xlim(0,0.2)
        plt.figure(figsize=(8, 6))  # 8 Zoll breit und 6 Zoll hoch
        plt.title('My Streaming Sensor Data')  # Plot the title
        plt.grid(True)  # Turn the grid on
        plt.ylabel('Spannung in Volt')  # Set ylabels
        plt.xlabel('Zeit in Sekunden')  # Set ylabels
        #plt.plot(ticker, messung1, 'blue', label='Detektor1, Messung1')  # plot messung1 #Detektor 1 ist der mit dem transparenten Filter
        #plt.plot(ticker, messung2, 'green', label='Detektor1, Messung2')  # plot messung2
        #plt.plot(ticker, (np.array(messung4)-np.array(messung3)))
        #plt.plot(ticker, (np.array(messung2) - np.array(messung1)))
        plt.plot(ticker, messung3, 'red', label='Detektor2, Messung1')  # plot messung2
        #plt.plot(ticker, messung4, 'purple', label='Detektor2, Messung2')  # plot messung2
        plt.plot(ticker, messungDet1_flattop, 'blue', label= 'Messung1*flattop')
        plt.plot(ticker, messungDet1_flattop_nuttall, 'orange', label='Messung1*flattop*nuttall')
        #plt.plot(ticker, messungDet2, 'green', label= 'Messung2 * window')
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

        #m1 = float(dataArray[0])
        #m2 = float(dataArray[1])
        m3 = float(dataArray[2])
        m4 = float(dataArray[3])
        T = time.perf_counter()

        #messung1.append(m1)
        #messung2.append(m2)
        messung3.append(m3)
        messung4.append(m4)
        readings = T - t_start
        ticker.append(readings) #ticker entspricht run in Benedikts file

    #makeFig()


    acquisitiontime_measured = T - t_start #Erfassungszeit
    print('Messdauer:\t\t', "%.2f" % acquisitiontime_measured, ' s')
    ser.write(('2').encode('ascii'))

    #######
    # FFT #
    #######

    #string_time = ', '.join(map(str, ticker))
    ticker=np.array(ticker)
    #print(ticker)

    print('fft start')

    samples = ticker.shape[0] #Anzahl an Abtastwerten
    print(f'sample = {samples}')
    #samplerate = acquisitiontime_measured / samples
    #print(f'samplerate= {samplerate}')
    samplerate = acquisitiontime_measured / samples  # Dauer einer Messung in Sekunden

    ##Frequenzen
    freqs = np.fft.fftfreq(samples, samplerate) #(??, was passiert hier?), was ist der Unterschied zu timearray?? in sekunden
    idx = np.argsort(freqs)  # Frequenzen sortieren
    freqs_slice = freqs[idx][int((samples+3)/ 2):]  # nur positive Frequenzen verwenden -> (samples+3)/ 2)

    ##Amplituden & Apply Flat Top window to the signal

    messung3 = np.array(messung3)
    messung4 =np.array(messung4)

    window1 = windows.flattop(len(ticker))
    #window2 = signal.blackmanharris(samples)
    window2= signal.windows.nuttall(len(ticker))

    #messungDet1 = (np.array(messung2) - np.array(messung1)) #Detektor 1, transparenter Filter, background
    #messungDet2 = (np.array(messung4) - np.array(messung3))
    #messungDet1 = messung3*window1*window2
    #messungDet2 = messung4*window1*window2

    messungDet1 = messung3
    messungDet1_flattop =messung3*window1
    messungDet1_flattop_nuttall=messung3*window1*window2
    messungDet2 = messung4*window1
    #messungDet1_test=messung3*window1*window2

    makeFig()

    #Amplitude Detektor 2, Channel 3, ohne window
    n =8

    amplitudeDet1=(1/samples)* np.abs(np.fft.fft(messungDet1))#*2 #Fast Fourier Transformation von Messung 3, /samplerate mitnehmen #was für eine Einheit kommt bei der fft raus. Spannung pro Zeitintervall
    amplitudeDet1_slice = amplitudeDet1[idx][int((samples +3)/ 2):]  # +11 (ungerade!), um Maximum bei f=0 auszuschließen
    #print(amplitudeDet1_slice)
    amplitudeDet1_slice=amplitudeDet1_slice[n:]
    freqs_slice=freqs_slice[n:]

    #print(amplitudeDet1_slice)
    #print(freqs_slice)

    maxvalueDet1 = np.amax(amplitudeDet1_slice)
    maxindexDet1 = np.argmax(amplitudeDet1_slice)
    dominantfreqDet1 = freqs_slice[maxindexDet1]

    # Amplitude Detektor 2, Channel 3, flattop window

    amplitudeDet1_flattop = (1 / samples) * np.abs(np.fft.fft(messungDet1_flattop))  # *2 #Fast Fourier Transformation von Messung 3, /samplerate mitnehmen #was für eine Einheit kommt bei der fft raus. Spannung pro Zeitintervall
    amplitudeDet1_slice_flattop = amplitudeDet1_flattop[idx][int((samples + 3) / 2):]  # +11 (ungerade!), um Maximum bei f=0 auszuschließen
    # print(amplitudeDet1_slice)
    amplitudeDet1_slice_flattop = amplitudeDet1_slice_flattop[n:]
    #freqs_slice = freqs_slice[n:]

    maxvalueDet1_flattop = np.amax(amplitudeDet1_slice_flattop)
    maxindexDet1_flattop = np.argmax(amplitudeDet1_slice_flattop)
    dominantfreqDet1_flattop = freqs_slice[maxindexDet1_flattop]

    # Amplitude Detektor 2, Channel 3, flattop window und nuttall

    amplitudeDet1_flattop_nuttall = (1 / samples) * np.abs(np.fft.fft(messungDet1_flattop_nuttall))  # *2 #Fast Fourier Transformation von Messung 3, /samplerate mitnehmen #was für eine Einheit kommt bei der fft raus. Spannung pro Zeitintervall
    amplitudeDet1_slice_flattop_nuttall = amplitudeDet1_flattop_nuttall[idx][int((samples + 3) / 2):]  # +11 (ungerade!), um Maximum bei f=0 auszuschließen
    # print(amplitudeDet1_slice)
    amplitudeDet1_slice_flattop_nuttall = amplitudeDet1_slice_flattop_nuttall[n:]
    #freqs_slice = freqs_slice[n:]

    maxvalueDet1_flattop_nuttall = np.amax(amplitudeDet1_slice_flattop_nuttall)
    maxindexDet1_flattop_nuttall = np.argmax(amplitudeDet1_slice_flattop_nuttall)
    dominantfreqDet1_flattop = freqs_slice[maxindexDet1_flattop_nuttall]

    #Amplitude Detektor 2, Channel 4

    amplitudeDet2 = (1 / samples) * np.abs(np.fft.fft(messungDet2))#*2
    amplitudeDet2_slice = amplitudeDet2[idx][int((samples +3)/2):]
    amplitudeDet2_slice=amplitudeDet2_slice[n:]

    maxvalueDet2 = np.amax(amplitudeDet2_slice)
    maxindexDet2 = np.argmax(amplitudeDet2_slice)
    dominantfreqDet2 = freqs_slice[maxindexDet2]

    #plt.plot(freqs,amplitudeDet1)
    #plt.show()

    print(f'Maxvalue Detektor2, channel 1, ohne window = {maxvalueDet1}')
    print(f'Maxvalue Detektor2, channel 1, flattop window = {maxvalueDet1_flattop}')
    print(f'Maxvalue Detektor2, channel 1, flattop und nutall window = {maxvalueDet1_flattop_nuttall}')
    print(f'Maxvalue Detektor2, channel 2 = {maxvalueDet2}')
    print(f'Maxvalue Detektor2, Channel 1 und 2= {maxvalueDet1 + maxvalueDet2}')
    print(u'Verhältnis:\t\t', maxvalueDet2 / maxvalueDet1)
    print(u'Frequenz1 (FFT):\t\t', "%.2f" % dominantfreqDet1, ' Hz')
    print(u'Frequenz2 (FFT):\t\t', "%.2f" % dominantfreqDet2, ' Hz')

    plt.figure(figsize=(8.2, 6.2))
    plt.xlim(0, 25)
    plt.plot(freqs_slice, amplitudeDet1_slice, color='red')
    plt.plot(dominantfreqDet1, maxvalueDet1, color='red', marker='*')
    plt.plot(freqs_slice, amplitudeDet1_slice_flattop, color='blue')
    plt.plot(dominantfreqDet1, maxvalueDet1_flattop, color='blue', marker='*')
    plt.plot(freqs_slice, amplitudeDet1_slice_flattop_nuttall, color='orange')
    plt.plot(dominantfreqDet1, maxvalueDet1_flattop_nuttall, color='orange', marker='*')
    #plt.plot(freqs_slice, amplitudeDet2_slice, color='green')
    #plt.plot(dominantfreqDet2, maxvalueDet2, color='green', marker='*')


    plt.title(f' Messung mit KompaktsensorV1, {dicke} nm AlOx Beschichtung')
    plt.xlabel(u'Frequenz (Hz)')
    plt.ylabel(u'Amplitude');

    #plt.savefig(f'Kompsens_{dicke}AlOx_Messung{NummerMessung}_{Kommentar}')
    plt.show()

    ################# EXCEL EINLESEN ###########################


    d = {'Messnummer': [NummerMessung],'Maxvalue ohne window':[maxvalueDet1], 'Maxvalue flattop': [maxvalueDet1_flattop], 'Maxvalue flattop nuttall': [maxvalueDet1_flattop_nuttall], 'Acquisitiontime': [acquisitiontime_measured], 'Kommentar': [Kommentar]}
    #d={'Messung3': [messung3], 'Messung4': [messung4], 'Zeit': ticker, 'MaxValue,Det2, channel3': [maxvalueDet1], 'Maxvalue, Det2, channel4':[maxvalueDet2] }
    #d = {'Messnummer': [NummerMessung],'dicke':[dicke], 'maxValueChan11': [maxvalueDet1], 'maxValueChan2': [maxvalueDet2], 'maxValueChan1u2': [maxvalueDet1+maxvalueDet2], 'Frequenz': [dominantfreqDet1], 'Kommentar': [Kommentar]}


    df1 = pd.DataFrame(data=d)

    #file_name = (file_name)

    with pd.ExcelWriter(file_name, engine="openpyxl", mode='a', if_sheet_exists='overlay') as writer:
        df1.to_excel(writer, header=False, startrow=NummerMessung, startcol=0, merge_cells=True)

    print('excel export successful')
