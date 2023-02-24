def auswerte(dicke, NummerMessung, Kommentar, file_name):
    import serial
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd


    ser = serial.Serial('COM3', 115200, timeout=0.1)#, timeout=0.01)#None)#0.01)

    ##Einstellungen
    acquisitiontime =2 #.5 # seconds
    messung1 = [] #Detektor 1, Messung1 Transparent
    messung2 = [] #Detektor 1, Messung2,Transparent
    messung3 = [] #Detektor 2, Messung 1, Filter
    messung4 = [] #Detektor 2, Messung 2, Filter
    ticker = []

    def makeFig():  # Create a function that makes our desired plot
        # plt.ylim(80, 90)  # Set y min and max values
        #plt.xlim(0,0.1)
        plt.title('My Streaming Sensor Data')  # Plot the title
        plt.grid(True)  # Turn the grid on
        plt.ylabel('Spannung in Volt')  # Set ylabels
        plt.xlabel('Zeit in Sekunden')  # Set ylabels
        #plt.plot(ticker, messung1, 'blue', label='Detektor1, Messung1')  # plot messung1 #Detektor 1 ist der mit dem transparenten Filter
        #plt.plot(ticker, messung2, 'green', label='Detektor1, Messung2')  # plot messung2
        #plt.plot(ticker, (np.array(messung4)-np.array(messung3)))
        #plt.plot(ticker, (np.array(messung2) - np.array(messung1)))
        plt.plot(ticker, messung3, 'red', label='Detektor2, Messung1')  # plot messung2
        plt.plot(ticker, messung4, 'purple', label='Detektor2, Messung2')  # plot messung2
        #plt.legend()
        #plt.savefig('Spannung_Zeit_Diagramm_Hintergrund_Luft')
        plt.show()

    ##Einlesen

    T0 = time.perf_counter()  # Zeit bis Messund startet (Kommunikationsdauer)

    ser.write(('1').encode('ascii'))


    print('Kommunigationsdauer:\t', "%.2f" % (time.perf_counter() - T0), ' s')

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

    ##Anpassung der Kurve auf vollständigen Durchlauf
    first_value=messung3[0]
    last_index = None
    for i in range(len(messung3) - 1, -1, -1):
        if messung3[i] == first_value:
            last_index = i
            break

    #last_index = min(range(len(messung3)), key=lambda i: abs(messung3[i] - first_value))

    # Kürze den Array an der Stelle des letzten Index
    if last_index is not None:
        messung1 = messung1[:last_index + 1]
        messung2 = messung2[:last_index + 1]
        messung3 = messung3[:last_index + 1]
        messung4 = messung4[:last_index + 1]
        ticker=ticker[:last_index + 1]

    print(f'first Value {first_value}')
    print(f'index {last_index}')
    print(messung3)
    makeFig()
    # plt.pause(.00001)  # Pause Briefly. Important to keep drawnow from crashing

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
    from numpy.fft import fft, ifft
    #runT = np.array(ticker).T  # .tolist()
    #samples = len(runT[0])  # Anzahl an Messungen die in der aquisition-Zeit durchgeführt wurden, Anzahl an Abtastpunkten im Intervall
    #print(f'sample = {samples}')
    samples = ticker.shape[0]
    print(f'sample = {samples}')
    #samplerate = acquisitiontime_measured / samples
    #print(f'samplerate= {samplerate}')
    samplerate = acquisitiontime_measured / samples  # Dauer einer Messung in Sekunden

    ##Frequenzen
    freqs = np.fft.fftfreq(samples, samplerate) #(??, was passiert hier?), was ist der Unterschied zu timearray?? in sekunden
    idx = np.argsort(freqs)  # Frequenzen sortieren
    #freqs_slice = freqs[idx]
    #freqs_slice = freqs[idx]
    #print(freqs_slice)
    freqs_slice = freqs[idx][int((samples) // 2):]  # nur positive Frequenzen verwenden

    ##Amplituden

    messungDet1 = (np.array(messung2) - np.array(messung1)) #Detektor 1, transparenter Filter, background
    messungDet2 = (np.array(messung4) - np.array(messung3))
    
    #Amplitude Detektor 1

    amplitudeDet1=(1/samples)* np.abs(np.fft.fft(messungDet1)) #Fast Fourier Transformation von Messung 3, /samplerate mitnehmen #was für eine Einheit kommt bei der fft raus. Spannung pro Zeitintervall
    amplitudeDet1_slice = amplitudeDet1[idx][int((samples) // 2):]  # +11 (ungerade!), um Maximum bei f=0 auszuschließen

    maxvalueDet1 = np.amax(amplitudeDet1_slice)
    maxindexDet1 = np.argmax(amplitudeDet1_slice)
    dominantfreqDet1 = freqs_slice[maxindexDet1]

    #Amplitude Detektor 2

    amplitudeDet2 = (1 / samples) * np.abs(np.fft.fft(messungDet2))
    amplitudeDet2_slice = amplitudeDet2[idx][int((samples)//2):]
    #amplitudeDet2_slice = amplitudeDet2[idx][int((samples + 3) / 2):]


    maxvalueDet2 = np.amax(amplitudeDet2_slice)
    maxindexDet2 = np.argmax(amplitudeDet2_slice)
    dominantfreqDet2 = freqs_slice[maxindexDet2]

    print(f'Maxvalue Detektor1 = {maxvalueDet1}')
    print(f'Maxvalue Detektor2 = {maxvalueDet2}')
    print(u'Verhältnis:\t\t', maxvalueDet2 / maxvalueDet1)
    print(u'Frequenz1 (FFT):\t\t', "%.2f" % dominantfreqDet1, ' Hz')
    print(u'Frequenz2 (FFT):\t\t', "%.2f" % dominantfreqDet2, ' Hz')

    plt.figure(figsize=(8.2, 6.2))
    #plt.xlim(0, 15)
    plt.plot(freqs_slice, amplitudeDet1_slice, color='blue')
    plt.plot(dominantfreqDet1, maxvalueDet1, color='blue', marker='*')
    plt.plot(freqs_slice, amplitudeDet2_slice, color='green')
    plt.plot(dominantfreqDet2, maxvalueDet2, color='green', marker='*')


    plt.title(f' Messung mit KompaktsensorV1, {dicke} nm AlOx Beschichtung')
    plt.xlabel(u'Frequenz (Hz)')
    plt.ylabel(u'Amplitude');

    #plt.savefig(f'Kompsens_{dicke}AlOx_Messung{NummerMessung}_{Kommentar}')
    plt.show()

    ################# EXCEL EINLESEN ###########################

    d = {'Messnummer': [NummerMessung], 'maxValue': [maxvalueDet2], 'dicke': [dicke], 'Frequenz': [dominantfreqDet2], 'Kommentar': [Kommentar], 'Verhältnis': [maxvalueDet2 / maxvalueDet1]}
    df1 = pd.DataFrame(data=d)

    #file_name = (file_name)

    with pd.ExcelWriter(file_name, engine="openpyxl", mode='a', if_sheet_exists='overlay') as writer:
        df1.to_excel(writer, header=False, startrow=NummerMessung, startcol=0, merge_cells=True)

    print('excel export successful')

