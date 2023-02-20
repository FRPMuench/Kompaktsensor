def auswerte(dicke, NummerMessung, Kommentar):
    import serial
    import time
    import numpy as np
    import matplotlib.pyplot as plt


    ser = serial.Serial('COM3', 9600, timeout=0.1)#, timeout=0.01)#None)#0.01)
    #plt.ion()

    #dicke= 10 # Schichtdicke des zu messenden Bechers in nm
    #NummerMessung = 1 #Nummerierung der Messung


    ##Einstellungen
    acquisitiontime = 0.4 #.5 # seconds
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
        plt.plot(ticker, (np.array(messung4)-np.array(messung3)))
        #plt.plot(ticker, (np.array(messung2) - np.array(messung1)))
        plt.plot(ticker, messung3, 'red', label='Detektor2, Messung1')  # plot messung2
        plt.plot(ticker, messung4, 'purple', label='Detektor2, Messung2')  # plot messung2
        plt.legend()
        #plt.savefig('Spannung_Zeit_Diagramm_Hintergrund_Luft')

        plt.show()

    ##Einlesen

    #while (ser.inWaiting() > 0):  # empty buffer
     #   garbage = ser.read_until(b'stopped') + ser.read()
      #  print(garbage)

    T0 = time.perf_counter()  # Zeit bis Messund startet (Kommunikationsdauer)

    ser.write(('1').encode('ascii'))

    # run = []

    #while (ser.inWaiting() < 10): #hier kommt 1000 hin -> was sagt die 1000 aus??
       # time.sleep(.01)#0.01

    print('Kommunigationsdauer:\t', "%.2f" % (time.perf_counter() - T0), ' s')

    #if (ser.inWaiting() >= 10): #hier kommt 1000 hin

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
        # ser_bytes = ser.readline().decode("utf-8", errors='replace').split()

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
        #ticker.append(time.perf_counter())#bis hier hin verrückt


    makeFig()
    # plt.pause(.00001)  # Pause Briefly. Important to keep drawnow from crashing

    acquisitiontime_measured = T - t_start #Erfassungszeit
    print('Messdauer:\t\t', "%.2f" % acquisitiontime_measured, ' s')
    ser.write(('2').encode('ascii'))


        #while (ser.inWaiting() > 0):  # empty buffer
         #   garbage = ser.read_until(b'stopped') + ser.read()
            #print(garbage)
          #  print('next')

        #while (ser.inWaiting() > 0):  # empty buffer
         #   print('gefangen in while Schleife')
          #  garbage = ser.read_until(b'stopped') + ser.read()


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
    samplerate = acquisitiontime_measured / samples
    #print(f'samplerate= {samplerate}')
    samplerate = acquisitiontime_measured / samples  # Dauer einer Messung in Sekunden

    ##Frequenzen
    freqs = np.fft.fftfreq(samples, samplerate) #(??, was passiert hier?), was ist der Unterschied zu timearray?? in sekunden
    idx = np.argsort(freqs)  # Frequenzen sortieren
    #freqs_slice = freqs[idx]
    #freqs_slice = freqs[idx]
    #print(freqs_slice)
    freqs_slice = freqs[idx][int((samples + 3) / 2):]  # nur positive Frequenzen verwenden

    ##Amplituden

    messungDet1 = (np.array(messung2) - np.array(messung1)) #Detektor 1, transparenter Filter, background
    #messungDet2 = np.array(messung3) + np.array(messung4) #Detektor 2, grauer Filter
    messungDet2 = (np.array(messung4) - np.array(messung3))
    #messungDet2_test = np.array(messung3)
    #messungDet2_test_ges= (np.array(messung4) - np.array(messung3))

    #messungDet1 = messung1 + messung3  # Detektor 1, transparenter Filter, background
    #messungDet2 = messung3 + messung4  # Detektor 2, grauer Filter

    #Amplitude Detektor 1

    amplitudeDet1=(1/samples)* np.abs(np.fft.fft(messungDet1)) #Fast Fourier Transformation von Messung 3, /samplerate mitnehmen #was für eine Einheit kommt bei der fft raus. Spannung pro Zeitintervall
    amplitudeDet1_slice = amplitudeDet1[idx][int((samples + 3) / 2):]  # +11 (ungerade!), um Maximum bei f=0 auszuschließen

    maxvalueDet1 = np.amax(amplitudeDet1_slice)
    maxindexDet1 = np.argmax(amplitudeDet1_slice)
    dominantfreqDet1 = freqs_slice[maxindexDet1]

    #Amplitude Detektor 2

    amplitudeDet2 = (1 / samples) * np.abs(np.fft.fft(messungDet2))
    amplitudeDet2_slice = amplitudeDet2[idx][int((samples+3)/2):]
    #amplitudeDet2_slice = amplitudeDet2[idx][int((samples + 3) / 2):]


    maxvalueDet2 = np.amax(amplitudeDet2_slice)
    maxindexDet2 = np.argmax(amplitudeDet2_slice)
    dominantfreqDet2 = freqs_slice[maxindexDet2]

    print(f'Maxvalue Detektor1 = {maxvalueDet1}')
    print(f'Maxvalue Detektor2 = {maxvalueDet2}')
    print(u'Verhältnis:\t\t', maxvalueDet2 / maxvalueDet1)
    print(u'Frequenz (FFT):\t\t', "%.2f" % dominantfreqDet2, ' Hz')

    plt.figure(figsize=(8.2, 6.2))
    plt.xlim(0, 15)
    #plt.plot(freqs_slice, amplitudeDet1_slice, color='blue')
    #plt.plot(dominantfreqDet1, maxvalueDet1, color='blue', marker='*')
    plt.plot(freqs_slice, amplitudeDet2_slice, color='green')
    plt.plot(dominantfreqDet2, maxvalueDet2, color='green', marker='*')


    plt.title(f' Messung mit KompaktsensorV1, {dicke} nm AlOx Beschichtung')
    plt.xlabel(u'Frequenz (Hz)')
    plt.ylabel(u'Amplitude');

    #plt.savefig(f'Kompsens_{dicke}AlOx_Messung{NummerMessung}_{Kommentar}')
    plt.show()

    ################# EXCEL EINLESEN ###########################
    import pandas as pd

    # df1 = pd..append([['a', 'b'], ['c', 'd']], index=['1', '5'], columns=['1', '5'])

    d = {'Messnummer': [NummerMessung], 'maxValue': [maxvalueDet2], 'dicke': [dicke], 'Frequenz': [dominantfreqDet2], 'Kommentar': [Kommentar]}
    df1 = pd.DataFrame(data=d)

    #df1 = pd.DataFrame([["XYZ"]], columns=[NummerMessung, maxvalueDet2, dicke])
    file_name = ('160223_Kompaktsensor_v2_Bechermessung2.xlsx')

    with pd.ExcelWriter('160223_Kompaktsensor_v2_Bechermessung2.xlsx', engine="openpyxl", mode='a', if_sheet_exists='overlay') as writer:
        df1.to_excel(writer, header=False, startrow=NummerMessung, startcol=0, merge_cells=True)

    # df1.to_excel(file_name, header=False, startrow=reihe-1, startcol =0, merge_cells=True)
    print('excel export successful')

