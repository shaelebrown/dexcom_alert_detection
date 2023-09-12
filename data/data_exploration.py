
# script for exploring data for
# dexcom alert detection task

# first load libraries
import pydub
from pydub import AudioSegment
from pydub.playback import play
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import scipy.io.wavfile
from scipy.io.wavfile import read
from scipy import signal
import matplotlib.pyplot as plt

# background audio data for this project was downloaded from
# https://github.com/karolpiczak/ESC-50#download

# descom alerts were collected personally

# play some example sounds
typing = AudioSegment.from_wav("data/ESC-50-master/audio/1-137-A-32.wav")
print(len(typing)) # 5000 = 5 sec
play(typing)

# read in dexcom alert recordings and play
clear_alerts = AudioSegment.from_file('data/alerts/clear_alerts.m4a')
unclear_alerts = AudioSegment.from_file('data/alerts/under_blanket_alerts.m4a')
play(clear_alerts)
play(unclear_alerts)

# get individual alert recordings
clear_high = clear_alerts[1000:4000]
clear_low = clear_alerts[7650:9650]
clear_urgent_low = clear_alerts[14500:17000]
unclear_high = unclear_alerts[700:3700]
unclear_low = unclear_alerts[6000:8000]
unclear_urgent_low = unclear_alerts[11500:14000]

# overlay alert over typing audio
start_time = random.sample(range(len(typing) - len(clear_low)),k = 1)
typing_alert = typing.overlay(clear_low, position = start_time[0])
play(typing_alert)

# function to compute spectrogram from audio file, 
# potentially overlaying alert sound
def spectrogram(fname, alert = None):
    temp = AudioSegment.from_file(fname)
    temp = temp.set_frame_rate(48000)
    if alert != None:
        start_time = random.sample(range(len(temp) - len(globals()[alert])),k = 1)
        temp = temp.overlay(globals()[alert], position = start_time[0])
    fname = "data/ESC-50-master/audio/temp.wav" 
    temp.export(fname,format = "wav")
    sr_value, x_value = scipy.io.wavfile.read(fname)
    _, _, Sxx= signal.spectrogram(x_value,sr_value)
    Sxx = Sxx.swapaxes(0,1)
    Sxx = np.expand_dims(Sxx, axis = 0)
    os.remove(fname)
    return Sxx

# create two sample spectrograms and plot
no_alert_spect = spectrogram(fname = "data/ESC-50-master/audio/1-137-A-32.wav", alert = None)
alert_spect = spectrogram(fname = "data/ESC-50-master/audio/1-137-A-32.wav", alert = 'clear_low')
urgent_alert_spect = spectrogram(fname = "data/ESC-50-master/audio/1-137-A-32.wav", alert = 'clear_urgent_low')
high_alert_spect = spectrogram(fname = "data/ESC-50-master/audio/1-137-A-32.wav", alert = 'clear_high')
t = 0.0026666666666666666 + 0.004666666666666666*np.arange(1071) # consistent across files
f = 187.5*np.arange(129) # consistent across all files

plt.pcolormesh(t, f, no_alert_spect[0,:,:].T)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('No Alert Spectrogram')
plt.show()

plt.pcolormesh(t, f, alert_spect[0,:,:].T) # you can see the alert in the frequency spectrum!
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Alert Spectrogram')
plt.show()
# alert in frequency domain for about 1.5s and 1600Hz (max 2600Hz)

plt.pcolormesh(t, f, urgent_alert_spect[0,:,:].T)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Alert Spectrogram')
plt.show()
# alert in frequency domain for about 1.8s and 500Hz (max 1900Hz)

plt.pcolormesh(t, f, high_alert_spect[0,:,:].T)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Alert Spectrogram')
plt.show()
# alert in frequency domain for about 2.9s and 1800Hz (max 2600Hz)

# therefore for conv networks we should have convolutions of
# width 9 (frequency) and height 321 (time)
# width 3 (frequency) and height 387 (time)
# width 11 (frequency) and height 623 (time)
# for clear and weak signals