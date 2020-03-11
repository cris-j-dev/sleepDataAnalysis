filePath = "/media/chan/DATA2/Dataset/NSRR/mesa/polysomnography/edfs"
fileName = "mesa-sleep-0001.csv"


import os
import numpy as np
def dataLoader(filename):
    data = np.loadtxt(filename, float, delimiter=',', skiprows=1)
    return data

data = dataLoader(os.path.join(filePath, fileName))
_data = np.reshape(data, (data.shape[1], data.shape[0]))
data1 = data[:, 0]
data2 = data[:, 1]
data3 = data[:, 2]
label = data[:, 3]

Hz = 256
window_size = 10 * Hz
step_size = 60 * Hz


eeg1Data = np.array([data1[0:window_size]])
eeg2Data = np.array([data2[0:window_size]])
eeg3Data = np.array([data3[0:window_size]])
labelData = np.array([label[0:window_size]])
# ftData = np.array([eeg1Data, eeg2Data, eeg3Data, labelData])
for i in range(step_size, len(data), step_size):
    eeg1Data = np.append(eeg1Data, [data1[i:i+window_size]], axis=0)
    eeg2Data = np.append(eeg2Data, [data2[i:i+window_size]], axis=0)
    eeg3Data = np.append(eeg3Data, [data3[i:i+window_size]], axis=0)
    labelData = np.append(labelData, [label[i:i+window_size]], axis=0)

ftData = np.array([eeg1Data, eeg2Data, eeg3Data, labelData])

import seaborn as sns
import matplotlib.pyplot as plt
from librosa.core import stft
from librosa.display import specshow
from librosa import amplitude_to_db
D = np.abs(stft(ftData[0][0]))
D_left = np.abs(stft(ftData[0][0], center=False))
D_short = np.abs(stft(ftData[0][0], hop_length=64))
specshow(D, y_axis='log', x_axis='time')
specshow(D_left, y_axis='log', x_axis='time')
specshow(D_short, y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

print(ftData[3][0])

ret = []
for i, d in enumerate(ftData[3]):
    unique, counts = np.unique(d, return_counts=True)
    ret.append(dict(zip(unique, counts)))

ret = np.array([])
for i, d in enumerate(ftData[3]):
    ret = np.append(ret, d[np.argmax(d)])


fig = plt.figure()
fig = fig.sub_plot()
ax1 = fig.add_subplot()






specshow(amplitude_to_db())
