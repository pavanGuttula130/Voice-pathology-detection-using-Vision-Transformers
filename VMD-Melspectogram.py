# %%
import os
import librosa
import numpy as np
from vmdpy import VMD
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# %%
train_data_path = r'E:\Semester-6\Speech-Processing\Project\Trail_5\Phase - II\MutliClass_SplitedData\Train'

alpha = 2000       # moderate bandwidth constraint  
tau = 0.           # noise-tolerance (no strict fidelity enforcement)  
K = 5              # 5 modes  lh
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7

for i in tqdm(os.listdir(train_data_path)):
    audio_file = os.path.join(train_data_path,i)
    audio_file_,sampling_rate = librosa.load(audio_file,sr=8000)
    
    u, u_hat, omega = VMD(audio_file_, alpha, tau, K, DC, init, tol)
    spectrogram = librosa.feature.melspectrogram(y=u[0], sr=8000)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)  # Convert to dB scale

    # Normalize spectrogram
    # normalized_spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    librosa.display.specshow(spectrogram,vmin=-80,vmax=10)
    path = "train_mel_trail_2"+"/"+i+".png"
    plt.savefig(path)

# %%
train_data_path = r'E:\Semester-6\Speech-Processing\Project\Trail_5\Phase - II\MutliClass_SplitedData\Test'

alpha = 2000       # moderate bandwidth constraint  
tau = 0.           # noise-tolerance (no strict fidelity enforcement)  
K = 5              # 5 modes  lh
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7

for i in tqdm(os.listdir(train_data_path)):
    audio_file = os.path.join(train_data_path,i)
    audio_file_,sampling_rate = librosa.load(audio_file,sr=8000)
    
    u, u_hat, omega = VMD(audio_file_, alpha, tau, K, DC, init, tol)
    spectrogram = librosa.feature.melspectrogram(y=u[0], sr=8000)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)  # Convert to dB scale

    # Normalize spectrogram
    # normalized_spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    librosa.display.specshow(spectrogram,vmin=-80,vmax=10)
    path = "test_mel_trail_2"+"/"+i+".png"
    plt.savefig(path)

# %%
def image_cleaner(img_path):
    img = cv2.imread(img_path)
    ## (1) Convert to gray, and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    ## (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    ## (4) Crop and save it
    x,y,w,h = cv2.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]
    cv2.imwrite(img_path, dst)

# %%
for i in tqdm(os.listdir(r'E:\Semester-6\Speech-Processing\Project\Trail_5\Phase - II\test_mel_trail_2')):
    image = os.path.join(r'E:\Semester-6\Speech-Processing\Project\Trail_5\Phase - II\test_mel_trail_2',i)
    image_cleaner(image)

# %%
for i in tqdm(os.listdir(r'E:\Semester-6\Speech-Processing\Project\Trail_5\Phase - II\train_mel_trail_2')):
    image = os.path.join(r'E:\Semester-6\Speech-Processing\Project\Trail_5\Phase - II\train_mel_trail_2',i)
    image_cleaner(image)


