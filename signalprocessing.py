import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pandas as pd

df = pd.read_csv("Lab3/maalinger_rapport/kaldfinger_tran_2", sep=" ", header=None)

tran_dir = "Lab3/maalinger_rapport/basis_tran_kraftiglys_1"
refl_dir = "Lab3/maalinger_rapport/basis_refl_kraftiglys_7"
kald_dir = "Lab3/maalinger_rapport/kaldfinger_tran_2"
varm_dir = "Lab3/maalinger_rapport/varmfinger_tran_1"
dir_SNR = "Lab3/maalinger_rapport/varmfinger_tran_5"
target_freq_SNR = 61

# Sampling parameters
fs = 30  # Sampling frequency (Hz)
low_cutoff = 0.6  # Lower cutoff frequency (Hz)
high_cutoff = 4.0  # Upper cutoff frequency (Hz)
order = 4  # Filter order

# Design a Butterworth bandpass filter
nyquist = 0.5 * fs  # Nyquist frequency
low = low_cutoff / nyquist  # Normalize low cutoff
high = high_cutoff / nyquist  # Normalize high cutoff

b, a = signal.butter(order, [low, high], btype='bandpass', analog=False)

def compute_fft(signal, fs):
    N = len(signal)  # Number of samples
    fft_values = np.fft.fft(signal)  # Compute FFT
    fft_freqs = np.fft.fftfreq(N, d=1/fs)  # Frequency bins
    return fft_freqs[:N // 2], np.abs(fft_values[:N // 2])  # Keep positive frequencies

def compute_dB(signal_values):
    dB_value = 10*np.log10(signal_values/np.max(signal_values))
    return dB_value

def make_time_signal(dir):
    df = pd.read_csv(dir, sep=" ", header=None)
    signal_red = df[0]
    signal_green = df[1]
    signal_blue = df[2]
    signal_red = signal_red - np.mean(signal_red)
    signal_green = signal_green - np.mean(signal_green)
    signal_blue = signal_blue - np.mean(signal_blue)

    signal_red = signal.filtfilt(b, a, signal_red)
    signal_green = signal.filtfilt(b, a, signal_green)
    signal_blue = signal.filtfilt(b, a, signal_blue)
    return signal_red, signal_green, signal_blue

def make_time_signal_no_filter(dir):
    df = pd.read_csv(dir, sep=" ", header=None)
    signal_red = df[0]
    signal_green = df[1]
    signal_blue = df[2]
    signal_red = signal_red - np.mean(signal_red)
    signal_green = signal_green - np.mean(signal_green)
    signal_blue = signal_blue - np.mean(signal_blue)

    return signal_red, signal_green, signal_blue

def make_FFT_signal_with_preprosessing(dir):
    red, green, blue = make_time_signal(dir)

    color_channels = [red, green, blue]
    freqs_list = []
    FFT_signals_list = []

    for element in color_channels:
        element = element * np.hanning(len(element))
        element = np.pad(element,int(len(element)*8),mode="constant",constant_values=0)
        freqs_element, FFT_signal = compute_fft(element, fs)
        FFT_signal = compute_dB(FFT_signal)
        freqs_list.append(freqs_element*60)
        FFT_signals_list.append(FFT_signal)
    return freqs_list, FFT_signals_list

def make_FFT_signal_with_preprosessing_no_dB(dir):
    red, green, blue = make_time_signal(dir)

    color_channels = [red, green, blue]
    freqs_list = []
    FFT_signals_list = []

    for element in color_channels:
        element = element * np.hanning(len(element))
        element = np.pad(element,int(len(element)*8),mode="constant",constant_values=0)
        freqs_element, FFT_signal = compute_fft(element, fs)
        freqs_list.append(freqs_element*60)
        FFT_signals_list.append(FFT_signal)
    return freqs_list, FFT_signals_list

def make_FFT_signal_no_preprosessing(dir):
    red, green, blue = make_time_signal_no_filter(dir)

    color_channels = [red, green, blue]
    freqs_list = []
    FFT_signals_list = []

    for element in color_channels:
        element = np.pad(element,int(len(element)*8),mode="constant",constant_values=0)
        freqs_element, FFT_signal = compute_fft(element, fs)
        FFT_signal = compute_dB(FFT_signal)
        freqs_list.append(freqs_element*60)
        FFT_signals_list.append(FFT_signal)
    return freqs_list, FFT_signals_list

def make_FFT_signal_no_preprosessing_no_dB(dir):
    red, green, blue = make_time_signal_no_filter(dir)

    color_channels = [red, green, blue]
    freqs_list = []
    FFT_signals_list = []

    for element in color_channels:
        element = np.pad(element,int(len(element)*8),mode="constant",constant_values=0)
        freqs_element, FFT_signal = compute_fft(element, fs)
        freqs_list.append(freqs_element*60)
        FFT_signals_list.append(FFT_signal)
    return freqs_list, FFT_signals_list



def SNR_freq_buckets(FFT_freqs, FFT_signal, target_freq_approx, lower_freq, higher_freq):
    signal_band_indicies = []
    noise_band_indicies = []
    lower_freq_index = 0
    higher_freq_index = 0
    target_freq = FFT_freqs[np.argmax(FFT_signal)]

    fault_message = "Unable to find correct frequency, check quality of measurement (frequency spectrum)"

    if target_freq > target_freq_approx:
        return fault_message, fault_message

    for i in range(len(FFT_freqs)):
        if (FFT_freqs[i] < lower_freq) and (FFT_freqs[i+1] > lower_freq):
            lower_freq_index = i
            break
    for i in range(len(FFT_freqs)):
        if (FFT_freqs[i] < higher_freq) and (FFT_freqs[i+1] > higher_freq):
            higher_freq_index = i
            break

    for i in range(lower_freq_index,higher_freq_index,1):
        if (abs(FFT_freqs[i]) > target_freq-5) and (abs(FFT_freqs[i]) < target_freq+5):
            signal_band_indicies.append(i)
        else:
            noise_band_indicies.append(i)

    signal_energy = np.sum(FFT_signal[signal_band_indicies]**2)
    noise_energy = np.sum(FFT_signal[noise_band_indicies]**2)

    signal_amplitude = np.max(FFT_signal)
    noise_mean = np.mean(FFT_signal[noise_band_indicies])

    snr = 10 * np.log10(signal_energy / noise_energy)       #beregner SNR
    snr_amplitude_estimate = 10 * np.log10(signal_amplitude / noise_mean)

    return snr, snr_amplitude_estimate

import numpy as np

def SNR_target_freq(FFT_freqs, FFT_signal, target_freq, lower_freq, higher_freq):
    noise_band_indices = []
    lower_freq_index = 0
    higher_freq_index = 0
    
    fault_message = "Unable to find correct frequency, check quality of measurement (frequency spectrum)"
    
    # Check if the target frequency is within the given range
    if not (lower_freq <= target_freq <= higher_freq):
        return fault_message
    
    # Find index bounds for the given frequency range
    for i in range(len(FFT_freqs) - 1):
        if FFT_freqs[i] < lower_freq <= FFT_freqs[i+1]:
            lower_freq_index = i
        if FFT_freqs[i] < higher_freq <= FFT_freqs[i+1]:
            higher_freq_index = i
            break
    
    # Identify the index of the maximum amplitude within the target frequency range
    target_indices = [i for i in range(lower_freq_index, higher_freq_index) if target_freq - 5 <= abs(FFT_freqs[i]) <= target_freq + 5]
    
    if not target_indices:
        return fault_message, fault_message
    
    max_signal_index = max(target_indices, key=lambda i: FFT_signal[i])
    signal_amplitude = FFT_signal[max_signal_index]
    
    # Identify noise indices
    for i in range(lower_freq_index, higher_freq_index):
        if i != max_signal_index:
            noise_band_indices.append(i)
    
    # Compute noise mean
    noise_mean = np.mean(FFT_signal[noise_band_indices])
    
    # Calculate SNR
    snr_amplitude_estimate = 10 * np.log10(signal_amplitude / noise_mean)
    
    return snr_amplitude_estimate
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#definisjoner av variable, ble mest brukt i utviklingen av koden
""" signal_original = df[1]
signal_ = signal_original - np.mean(signal_original)
signal_with_hanning = signal_ * np.hanning(len(signal_))
signal_with_hanning_and_zeropad = np.pad(signal_with_hanning,int(len(signal_)*8),mode="constant",constant_values=0)
signal_with_zeropad = np.pad(signal_,int(len(signal_)*8),mode="constant",constant_values=0)
time_original = np.linspace(0,30,len(signal_original))
time = np.linspace(0,30,len(signal_))

filtered_signal = signal.filtfilt(b, a, signal_)
filtered_signal_with_zeropad = signal.filtfilt(b, a, signal_with_zeropad)
filtered_signal_with_hanning = signal.filtfilt(b, a, signal_with_hanning)
filtered_signal_with_hanning_and_zeropad = signal.filtfilt(b, a, signal_with_hanning_and_zeropad)

freqs_wofilter, signal_FFT_wofilter = compute_fft(signal_, fs)
freqs_wofilter_zeropad, signal_FFT_wofilter_zeropad = compute_fft(signal_with_zeropad, fs)
freqs, filtered_signal_FFT = compute_fft(filtered_signal, fs)
freqs_zeropad, filtered_signal_with_zeropad_FFT = compute_fft(filtered_signal_with_zeropad, fs)
freqs_with_hanning_and_zeropad, filtered_signal_FFT_with_hanning_and_zeropad = compute_fft(filtered_signal_with_hanning_and_zeropad, fs)

signal_FFT_wofilter_dB = 10*np.log10(signal_FFT_wofilter/np.max(signal_FFT_wofilter))
signal_with_zeropad_dB = compute_dB(filtered_signal_with_zeropad_FFT)
signal_wofilter_zeropad_dB = compute_dB(signal_FFT_wofilter_zeropad)
filtered_signal_FFT_dB = 10*np.log10(filtered_signal_FFT/np.max(filtered_signal_FFT))
filtered_signal_FFT_with_hanning_and_zeropad_dB = 10*np.log10(filtered_signal_FFT_with_hanning_and_zeropad/np.max(filtered_signal_FFT_with_hanning_and_zeropad))

bpm = freqs * 60
bpm_with_hanning = freqs_with_hanning_and_zeropad * 60
estimated_heartrate = bpm[np.argmax(filtered_signal_FFT)] """
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#Plott av tre rådatasignaler fra en måling, rød, grønn og blå fargekanal
""" red,green,blue = make_time_signal_no_filter(varm_dir)
fig, axes = plt.subplots(3, 1, figsize=(10, 9))

# First subplot
axes[0].plot(time_original[15:], red[15:], label="Unfiltered Signal - Red", linewidth=1.5, color="r", alpha=0.8)
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Amplitude [mean pixel value]")
axes[0].legend(loc="lower right")
axes[0].set_title("Rosbustness-test, warm finger. Time-plot of unfiltered signals from all color channels.")
axes[0].grid()

# Second subplot
axes[1].plot(time_original[15:], green[15:], label="Unfiltered Signal - Green", linewidth=1.5, color="g", alpha=0.8)
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Amplitude [mean pixel value]")
axes[1].legend(loc="lower right")
axes[1].grid()

# Third subplot
axes[2].plot(time_original[15:], blue[15:], label="Unfiltered Signal - Blue", linewidth=1.5, color="b", alpha=0.8)
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Amplitude [mean pixel value]")
axes[2].legend(loc="lower right")
axes[2].grid()

plt.tight_layout()
plt.show() """
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Plott av alle tre fargekanalene - frekvensspekter med preprosessering
""" freqs_list, FFT_signals_list = make_FFT_signal_with_preprosessing(dir_SNR)

fig, axes = plt.subplots(3, 1, figsize=(10, 9))

# First subplot
axes[0].plot(freqs_list[0][:], FFT_signals_list[0][:], label="Filtered Signal with Hanning window - Red", linewidth=1.5, color="r", alpha=0.8)
axes[0].set_xlabel("Frequency [bpm]")
axes[0].set_ylabel("Amplitude [dB]")
axes[0].set_xlim(0, 240)
axes[0].set_ylim(-30, 0)
axes[0].axvline(x=40, color='orange', linestyle='--', linewidth=1.6, label="bpm=40")
axes[0].axvline(x=220, color='purple', linestyle='--', linewidth=1.6, label="bpm=220")
axes[0].legend(loc="lower right")
axes[0].set_title("FFT signals of all color channels with pre-prosessing")
axes[0].grid()

# Second subplot
axes[1].plot(freqs_list[1][:], FFT_signals_list[1][:], label="Filtered Signal with Hanning window - Green", linewidth=1.5, color="g", alpha=0.8)
axes[1].set_xlabel("Frequency [bpm]")
axes[1].set_ylabel("Amplitude [dB]")
axes[1].set_xlim(0, 240)
axes[1].set_ylim(-30, 0)
axes[1].axvline(x=40, color='orange', linestyle='--', linewidth=1.6, label="bpm=40")
axes[1].axvline(x=220, color='purple', linestyle='--', linewidth=1.6, label="bpm=220")
axes[1].legend(loc="lower right")
axes[1].grid()

# Third subplot
axes[2].plot(freqs_list[2][:], FFT_signals_list[2][:], label="Filtered Signal with Hanning window - Blue", linewidth=1.5, color="b", alpha=0.8)
axes[2].set_xlabel("Frequency [bpm]")
axes[2].set_ylabel("Amplitude [dB]")
axes[2].set_xlim(0, 240)
axes[2].set_ylim(-30, 0)
axes[2].axvline(x=40, color='orange', linestyle='--', linewidth=1.6, label="bpm=40")
axes[2].axvline(x=220, color='purple', linestyle='--', linewidth=1.6, label="bpm=220")
axes[2].legend(loc="lower right")
axes[2].grid()

plt.tight_layout()
plt.show() """
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Plott av alle tre fargekanalene - frekvensspekter uten preprosessering
freqs_list, FFT_signals_list = make_FFT_signal_no_preprosessing(varm_dir)

fig, axes = plt.subplots(3, 1, figsize=(10, 9))

# First subplot
axes[0].plot(freqs_list[0][:], FFT_signals_list[0][:], label="Unfiltered signal - Red", linewidth=1.5, color="r", alpha=0.8)
axes[0].set_xlabel("Frequency [bpm]")
axes[0].set_ylabel("Amplitude [dB]")
axes[0].set_xlim(0, 240)
axes[0].set_ylim(-30, 0)
axes[0].axvline(x=40, color='orange', linestyle='--', linewidth=1.6, label="bpm=40")
axes[0].axvline(x=220, color='purple', linestyle='--', linewidth=1.6, label="bpm=220")
axes[0].legend(loc="lower right")
axes[0].set_title("FFT signals of all color channels without pre-prosessing (i.e. Hanning window and filtering)")
axes[0].grid()

# Second subplot
axes[1].plot(freqs_list[1][:], FFT_signals_list[1][:], label="Unfiltered signal - Green", linewidth=1.5, color="g", alpha=0.8)
axes[1].set_xlabel("Frequency [bpm]")
axes[1].set_ylabel("Amplitude [dB]")
axes[1].set_xlim(0, 240)
axes[1].set_ylim(-30, 0)
axes[1].axvline(x=40, color='orange', linestyle='--', linewidth=1.6, label="bpm=40")
axes[1].axvline(x=220, color='purple', linestyle='--', linewidth=1.6, label="bpm=220")
axes[1].legend(loc="lower right")
axes[1].grid()

# Third subplot
axes[2].plot(freqs_list[2][:], FFT_signals_list[2][:], label="Unfiltered signal - Blue", linewidth=1.5, color="b", alpha=0.8)
axes[2].set_xlabel("Frequency [bpm]")
axes[2].set_ylabel("Amplitude [dB]")
axes[2].set_xlim(0, 240)
axes[2].set_ylim(-30, 0)
axes[2].axvline(x=40, color='orange', linestyle='--', linewidth=1.6, label="bpm=40")
axes[2].axvline(x=220, color='purple', linestyle='--', linewidth=1.6, label="bpm=220")
axes[2].legend(loc="lower right")
axes[2].grid()

plt.tight_layout()
plt.show()
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot the original and filtered signals
""" plt.figure(figsize=(10, 5))
plt.plot(time_original[15:], signal_[15:], label="Original Signal", alpha=0.5)
plt.plot(time_original[15:], filtered_signal[15:], label="Filtered Signal", linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend(loc="upper right")
plt.title("Band-Pass Filtering Between 36bpm and 240 bpm")
plt.grid()
plt.show() """
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Plott av signal uten zeropad
""" plt.figure(figsize=(10, 5))
plt.plot(bpm, signal_FFT_wofilter_dB, label="Unfiltered Signal", alpha=0.5, color='orange')
plt.plot(bpm, filtered_signal_FFT_dB, label="Filtered Signal with hanning-window", alpha=0.5, color='blue')
plt.xlabel("Frequency [bpm]")
plt.ylabel("Magnitude []")
plt.xlim(0, 150)  # Limit x-axis to focus on relevant frequencies
plt.legend(loc="upper right")
plt.title("FFT of Filtered Signal")
plt.grid()
plt.show() """
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#Plott av signal uten filter, med zeropad. Med og uten hanning vindu
""" plt.figure(figsize=(10, 5))
plt.plot(bpm_with_hanning, signal_wofilter_zeropad_dB, label="Filtered Signal with zeropad", alpha=0.5, color='orange')
plt.plot(bpm_with_hanning, filtered_signal_FFT_with_hanning_and_zeropad_dB, label="Filtered Signal with hanning-window and zeropad", alpha=0.5, color='blue')
#plt.plot(bpm, signal_FFT_wofilter_dB)
plt.xlabel("Frequency [bpm]")
plt.ylabel("Magnitude [dB]")
plt.xlim(0, 220)  # Limit x-axis to focus on relevant frequencies
plt.ylim(-50,0)
plt.legend(loc="lower right")
plt.title("FFT of Unfiltered Signal, with and without hanning window")
plt.grid()
plt.show()
signal_wofilter_zeropad_dB """
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Frekvensplott av filtrert zero-paddet signal, med og uten hanning vindu. Amplitude er plottet i dB
""" plt.figure(figsize=(10, 5))
plt.plot(bpm_with_hanning, signal_with_zeropad_dB, label="Filtered Signal with zeropad", alpha=0.5, color='orange')
plt.plot(bpm_with_hanning, filtered_signal_FFT_with_hanning_and_zeropad_dB, label="Filtered Signal with hanning-window and zeropad", alpha=0.5, color='blue')
#plt.plot(bpm, signal_FFT_wofilter_dB)
plt.xlabel("Frequency [bpm]")
plt.ylabel("Magnitude [dB]")
plt.xlim(0, 220)  # Limit x-axis to focus on relevant frequencies
plt.ylim(-50,0)
plt.legend(loc="lower right")
plt.title("FFT of Filtered Signal")
plt.grid()
plt.show() """
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Beregning av SNR vha. frekvensbøtte-metoden + max amp/mean(noise) metoden
""" SNR_freq_buckets_result, SNR_max_amp_result = SNR_freq_buckets(bpm_with_hanning,filtered_signal_FFT_with_hanning_and_zeropad,100,40,220)
print(f"SNR-result for the frequency-bucket method: {SNR_freq_buckets_result} \nSNR-result for the maximum amplitude divided by noise method: {SNR_max_amp_result}") """

#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Beregning av SNR vha. max amp/mean(noise) metoden, med definisjon av frekvensen vi beregner for

""" lower_freq = 40
higher_freq = 220


freqs_list, FFT_signal_without_preprocessing_list = make_FFT_signal_no_preprosessing_no_dB(dir_SNR)

SNR_estimate = SNR_target_freq(freqs_list[1],FFT_signal_without_preprocessing_list[1],target_freq_SNR,lower_freq,higher_freq)
SNR_estimate_blue = SNR_target_freq(freqs_list[2],FFT_signal_without_preprocessing_list[2],target_freq_SNR,lower_freq,higher_freq)

print(f"Green estimate: {SNR_estimate}")
print(f"Blue estimate: {SNR_estimate_blue}") """
