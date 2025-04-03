import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pandas as pd
import os

def variance_est(liste):
    sum = 0
    for element in liste:
        sum += (element - np.mean(liste))**2
    variance = (1 / (len(liste)-1)) * sum
    return variance

def SNR_max_amp(FFT_freqs, FFT_signal, target_freq_approx, lower_freq, higher_freq):
    signal_band_indicies = []
    noise_band_indicies = []
    lower_freq_index = 0
    higher_freq_index = 0
    target_freq = FFT_freqs[np.argmax(FFT_signal)]

    fault_message = "Unable to find correct frequency, check quality of measurement (frequency spectrum)"

    if target_freq > target_freq_approx:
        return fault_message

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

    signal_amplitude = np.max(FFT_signal)
    noise_mean = np.mean(FFT_signal[noise_band_indicies])

    snr_amplitude_estimate = 10 * np.log10(signal_amplitude / noise_mean)

    return snr_amplitude_estimate

def estimate_heartrate_3_colorchannels(filepath):
    df = pd.read_csv(filepath, sep=" ", header=None)

    signal_red = df[0]
    signal_green = df[1]
    signal_blue = df[2]

    signal_red -= np.mean(signal_red)
    signal_green -= np.mean(signal_green)
    signal_blue -= np.mean(signal_blue)

    #------------------------------------------------------------------------------------------------------------------------------------------------
    #endrer signalet til med/uten hanning-vindu. Kommenter ut innenfor linjene for å fjerne hanning-vinduet
    signal_red = signal_red * np.hanning(len(signal_red))
    signal_green = signal_green * np.hanning(len(signal_green))
    signal_blue = signal_blue * np.hanning(len(signal_blue))
    #------------------------------------------------------------------------------------------------------------------------------------------------

    signal_red = np.pad(signal_red,int(len(signal_red)*8),mode="constant",constant_values=0)
    signal_blue = np.pad(signal_blue,int(len(signal_blue)*8),mode="constant",constant_values=0)
    signal_green = np.pad(signal_green,int(len(signal_green)*8),mode="constant",constant_values=0)

    #signal_ = signal_ * np.hanning(len(signal_))
    #time_original = np.linspace(0,30,len(signal_original))
    time = np.linspace(0,30,len(signal_red))

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

    # Compute FFT
    def compute_fft(signal, fs):
        N = len(signal)  # Number of samples
        fft_values = np.fft.fft(signal)  # Compute FFT
        fft_freqs = np.fft.fftfreq(N, d=1/fs)  # Frequency bins
        return fft_freqs[:N // 2], np.abs(fft_values[:N // 2])  # Keep positive frequencies

    # Apply the filter to the signal
    filtered_signal_red = signal.filtfilt(b, a, signal_red)
    filtered_signal_green = signal.filtfilt(b, a, signal_green)
    filtered_signal_blue = signal.filtfilt(b, a, signal_blue)

    #------------------------------------------------------------------------------------------------------------------------------------------------
    #endrer signalet til med/uten filter. Kommenter ut innenfor linjene for å legge til filteret. Ekstremt cursed, men gjorde det litt raskt
    """ filtered_signal_red = signal_red
    filtered_signal_green = signal_green
    filtered_signal_blue = signal_blue """
    #------------------------------------------------------------------------------------------------------------------------------------------------

    freqs_red, filtered_signal_FFT_red = compute_fft(filtered_signal_red, fs)
    freqs_green, filtered_signal_FFT_green = compute_fft(filtered_signal_green, fs)
    freqs_blue, filtered_signal_FFT_blue = compute_fft(filtered_signal_blue, fs)

    #filtered_signal_FFT = 10*np.log10(filtered_signal_FFT/np.max(filtered_signal_FFT))

    bpm_red = freqs_red * 60
    bpm_green = freqs_green * 60
    bpm_blue = freqs_blue * 60

    estimated_heartrate_red = bpm_red[np.argmax(filtered_signal_FFT_red)]
    estimated_heartrate_green = bpm_green[np.argmax(filtered_signal_FFT_green)]
    estimated_heartrate_blue = bpm_blue[np.argmax(filtered_signal_FFT_blue)]

    estimated_heartrates = [estimated_heartrate_red,estimated_heartrate_green,estimated_heartrate_blue]
    mean = np.mean(estimated_heartrates)
    variance = variance_est(estimated_heartrates)
    
    std = np.sqrt(variance)

    snr_red = SNR_max_amp(bpm_red,filtered_signal_FFT_red,80,40,220)
    snr_green = SNR_max_amp(bpm_green,filtered_signal_FFT_green,80,40,220)
    snr_blue = SNR_max_amp(bpm_blue,filtered_signal_FFT_blue,80,40,220)

    return estimated_heartrate_red,estimated_heartrate_green,estimated_heartrate_blue,snr_red,snr_green,snr_blue,mean,std



def estimate_multiple_heartrates(dir):
    counting_variable = 1
    print("----------------------------------------------------------------------------------------")
    for file in os.listdir(dir):
        if file != ".DS_Store":
            number = ""
            if counting_variable == 1:
                number = str(counting_variable)+"st"
            elif counting_variable == 2:
                number = str(counting_variable)+"nd"
            elif counting_variable == 3:
                number = str(counting_variable)+"rd"
            else:
                number = str(counting_variable)+"th"
            heartrate_red,heartrate_green,heartrate_blue,snr_red,snr_green,snr_blue,mean,std = estimate_heartrate_3_colorchannels(str(dir)+"/"+file)
            print("----------------------------------------------------------------------------------------")
            print(f"Red estimate ({dir} -> {file}): {heartrate_red:.3f})")
            print(f"Green estimate ({dir}): {heartrate_green:.3f})")
            print(f"Blue estimate ({dir}): {heartrate_blue:.3f})")
            print(f"Red SNR-estimate ({dir} -> {file}): {snr_red:.3f})")
            print(f"Green SNR-estimate ({dir}): {snr_green:.3f})")
            print(f"Blue SNR-estimate ({dir}): {snr_blue:.3f})")
            print(f"Mean ({dir}): {mean:.3f})")
            print(f"Std ({dir}): {std:.3f})")
            print("----------------------------------------------------------------------------------------")
            counting_variable += 1
    print("----------------------------------------------------------------------------------------")

#------------------------------------------------------------------------------------------------------------------------------------------------
def estimate_multiple_heartrates_to_csv(dir):
    counting_variable = 1
    file_names = []
    red_estimates = []
    green_estimates = []
    blue_estimates = []
    SNR_red_estimates = []
    SNR_green_estimates = []
    SNR_blue_estimates = []
    mean_estimates = []
    std_estimates = []
    for file in os.listdir(dir):
        if file != ".DS_Store":
            heartrate_red,heartrate_green,heartrate_blue,SNR_red,SNR_green,SNR_blue,mean,std = estimate_heartrate_3_colorchannels(str(dir)+"/"+str(file))

            file_names.append(str(file))
            red_estimates.append(heartrate_red)
            green_estimates.append(heartrate_green)
            blue_estimates.append(heartrate_blue)
            SNR_red_estimates.append(SNR_red)
            SNR_green_estimates.append(SNR_green)
            SNR_blue_estimates.append(SNR_blue)
            mean_estimates.append(mean)
            std_estimates.append(std)

            counting_variable += 1
    data = {
        "File names": file_names,
        "Heartrate estimates (red color-channel)": red_estimates,
        "Heartrate estimates (green color-channel)": green_estimates,
        "Heartrate estimates (blue color-channel)": blue_estimates,
        "SNR estimates (red color-channel)": SNR_red_estimates,
        "SNR estimates (green color-channel)": SNR_green_estimates,
        "SNR estimates (blue color-channel)": SNR_blue_estimates,
        "Mean estimates": mean_estimates,
        "Std estimates": std_estimates
    }
    df = pd.DataFrame(data)
    df.to_csv("Lab3/resultater_X.csv",index=False)
    print("----------------------------------------------------------------------------------------")
#------------------------------------------------------------------------------------------------------------------------------------------------



estimate_multiple_heartrates_to_csv("Lab3/maalinger_rapport")

