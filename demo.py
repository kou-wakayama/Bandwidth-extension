#!/usr/bin/env python
# coding: utf-8

# 録音


import pyaudio  
import wave     
 
RECORD_SECONDS = 5 
WAVE_OUTPUT_FILENAME = "sample.wav" 
iDeviceIndex = 0 
 
FORMAT = pyaudio.paInt16 
CHANNELS = 1             
RATE = 24000             
CHUNK = 2**11            
audio = pyaudio.PyAudio() 
 
stream = audio.open(format=FORMAT, channels=CHANNELS,
        rate=RATE, input=True,
        input_device_index = iDeviceIndex, 
        frames_per_buffer=CHUNK)
 
#--------------録音開始---------------
 
print ("recording...")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
 
 
print ("finished recording")
 
#--------------録音終了---------------
 
stream.stop_stream()
stream.close()
audio.terminate()
 
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()


# 録音した音声のスペクトログラムと波形


import scipy.io.wavfile as wio
import matplotlib.pyplot as plt
 
rate, data = wio.read(WAVE_OUTPUT_FILENAME)

plt.plot(data)

plt.show()


pxx, freq, bins, t = plt.specgram(data,Fs = rate, cmap="CMRmap", vmin = -30.0, vmax = 40.0)
plt.show()


# 録音した音声をダウンサンプリング


from scipy import signal
import numpy as np

data_down = signal.resample(data, 80000)
data_re = signal.resample(data_down, 120000)


# 音声を再生


import soundfile as sf


sf.write(
    "data_down.wav",
    data_re / 2 ** 15,
    24000,
    "PCM_16",
)

def PlayWavFie(Filename = "hifi_gan.wav"):
    try:
        wf = wave.open(Filename, "r")
    except FileNotFoundError: #ファイルが存在しなかった場合
        print("[Error 404] No such file or directory: " + Filename)
        return 0
        
    # ストリームを開く
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # 音声を再生
    chunk = 1024
    data = wf.readframes(chunk)
    while len(data) != 0:
        stream.write(data)
        data = wf.readframes(chunk)
    stream.close()
    p.terminate()

PlayWavFie("data_down.wav") 


# ダウンサンプリングした音声のスペクトログラムを表示


pxx, freq, bins, t = plt.specgram(data_re,Fs = 24000, cmap="CMRmap", vmin = -30.0, vmax = 40.0)
plt.show()


# 録音した音声の特徴量を抽出


sampling_rate = 24000
hop_size = 300

import librosa

def logmelfilterbank(
    audio,
    sampling_rate,
    fft_size=2048,
    hop_size=256,
    win_length=1200,
    window="hann",
    num_mels=80,
    fmin=80,
    fmax=7600,
    eps=1e-10,
    log_base=10.0,
):
    
    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T
    
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(
        sr=sampling_rate,
        n_fft=fft_size,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel = np.maximum(eps, np.dot(spc, mel_basis.T))
    
    if log_base is None:
        return np.log(mel)
    elif log_base == 10.0:
        return np.log10(mel)
    elif log_base == 2.0:
        return np.log2(mel)
    else:
        raise ValueError(f"{log_base} is not supported.")



mel = logmelfilterbank(
            data_re / 2 ** 15,
            sampling_rate=24000,
            hop_size=300,
            fft_size=2048,
            win_length=1200,
            window="hann",
            num_mels=80,
            fmin=80,
            fmax=7600,
            # keep compatibility
            log_base=10.0,
        )




# hd5fファイルを読み込む


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


# 学習済みモデルをロード


import torch

def load_model(checkpoint, config=None, stats=None):
    """Load trained model.

    Args:
        checkpoint (str): Checkpoint path.
        config (dict): Configuration dict.
        stats (str): Statistics file path.

    Return:
        torch.nn.Module: Model instance.

    """
    # load config if not provided
    if config is None:
        dirname = os.path.dirname(checkpoint)
        config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)

    # lazy load for circular error
    import parallel_wavegan.models

    # get model and load parameters
    model_class = getattr(
        parallel_wavegan.models,
        config.get("generator_type", "ParallelWaveGANGenerator"),
    )
    # workaround for typo #295
    generator_params = {
        k.replace("upsample_kernal_sizes", "upsample_kernel_sizes"): v
        for k, v in config["generator_params"].items()
    }
    model = model_class(**generator_params)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model"]["generator"]
    )

    # check stats existence
    if stats is None:
        dirname = os.path.dirname(checkpoint)
        if config["format"] == "hdf5":
            ext = "h5"
        else:
            ext = "npy"
        if os.path.exists(os.path.join(dirname, f"stats.{ext}")):
            stats = os.path.join(dirname, f"stats.{ext}")

    # load stats
    if stats is not None:
        model.register_stats(stats)

    # add pqmf if needed
    if config["generator_params"]["out_channels"] > 1:
        # lazy load for circular error
        from parallel_wavegan.layers import PQMF

        pqmf_params = {}
        if LooseVersion(config.get("version", "0.1.0")) <= LooseVersion("0.4.2"):
            # For compatibility, here we set default values in version <= 0.4.2
            pqmf_params.update(taps=62, cutoff_ratio=0.15, beta=9.0)
        model.pqmf = PQMF(
            subbands=config["generator_params"]["out_channels"],
            **config.get("pqmf_params", pqmf_params),
        )

    return model


import yaml
import os

config = "config.yml"
with open(config) as f:
    config = yaml.load(f, Loader=yaml.Loader)
        
c = mel.astype(np.float32)
    
device = torch.device("cpu")
model = load_model("checkpoint-2500000steps.pkl", config)
model.remove_weight_norm()
model = model.eval().to(device)


# 合成音声を生成


c = torch.tensor(c, dtype=torch.float).to(device)
y = model.inference(c, normalize_before=True).view(-1)


# 合成音声を保存


import soundfile as sf


sf.write(
    os.path.join(config["outdir"], "hifi_gen.wav"),
    y.detach().numpy(),
    config["sampling_rate"],
    "PCM_16",
)


wio.write("hifi_gan.wav", rate=24000, data=y.detach().numpy())


# 合成音声のスペクトログラムを表示



rate, data = wio.read("exp/hifi_gen.wav")
pxx, freq, bins, t = plt.specgram(data,Fs = rate, cmap="CMRmap", vmin = -30.0, vmax = 40.0)
plt.show()


# 合成音声を再生


def PlayWavFie(Filename = "hifi_gan.wav"):
    try:
        wf = wave.open(Filename, "r")
    except FileNotFoundError: #ファイルが存在しなかった場合
        print("[Error 404] No such file or directory: " + Filename)
        return 0
        
    # ストリームを開く
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # 音声を再生
    chunk = 1024
    data = wf.readframes(chunk)
    while len(data) != 0:
        stream.write(data)
        data = wf.readframes(chunk)
    stream.close()
    p.terminate()


if __name__ == "__main__":
    PlayWavFie("exp/hifi_gen.wav") 






