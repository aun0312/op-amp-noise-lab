import io
import numpy as np
import streamlit as st
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd
import matplotlib.pyplot as plt

st.set_page_config(page_title='Op-Amp Noise Lab', layout='wide')
st.title('Op-Amp Noise Lab')
st.caption('음성 + 간섭 생성 · FFT · PSpice / myDAQ 데이터 내보내기')
f = st.file_uploader('음성 WAV 파일', type=['wav'])
fs = st.selectbox('출력 샘플링 주파수 (Hz)', [48000, 44100, 24000, 16000, 8000, 96000])
duration = st.slider('최대 길이 (초)', 0.1, 10.0, 1.0, 0.1)
if f is not None:
    try:
        original_fs, x = wavfile.read(io.BytesIO(f.getvalue()))
        if x.dtype.kind == 'u':
            mid = (np.iinfo(x.dtype).max + 1)/2
            x = (x.astype(float)-mid)/mid
        elif x.dtype.kind == 'i':
            x = x.astype(float)/(-float(np.iinfo(x.dtype).min))
        else:
            x = x.astype(float)
        if x.ndim == 2: x = x.mean(axis=1)
        x = x[:max(1,round(duration*original_fs))]
        g = gcd(int(original_fs), int(fs))
        x = resample_poly(x, fs//g, original_fs//g)
    except Exception as e:
        st.error(f'WAV 읽기 실패: {e}'); st.stop()
else:
    st.info('파일이 없으면 1 kHz 테스트 정현파를 사용합니다.')
    x = np.sin(2*np.pi*1000*np.arange(round(duration*fs))/fs)
if len(x)<32 or not np.all(np.isfinite(x)) or np.max(np.abs(x))==0:
    st.error('유효한 비무음 신호가 필요합니다.'); st.stop()
x = x - np.mean(x)
if np.max(np.abs(x)) == 0: st.error('AC 신호가 없습니다.'); st.stop()
peak = st.sidebar.slider('원본 최대 진폭 (mV)', 1, 500, 10)/1000
x = peak*x/np.max(np.abs(x))
t = np.arange(len(x))/fs
kind = st.sidebar.selectbox('잡음 종류', ['정현파 간섭', 'White noise', '고주파 대역 잡음', '60 Hz 전원 간섭'])
maxf = int(fs/2*0.95)
rng = np.random.default_rng(42)
if kind == '정현파 간섭':
    freq = st.sidebar.slider('간섭 주파수 (Hz)', 20, maxf, min(10000,maxf), 10)
    n = np.sin(2*np.pi*freq*t)
elif kind == '60 Hz 전원 간섭': n = np.sin(2*np.pi*60*t)
elif kind == 'White noise': n = rng.standard_normal(len(x))
else:
    lo, hi = st.sidebar.slider('잡음 대역 (Hz)', 20, maxf, (min(5000,maxf//2),min(15000,maxf)), 10)
    if lo>=hi: st.warning('서로 다른 대역 경계를 지정하세요.'); st.stop()
    n = sosfiltfilt(butter(4,[lo,hi],btype='bandpass',fs=fs,output='sos'),rng.standard_normal(len(x)))
snr = st.sidebar.slider('목표 입력 SNR (dB)', -20, 40, 0)
rms = lambda a: np.sqrt(np.mean(a*a))
n = n*rms(x)/(max(rms(n),1e-15)*10**(snr/20))
if st.sidebar.checkbox('잡음 끄기'): n = np.zeros_like(x)
y = x+n
cols=st.columns(3)
cols[0].metric('원본 peak',f'{np.max(np.abs(x))*1000:.2f} mV')
cols[1].metric('잡음 RMS',f'{rms(n)*1000:.2f} mV')
cols[2].metric('합성 peak',f'{np.max(np.abs(y))*1000:.2f} mV')
scale = 0.95/max(np.max(np.abs(x)),np.max(np.abs(y)),1e-12)
def wav(a, gain=1):
    b=io.BytesIO(); wavfile.write(b,fs,np.asarray(a*gain,dtype=np.float32)); return b.getvalue()
a,b=st.columns(2)
with a: st.write('원본 청취'); st.audio(wav(x,scale),format='audio/wav')
with b: st.write('잡음 합성 청취'); st.audio(wav(y,scale),format='audio/wav')
st.caption('청취용 음원에는 두 신호에 동일한 재생용 이득을 적용합니다. 내보내는 데이터의 전압은 변경하지 않습니다.')
fig,ax=plt.subplots(2,1,figsize=(10,5),layout='constrained')
end=min(len(t),round(fs*0.03))
ax[0].plot(t[:end]*1000,x[:end]*1000,label='Clean'); ax[0].plot(t[:end]*1000,y[:end]*1000,label='Noisy',alpha=.75); ax[0].set(xlabel='Time (ms)',ylabel='Voltage (mV)'); ax[0].legend(); ax[0].grid(alpha=.2)
w=np.hanning(len(x)); freq=np.fft.rfftfreq(len(x),1/fs)
for a,label in [(x,'Clean'),(y,'Noisy')]:
    z=np.abs(np.fft.rfft(a*w))*2/w.sum(); z[0]/=2
    if len(a)%2==0: z[-1]/=2
    ax[1].plot(freq,20*np.log10(np.maximum(z,1e-12)),label=label,alpha=.8)
ax[1].set(xlabel='Frequency (Hz)',ylabel='Magnitude (dBV peak)',ylim=None); ax[1].legend(); ax[1].grid(alpha=.2)
st.pyplot(fig); plt.close(fig)
def txt(a):
    b=io.StringIO(); np.savetxt(b,np.column_stack((t,a)),fmt='%.10e',delimiter='\t'); return b.getvalue()
st.subheader('실험용 파일 내보내기')
st.download_button('합성 신호 TXT (시간 s / 전압 V)',txt(y),'voice_noise.txt','text/plain')
st.download_button('원본 기준 신호 TXT',txt(x),'voice_clean.txt','text/plain')
st.download_button('합성 WAV (전압 수치 보존)',wav(y),'voice_noise.wav','audio/wav')
st.download_button('청취용 WAV',wav(y,scale),'voice_noise_listen.wav','audio/wav')
st.caption('PSpice VPWL_FILE용 헤더 없는 두 열 형식입니다. myDAQ AWG 가져오기 형식과 장비 샘플링 제한은 설치 버전에서 확인하세요. 출력 전압 범위 및 공통 접지도 확인해야 합니다.')
