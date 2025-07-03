import os, glob, soundfile, torchaudio, torch
from math import floor
from tqdm import tqdm
import numpy as np
from speaker_embedding import ECAPA_TDNN
from pydub import AudioSegment, silence
from eval import *
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

def init_speaker_encoder(device, source):
	speaker_encoder = ECAPA_TDNN(C=1024).cuda()
	speaker_encoder.eval()
	loadedState = torch.load(source, map_location=device)
	selfState = speaker_encoder.state_dict()
	for name, param in loadedState.items():
		if name in selfState:
			selfState[name].copy_(param)
	for param in speaker_encoder.parameters():
		param.requires_grad = False 
	return speaker_encoder

def load_audio(file_path, target_sr=16000):
    waveform, sample_rate = soundfile.read(file_path)
    waveform = torch.from_numpy(waveform).float()
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr

def pyannote_vad(audio_path, sr):
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token="your_token")
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
    "min_duration_on": 0.5,
    "min_duration_off": 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad_result = pipeline(audio_path)
    segments = [(segment.start, segment.end) for segment in vad_result.get_timeline()]
    return segments

def energy_vad(audio, sr, min_silence_len=100, silence_thresh_db=-20):
    audio_pydub = AudioSegment(audio.numpy().tobytes(), frame_rate=sr, sample_width=2, channels=1)
    non_silent = silence.detect_nonsilent(audio_pydub, min_silence_len=min_silence_len, silence_thresh=silence_thresh_db)
    #print(non_silent)
    segments = [(start / 1000.0, end / 1000.0) for start, end in non_silent]
    return segments

def extract_embeddings(audio, emb_path, sr, segments, model, device='cpu'):
    embeddings = []
    for start, end in segments:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_audio = audio[start_sample:end_sample]
        if len(segment_audio) < sr//2:
            continue
        segment_audio = segment_audio.unsqueeze(0).to(device)
        with torch.inference_mode():
            emb = model.forward(segment_audio.cuda()).cpu().numpy().squeeze()
            embeddings.append(emb)

    if embeddings:
        embeddings = np.stack(embeddings)
        embeddings = np.mean(embeddings,axis=0)
        np.save(emb_path,embeddings)
        return embeddings
    
def run_emb(project_dir, force_emb, language):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = os.path.join(project_dir,'ecapa_tdnn.model')
    audio_dir = os.path.join(project_dir,'wavs')
    emb_dir = os.path.join(project_dir,'embeddings')
    os.makedirs(emb_dir, exist_ok = True)
    emb_model = init_speaker_encoder(device, source)
    all_embeddings = []
    filenames = []
    segments = []
    for filepath in tqdm(glob.glob(os.path.join(audio_dir,'*' if language == 'both' else language, "*.wav")), desc="Extracting Embeddings"):
        emb_path = os.path.join(emb_dir,os.path.basename(filepath).replace('.wav','.npy'))
        filenames.append(os.path.basename(filepath))
        if not force_emb and os.path.exists(emb_path):
             all_embeddings.append(np.expand_dims(np.load(emb_path), axis=0))
             continue
        audio, sr = load_audio(filepath)
        segments = energy_vad(audio, sr)
#        segments = pyannote_vad(filepath, sr)
        all_embeddings.append(np.expand_dims(extract_embeddings(audio, emb_path, sr, segments, emb_model, device), axis=0))

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings, filenames