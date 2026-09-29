import os,sys,math,hashlib,json,subprocess,shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from storage import ROOT,DATA,asset_path,read_json

SR=48000
if os.environ.get('IRODORI_HOME'):
    sys.path.insert(0,str(Path(os.environ['IRODORI_HOME']).expanduser().resolve()))
os.environ['HF_HUB_OFFLINE']='1'

def split_text(text,max_chars=90):
    chunks=[]
    while len(text)>max_chars:
        window=text[:max_chars]
        cut=max((window.rfind(mark)+1 for mark in '。！？!?\n'),default=0)
        if not cut:cut=max((window.rfind(mark)+1 for mark in '、, '),default=0)
        cut=cut or max_chars;chunks.append(text[:cut]);text=text[cut:]
    if text:chunks.append(text)
    return chunks

def ffmpeg():
    p=shutil.which(os.environ.get('FFMPEG_BINARY','ffmpeg'))
    if not p:raise RuntimeError('ffmpegが見つかりません。PATHまたはFFMPEG_BINARYを設定してください。')
    return p

def load_audio(path,sr=SR):
    try:a,rate=sf.read(path,dtype='float32',always_2d=True)
    except Exception:
        result=subprocess.run([ffmpeg(),'-v','error','-i',str(path),'-f','f32le','-ac','1','-ar',str(sr),'-'],capture_output=True,check=True)
        return np.frombuffer(result.stdout,dtype=np.float32).copy()
    a=a.mean(axis=1);g=math.gcd(rate,sr)
    return resample_poly(a,sr//g,rate//g).astype(np.float32) if rate!=sr else a

def reference(cast,seconds):
    meta=read_json(DATA/'assets'/f'{cast.voice}.json')
    key=hashlib.sha256(f"{meta['sha256']}:{seconds}".encode()).hexdigest()
    out=DATA/'cache'/f'ref_{key}.wav'
    if out.exists():return out
    a=load_audio(asset_path(cast.voice));n=min(len(a),seconds*SR);best=(-1,0)
    for start in range(0,max(1,len(a)-n+1),2*SR):
        x=a[start:start+n];block=4800
        if len(x)<block:continue
        rms=np.sqrt(np.mean(x[:len(x)//block*block].reshape(-1,block)**2,axis=1)+1e-10)
        score=float(np.mean(rms>.012))-float(np.mean(abs(x)>.99))*3
        if score>best[0]:best=(score,start)
    if best[0]<.05:raise ValueError(f'{cast.name}の音声がほぼ無音です。別の素材を選んでください。')
    x=a[best[1]:best[1]+n];x*=min(.12/max(float(np.sqrt(np.mean(x*x))),1e-6),.9/max(float(abs(x).max()),1e-6))
    sf.write(out,x,SR);return out

class Synthesizer:
    def __init__(self):self.runtime=None
    def line(self,line,cast,settings):
        meta=read_json(DATA/'assets'/f'{cast.voice}.json')
        params={'text':line.text,'emotion':line.emotion,'style':cast.style,'voice':meta['sha256'],'ref':settings.reference_seconds,'steps':settings.steps,'seed':settings.seed,'strength':settings.voice_strength,'speed':settings.speed*cast.pace,'version':3}
        key=hashlib.sha256(json.dumps(params,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        path=DATA/'cache'/f'{key}.wav'
        if path.exists():return load_audio(path),True
        import torch
        from irodori_tts.inference_runtime import RuntimeKey,SamplingRequest,get_cached_runtime
        if self.runtime is None:
            from huggingface_hub.constants import HF_HUB_CACHE
            explicit=os.environ.get('IRODORI_CHECKPOINT')
            snapshots=Path(HF_HUB_CACHE)/'models--Aratako--Irodori-TTS-v4.1-Small-Quantized/snapshots'
            checkpoint=Path(explicit).expanduser() if explicit else next(snapshots.glob('*/int8-weight-only/model.safetensors'),None)
            if not checkpoint or not checkpoint.is_file():raise RuntimeError('Irodoriモデルが見つかりません。IRODORI_CHECKPOINTまたはモデルキャッシュを設定してください。')
            self.runtime,_=get_cached_runtime(RuntimeKey(checkpoint=str(checkpoint),model_device='cuda',model_precision='bf16',codec_device='cpu',codec_precision='fp32'))
        ref=reference(cast,settings.reference_seconds);parts=[]
        for text in split_text(line.text,90):
            if not text.strip():continue
            caption=f'一人の話者による自然な日本語の会話。{cast.style}。感情は{line.emotion}。明瞭な発話。音楽、背景音、効果音のないドライな声のみの録音。'
            result=self.runtime.synthesize(SamplingRequest(text=text,caption=caption,ref_wav=str(ref),max_ref_seconds=settings.reference_seconds,num_steps=settings.steps,seed=settings.seed,seconds=min(25,max(2,len(text)/6.5)),cfg_scale_speaker=settings.voice_strength,cfg_scale_caption=3.0),log_fn=lambda _:None)
            a=result.audio.detach().cpu().float().reshape(-1).numpy()
            if result.sample_rate!=SR:
                g=math.gcd(result.sample_rate,SR);a=resample_poly(a,SR//g,result.sample_rate//g)
            if not np.isfinite(a).all() or np.max(abs(a))<.0001:raise RuntimeError('音声生成が無音になりました。再実行してください。')
            block=480;levels=np.array([np.sqrt(np.mean(a[k:k+block]**2)) for k in range(0,len(a),block)])
            active=np.where(levels>max(.002,float(levels.max())*.012))[0]
            if len(active):a=a[max(0,active[0]*block-2400):min(len(a),(active[-1]+1)*block+4800)]
            voiced=a[abs(a)>.01];rms=float(np.sqrt(np.mean(voiced**2))) if len(voiced) else .1
            a*=min(.12/max(rms,1e-6),.88/max(float(abs(a).max()),1e-6))
            fade=min(240,len(a)//2);a[:fade]*=np.linspace(0,1,fade);a[-fade:]*=np.linspace(1,0,fade)
            if parts:parts.append(np.zeros(4800,dtype=np.float32))
            parts.append(a)
        audio=np.concatenate(parts).astype(np.float32)
        if settings.speed*cast.pace!=1:
            raw=DATA/'cache'/f'{key}.raw.wav';sf.write(raw,audio,SR)
            subprocess.run([ffmpeg(),'-v','error','-y','-i',str(raw),'-af',f'atempo={math.sqrt(settings.speed*cast.pace)},atempo={math.sqrt(settings.speed*cast.pace)}','-ar',str(SR),str(path)],check=True)
            raw.unlink();audio=load_audio(path)
        else:sf.write(path,audio,SR)
        return audio,False

def stereo(audio,pan=0,gain=0):
    angle=(pan+1)*np.pi/4;g=10**(gain/20)
    return np.stack([audio*np.cos(angle)*g,audio*np.sin(angle)*g],axis=1).astype(np.float32)

def bgm_track(n,style,seed):
    # Original synthesized accompaniment, no external music samples.
    out=np.zeros(n,dtype=np.float32);tempo={'warm':86,'calm':64,'tension':76}[style];beat=60/tempo
    chords={'warm':[(60,64,67),(57,60,64),(53,57,60),(55,59,62)],'calm':[(60,64,67),(65,69,72),(57,60,64),(55,62,67)],'tension':[(57,60,64),(53,56,60),(50,53,57),(52,56,59)]}[style]
    rng=np.random.default_rng(seed)
    for k,start in enumerate(np.arange(0,n/SR,beat)):
        chord=chords[(k//8)%4];note=chord[k%3]+(12 if k%4==3 else 0);freq=440*2**((note-69)/12)
        size=min(round(beat*2*SR),n-int(start*SR));t=np.arange(size,dtype=np.float32)/SR
        tone=(np.sin(2*np.pi*freq*t)+.2*np.sin(4*np.pi*freq*t))*np.exp(-t*3)*np.minimum(t/.012,1)
        j=int(start*SR);out[j:j+size]+=(tone*.3).astype(np.float32)
        if k%4==0:
            bass=440*2**((chord[0]-24-69)/12);out[j:j+size]+=(.15*np.sin(2*np.pi*bass*t)*np.exp(-t*1.8)*np.minimum(t/.02,1)).astype(np.float32)
    out/=max(float(np.sqrt(np.mean(out*out))),1e-6)
    fade=min(3*SR,n//2);out[:fade]*=np.linspace(0,1,fade);out[-fade:]*=np.linspace(1,0,fade)
    return out

def effect(name,seed):
    duration={'chime':1.2,'bell':1.5,'waves':3,'rain':3,'steps':1.2}[name];t=np.arange(round(duration*SR),dtype=np.float32)/SR
    if name in ('waves','rain'):
        rng=np.random.default_rng(seed);noise=rng.normal(0,1,len(t)).astype(np.float32)
        from scipy.signal import lfilter
        x=lfilter([.03],[1,-.97],noise).astype(np.float32) if name=='waves' else noise*.12
        x*=np.sin(np.pi*t/duration)**2
    elif name=='steps':
        rng=np.random.default_rng(seed);phase=t%.4;x=rng.normal(0,1,len(t))*np.exp(-phase*50)*.25
    else:
        x=np.zeros_like(t)
        for i,f in enumerate(([660,880,1100] if name=='chime' else [880,1763,2350])):
            u=np.maximum(t-i*.11,0);x+=np.sin(2*np.pi*f*u)*np.exp(-u*5)*(t>=i*.11)/(i+1)
    x=x.astype(np.float32);x/=max(float(abs(x).max()),1e-6);return x

def choose_effect(text,scene_change):
    if any(w in text for w in ('雨','傘')):return 'rain'
    if any(w in text for w in ('波音','水音','波打つ')):return 'waves'
    if any(w in text for w in ('到着','ドア','お店に入')):return 'bell'
    return 'chime' if scene_change else 'none'

def mix(voice,timeline,settings):
    n=len(voice);out=voice.copy();events=[]
    style=settings.bgm
    if style=='auto':style='tension' if settings.emotion in ('緊張','怒り') else 'calm' if settings.emotion in ('悲しい','穏やか','ささやき') else 'warm'
    if style!='none':
        if style=='uploaded':
            x=load_audio(asset_path(settings.bgm_asset));x=np.tile(x,math.ceil(n/len(x)))[:n]
            x=x/max(float(np.sqrt(np.mean(x*x))),1e-6)
            fade=min(2*SR,n//2);x[:fade]*=np.linspace(0,1,fade);x[-fade:]*=np.linspace(1,0,fade)
        else:x=bgm_track(n,style,settings.seed)
        envelope=np.ones(n,dtype=np.float32)
        if settings.ducking:
            for row in timeline:
                start=max(0,int(row['start']*SR));end=min(n,int(row['end']*SR));envelope[start:end]=.32
                ramp=min(int(.18*SR),start)
                if ramp:envelope[start-ramp:start]=np.minimum(envelope[start-ramp:start],np.linspace(1,.32,ramp))
                ramp=min(int(.3*SR),n-end)
                if ramp:envelope[end:end+ramp]=np.minimum(envelope[end:end+ramp],np.linspace(.32,1,ramp))
        out+=stereo(x*envelope*10**(settings.bgm_db/20))
    previous_scene=None;last_auto=-20
    if settings.se_enabled:
        for i,row in enumerate(timeline):
            name=row['se'];change=row['scene']!=previous_scene;previous_scene=row['scene']
            if name=='auto':
                if row['start']-last_auto<12:continue
                name=choose_effect(row['text'],change)
                if name!='none':last_auto=row['start']
            if name=='none':continue
            x=effect(name,settings.seed+i)*10**(settings.se_db/20);start=max(0,int((row['start']-.2)*SR));length=min(len(x),n-start)
            if length>0:out[start:start+length]+=stereo(x[:length]);events.append({'effect':name,'start':start/SR})
    peak=float(abs(out).max());gain=min(1,.96/max(peak,1e-6));out*=gain
    return out,{'bgm':style,'effects':events,'limiting_gain':gain,'peak':float(abs(out).max())}
