import sys,os,json,time,traceback,math,subprocess,zipfile
from urllib.request import Request,urlopen
import numpy as np
import soundfile as sf
from models import Project,Line
from storage import ROOT,DATA,read_json,write_json

KEY=sys.argv[1] if len(sys.argv)>1 else ''
FOLDER=DATA/'jobs'/KEY
def progress(value,message,**extra):
    state=read_json(FOLDER/'status.json');state.update(state='running',progress=value,message=message,**extra);write_json(FOLDER/'status.json',state)

def ollama(payload):
    req=Request(os.environ.get('DRAMA_OLLAMA','http://127.0.0.1:11435')+'/api/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=600) as r:return json.load(r)

def script(p):
    from story import generate, estimate
    try:
        p.lines=generate(p,ollama,progress)
        p=Project.model_validate(p.model_dump())
        warnings=[]
        joined=''.join(l.text for l in p.lines)
        for key in p.settings.key_lines.splitlines():
            if key.strip() and key.strip() not in joined:warnings.append(f'主要セリフを確認してください：{key.strip()}')
        for c in p.cast:
            if c.enabled and not any(l.speaker==c.id for l in p.lines):warnings.append(f'{c.name}の発言がありません。追加してください。')
        write_json(FOLDER/'result.json',{'project':p.model_dump(),'warnings':warnings,'estimated_seconds':round(estimate(p))})
    finally:
        try:ollama({'model':p.settings.writer_model,'messages':[],'keep_alive':0,'stream':False})
        except Exception:pass

def write_synopsis(p):
    from story import synopsis
    try:
        progress(.1,'あらすじを作成中')
        write_json(FOLDER/'result.json',{'synopsis':synopsis(p,ollama)})
    finally:
        try:ollama({'model':p.settings.writer_model,'messages':[],'keep_alive':0,'stream':False})
        except Exception:pass


def stamp(seconds):
    ms=round(seconds*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'

def render(p,kind,line_id):
    # A cancelled writing job may have left a model resident in Ollama.
    for model in ('qwen3:8b','qwen3:4b'):
        try:ollama({'model':model,'messages':[],'keep_alive':0,'stream':False})
        except Exception:pass
    from audio_engine import Synthesizer,SR,stereo,mix,ffmpeg
    synth=Synthesizer();cast={c.id:c for c in p.cast if c.enabled};chosen=[l for l in p.lines if not line_id or l.id==line_id]
    pieces=[];timeline=[];cursor=0;cache_count=0
    for i,line in enumerate(chosen):
        progress(.03+.77*i/len(chosen),f'音声を生成中：{i+1}/{len(chosen)} · {cast[line.speaker].name}')
        a,cached=synth.line(line,cast[line.speaker],p.settings);cache_count+=int(cached)
        if kind=='preview':
            sf.write(FOLDER/'preview.wav',stereo(a,cast[line.speaker].pan,cast[line.speaker].gain).clip(-.98,.98),SR);return
        c=cast[line.speaker];pieces.append(stereo(a,c.pan,c.gain))
        timeline.append({**line.model_dump(),'name':c.name})
    progress(.83,'同時発話と会話の間を配置中')
    from story import timing
    positions,seconds=timing(chosen,[len(a)/SR for a in pieces])
    voice=np.zeros((round(seconds*SR),2),dtype=np.float32)
    for piece,row,(start,end) in zip(pieces,timeline,positions):
        offset=round(start*SR)
        voice[offset:offset+len(piece)]+=piece
        row.update(start=start,end=end)
    warnings=[];ratio=1.0
    if p.settings.fit_duration:
        ratio=len(voice)/SR/p.settings.duration
        if .8<=ratio<=1.25:
            sf.write(FOLDER/'unscaled.wav',voice,SR,subtype='FLOAT')
            subprocess.run([ffmpeg(),'-y','-v','error','-i',str(FOLDER/'unscaled.wav'),'-af',f'atempo={ratio},apad,atrim=duration={p.settings.duration}','-c:a','pcm_f32le',str(FOLDER/'scaled.wav')],check=True)
            voice,_=sf.read(FOLDER/'scaled.wav',dtype='float32',always_2d=True)
            for row in timeline:row['start']/=ratio;row['end']/=ratio
        else:
            warnings.append(f'自然さを保つため尺の強制調整を行いませんでした。台本の分量を調整してください（指定{p.settings.duration}秒、実音声{len(voice)/SR:.1f}秒）。');ratio=1
    progress(.89,'BGM・SEを配置してミックス中')
    audio,mix_info=mix(voice,timeline,p.settings);sf.write(FOLDER/'voices.wav',voice.clip(-.98,.98),SR,subtype='PCM_16');sf.write(FOLDER/'mix.wav',audio,SR,subtype='PCM_16')
    subprocess.run([ffmpeg(),'-y','-v','error','-i',str(FOLDER/'mix.wav'),'-c:a','libmp3lame','-b:a','192k',str(FOLDER/'mix.mp3')],check=True)
    from story import script_text
    (FOLDER/'script.txt').write_text(script_text(p),encoding='utf-8')
    (FOLDER/'subtitles.srt').write_text('\n\n'.join(f"{i+1}\n{stamp(r['start'])} --> {stamp(r['end'])}\n{r['name']}：{r['text']}" for i,r in enumerate(timeline)),encoding='utf-8')
    manifest={'synthetic':True,'seconds':len(audio)/SR,'sample_rate':SR,'cached_lines':cache_count,'tempo_factor':ratio,'warnings':warnings,'mix':mix_info,'timeline':timeline,'project':p.model_dump()}
    write_json(FOLDER/'manifest.json',manifest);write_json(FOLDER/'result.json',{'seconds':manifest['seconds'],'warnings':warnings,'cached_lines':cache_count})
    progress(.96,'書き出しファイルをまとめています')
    with zipfile.ZipFile(FOLDER/'bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['mix.mp3','mix.wav','voices.wav','script.txt','subtitles.srt','manifest.json']:z.write(FOLDER/name,name)
    for name in ['unscaled.wav','scaled.wav']:(FOLDER/name).unlink(missing_ok=True)

def main():
    inp=read_json(FOLDER/'input.json');p=Project.model_validate(inp['project']);progress(.01,'ローカルAIを準備しています')
    if inp['kind']=='synopsis':write_synopsis(p)
    elif inp['kind']=='script':script(p)
    else:render(p,inp['kind'],inp.get('line_id',''))
    s=read_json(FOLDER/'status.json');s.update(state='finishing',progress=.99,message='処理を終了しています');write_json(FOLDER/'status.json',s)

if __name__=='__main__':
    try:main()
    except Exception as e:
        traceback.print_exc();s=read_json(FOLDER/'status.json');s.update(state='failed',message=str(e)[:600]);write_json(FOLDER/'status.json',s);sys.exit(1)
