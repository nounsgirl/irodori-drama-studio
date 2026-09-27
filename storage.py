from pathlib import Path
import json, os, uuid, hashlib, shutil
from models import Project, Cast, Line

ROOT=Path(__file__).resolve().parent
DATA=Path(os.environ.get('DRAMA_DATA_DIR',str(ROOT/'data'))).expanduser().resolve()
for name in ['projects','assets','jobs','cache','exports']:(DATA/name).mkdir(parents=True,exist_ok=True)

def write_json(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def assets():return [read_json(f) for f in (DATA/'assets').glob('*.json')]

def asset_path(asset_id):
    if not asset_id or not all(c in '0123456789abcdef' for c in asset_id) or len(asset_id)!=32:raise ValueError('音声素材が未設定です')
    meta=read_json(DATA/'assets'/f'{asset_id}.json')
    return DATA/'assets'/meta['file']

def import_asset(path,name=None,kind='voice',fixed_id=None):
    import soundfile as sf
    path=Path(path)
    try:info=sf.info(path)
    except Exception:
        # Decode containers unsupported by libsndfile (for example M4A).
        import subprocess
        ff=shutil.which('ffmpeg')
        if not ff:raise ValueError('この形式を読むにはffmpegが必要です。WAVまたはMP3を選んでください。')
        converted=DATA/'assets'/f'{uuid.uuid4().hex}.wav'
        subprocess.run([ff,'-v','error','-y','-i',str(path),'-t','7201','-ac','1','-ar','48000',str(converted)],check=True,capture_output=True)
        try:return import_asset(converted,name or path.name,kind,fixed_id)
        finally:converted.unlink(missing_ok=True)
    if not 0.5<=info.duration<=7200:raise ValueError('音声は0.5秒〜2時間にしてください')
    key=fixed_id or uuid.uuid4().hex;dest=DATA/'assets'/f'{key}{path.suffix.lower()}'
    if path.resolve()!=dest.resolve():shutil.copyfile(path,dest)
    meta={'id':key,'name':name or path.name,'kind':kind,'file':dest.name,'seconds':round(info.duration,2),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    write_json(DATA/'assets'/f'{key}.json',meta);return meta

def bootstrap():
    if list((DATA/'projects').glob('*.json')):return
    # Start with an empty project. Assets are only added through explicit uploads.
    p=Project(cast=[Cast(name='話者A',role='進行役'),Cast(name='話者B',role='ゲスト')])
    write_json(DATA/'projects'/f'{p.id}.json',p.model_dump())
