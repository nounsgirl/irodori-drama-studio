from pathlib import Path
import os,sys,json,uuid,subprocess,threading,time,traceback
from urllib.request import urlopen
from fastapi import FastAPI,HTTPException,UploadFile,File,Request
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from models import Project
from storage import ROOT,DATA,read_json,write_json,assets,asset_path,import_asset,bootstrap

app=FastAPI(title='Irodori Drama Studio',docs_url=None,redoc_url=None)
lock=threading.Lock();processes={}
OLLAMA=os.environ.get('DRAMA_OLLAMA','http://127.0.0.1:11435')

@app.middleware('http')
async def local_only(request:Request,call_next):
    from urllib.parse import urlsplit
    if request.url.hostname not in ('127.0.0.1','localhost'):return JSONResponse({'detail':'ローカル接続専用です'},403)
    origin=request.headers.get('origin')
    if request.method not in ('GET','HEAD') and origin and origin!=str(request.base_url).rstrip('/'):
        return JSONResponse({'detail':'別のサイトからの操作は許可されません'},403)
    return await call_next(request)

@app.on_event('startup')
def startup():
    bootstrap()
    for f in (DATA/'jobs').glob('*/status.json'):
        s=read_json(f)
        if s['state'] in ('running','queued','finishing'):s.update(state='failed',message='アプリが終了したため中断しました。再実行できます。');write_json(f,s)

@app.get('/api/health')
def health():
    models=[];ready=False
    try:
        with urlopen(OLLAMA+'/api/tags',timeout=2) as r: models=[m['name'] for m in json.load(r).get('models',[])]
        ready='qwen3:8b' in models
    except Exception:pass
    return {'ok':True,'writer_ready':ready,'models':models,'engine':'Irodori v4.1','writer':'Qwen3 8B · ローカル'}

@app.get('/api/projects')
def projects():return sorted([{'id':(p:=read_json(f))['id'],'title':p['title'],'lines':len(p['lines']),'modified':f.stat().st_mtime} for f in (DATA/'projects').glob('*.json')],key=lambda x:-x['modified'])

def project_file(key):
    if len(key)!=32 or any(c not in '0123456789abcdef' for c in key):raise HTTPException(404,'見つかりません')
    return DATA/'projects'/f'{key}.json'

@app.get('/api/projects/{key}')
def get_project(key:str):
    f=project_file(key)
    if not f.exists():raise HTTPException(404,'企画が見つかりません')
    return Project.model_validate(read_json(f)).model_dump()

@app.put('/api/projects/{key}')
def save_project(key:str,p:Project):
    if key!=p.id:raise HTTPException(400,'企画IDが一致しません')
    write_json(project_file(key),p.model_dump());return p

@app.get('/api/assets')
def list_assets():return assets()

@app.post('/api/assets')
async def upload(kind:str='voice',file:UploadFile=File(...)):
    if kind not in ('voice','bgm'):raise HTTPException(400,'素材の種類が不正です')
    suffix=Path(file.filename or '').suffix.lower()
    if suffix not in ('.wav','.mp3','.flac','.ogg','.m4a'):raise HTTPException(400,'WAV・MP3・FLAC・OGG・M4Aに対応しています')
    key=uuid.uuid4().hex;path=DATA/'assets'/f'{key}{suffix}';size=0
    try:
        with path.open('wb') as out:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>150*1024*1024:raise ValueError('150MB以下の音声を選んでください')
                out.write(chunk)
        result=import_asset(path,file.filename,kind,key)
        if result['file']!=path.name:path.unlink(missing_ok=True)
        return result
    except Exception as e:
        path.unlink(missing_ok=True);raise HTTPException(400,str(e))

@app.get('/api/assets/{key}/audio')
def asset_audio(key:str):
    try:return FileResponse(asset_path(key))
    except Exception:raise HTTPException(404,'素材が見つかりません')

def finish_job(key,proc,log):
    proc.wait();log.close()
    with lock:
        processes.pop(key,None);path=DATA/'jobs'/key/'status.json';s=read_json(path)
        if s['state']=='finishing' and proc.returncode==0:
            s.update(state='done',progress=1,message='完成しました');write_json(path,s)
        elif s['state'] not in ('done','failed','cancelled'):
            s.update(state='failed',message='処理が終了しました。ログを確認してください。');write_json(path,s)

@app.post('/api/jobs/{kind}')
def start_job(kind:str,p:Project,line_id:str=''):
    if kind not in ('synopsis','script','render','preview'):raise HTTPException(400,'不明な処理です')
    enabled=[c for c in p.cast if c.enabled]
    if not enabled:raise HTTPException(400,'参加者を1人以上選んでください')
    if kind=='script' and not p.synopsis.strip():raise HTTPException(400,'先にあらすじを作成・編集してください')
    if kind in ('render','preview'):
        chosen=[l for l in p.lines if not line_id or l.id==line_id]
        if not chosen:raise HTTPException(400,'台本にセリフを入力してください')
        valid={c.id:c for c in enabled}
        for line in chosen:
            if line.speaker not in valid:raise HTTPException(400,'不参加の話者のセリフがあります')
            try:asset_path(valid[line.speaker].voice)
            except Exception:raise HTTPException(400,f'{valid[line.speaker].name}のマスター音声を選んでください')
    with lock:
        if processes:raise HTTPException(409,'別の処理が進行中です。終了を待つか中止してください。')
        key=uuid.uuid4().hex;folder=DATA/'jobs'/key;folder.mkdir()
        write_json(folder/'input.json',{'project':p.model_dump(),'kind':kind,'line_id':line_id})
        write_json(folder/'status.json',{'id':key,'kind':kind,'project_id':p.id,'state':'queued','progress':0,'message':'準備しています','created':time.time()})
        log=(folder/'worker.log').open('w',encoding='utf-8')
        env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
        try:proc=subprocess.Popen([sys.executable,str(ROOT/'worker.py'),key],cwd=ROOT,stdout=log,stderr=log,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except Exception:log.close();raise
        processes[key]=proc;threading.Thread(target=finish_job,args=(key,proc,log),daemon=True).start()
    return {'id':key}

@app.get('/api/jobs')
def job_list():return sorted([read_json(f) for f in (DATA/'jobs').glob('*/status.json')],key=lambda x:-x.get('created',0))[:50]

def job_folder(key):
    if len(key)!=32 or any(c not in '0123456789abcdef' for c in key):raise HTTPException(404,'処理が見つかりません')
    f=DATA/'jobs'/key
    if not f.exists():raise HTTPException(404,'処理が見つかりません')
    return f

@app.get('/api/jobs/{key}')
def job(key:str):return read_json(job_folder(key)/'status.json')

@app.post('/api/jobs/{key}/cancel')
def cancel(key:str):
    folder=job_folder(key)
    with lock:
        proc=processes.get(key)
        if proc:proc.terminate();proc.wait(timeout=10)
        s=read_json(folder/'status.json')
        if s['state'] not in ('done','failed'):s.update(state='cancelled',message='中止しました。生成済み音声は次回再利用されます。');write_json(folder/'status.json',s)
    return s

@app.get('/api/jobs/{key}/files/{name}')
def job_file(key:str,name:str):
    if name not in ('result.json','mix.mp3','mix.wav','voices.wav','preview.wav','script.txt','subtitles.srt','bundle.zip','worker.log','manifest.json'):raise HTTPException(404)
    f=job_folder(key)/name
    if not f.exists():raise HTTPException(404,'まだ作成されていません')
    return FileResponse(f,filename=name if name.endswith(('.zip','.txt','.srt')) else None)

app.mount('/',StaticFiles(directory=ROOT/'static',html=True),name='ui')

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8787,access_log=False)
