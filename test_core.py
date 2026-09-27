import unittest,tempfile,json,zipfile,os
from pathlib import Path
_isolated_data=tempfile.TemporaryDirectory(prefix='studio-test-')
os.environ['DRAMA_DATA_DIR']=_isolated_data.name
import numpy as np
from fastapi.testclient import TestClient
from models import Project,Cast,Line,Settings
from audio_engine import mix,SR,stereo,effect
from server import app

class CoreTests(unittest.TestCase):
    def test_clean_bootstrap(self):
        from storage import bootstrap,DATA,read_json
        bootstrap()
        projects=list((DATA/'projects').glob('*.json'))
        self.assertEqual(len(projects),1)
        p=read_json(projects[0])
        self.assertEqual(p['lines'],[])
        self.assertEqual(p['summary'],'')
        self.assertEqual([c['name'] for c in p['cast']],['話者A','話者B'])
        self.assertTrue(all(c['voice']=='' and c['personality']=='' for c in p['cast']))
        self.assertEqual(list((DATA/'assets').iterdir()),[])
    def test_split_preserves_text(self):
        from audio_engine import split_text
        text='これは一般的なテスト文です。' * 15
        parts=split_text(text,40)
        self.assertEqual(''.join(parts),text)
        self.assertTrue(all(len(p)<=40 for p in parts))
    def test_unknown_speaker_and_duplicate_rejected(self):
        with self.assertRaises(ValueError):Project(cast=[Cast(id='a',name='A')],lines=[Line(speaker='x',text='不明')])
        with self.assertRaises(ValueError):Project(cast=[Cast(id='a',name='A'),Cast(id='a',name='B')])
    def test_bounds(self):
        for kwargs in ({'duration':0},{'steps':1000},{'bgm_db':30},{'speed':0}):
            with self.assertRaises(ValueError):Settings(**kwargs)
    def test_mixer_ducking_and_switches(self):
        voice=np.zeros((4*SR,2),dtype=np.float32)
        timeline=[{'start':1,'end':3,'se':'none','text':'こんにちは','scene':'1'}]
        off,_=mix(voice,timeline,Settings(bgm='none',se_enabled=False));self.assertTrue(np.array_equal(off,voice))
        full,_=mix(voice,timeline,Settings(bgm='warm',ducking=False,se_enabled=False))
        duck,_=mix(voice,timeline,Settings(bgm='warm',ducking=True,se_enabled=False))
        self.assertLess(np.mean(duck[SR:3*SR]**2),np.mean(full[SR:3*SR]**2)*.12)
        self.assertGreater(np.max(abs(full)),0)
    def test_effect_schedule_and_clipping(self):
        voice=np.ones((4*SR,2),dtype=np.float32)*.95
        timeline=[{'start':0,'end':2,'se':'rain','text':'雨','scene':'1'},{'start':2,'end':3,'se':'none','text':'静か','scene':'1'}]
        out,meta=mix(voice,timeline,Settings(bgm='warm'))
        self.assertEqual(len(meta['effects']),1);self.assertEqual(meta['effects'][0]['effect'],'rain')
        self.assertTrue(np.isfinite(out).all());self.assertLessEqual(float(abs(out).max()),.961)
    def test_local_api_and_validation(self):
        c=TestClient(app,base_url='http://127.0.0.1:8787')
        r=c.put('/api/projects/'+'a'*32,json={'id':'a'*32,'settings':{'duration':-1}});self.assertEqual(r.status_code,422)
        self.assertEqual(c.post('/api/jobs/script',json={},headers={'Origin':'https://evil.example'}).status_code,403)
        self.assertEqual(c.get('/api/jobs/invalid').status_code,404)
        self.assertEqual(c.get('/api/assets/invalid/audio').status_code,404)
        self.assertEqual(c.post('/api/jobs/script',json={'cast':[]}).status_code,400)
    def test_pan(self):
        a=np.ones(48,dtype=np.float32);self.assertTrue(np.allclose(stereo(a,-1)[:,1],0));self.assertTrue(np.allclose(stereo(a,1)[:,0],0,atol=1e-7))

if __name__=='__main__':unittest.main()
