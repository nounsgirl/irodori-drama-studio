import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from models import Project, Cast, Line
from story import estimate, timing, repeated, remove_repeats, parse_lines, generate, script_text


class StoryTests(unittest.TestCase):
    def project(self):
        return Project(cast=[Cast(id='a',name='A'),Cast(id='b',name='B',pace=.75)],synopsis='鍵を探し、見つけて帰る。')

    def test_legacy_defaults_and_fields(self):
        p=self.project()
        self.assertEqual(p.cast[0].first_person,'私')
        p.cast[0].called_as='先輩'
        self.assertEqual(Project.model_validate(p.model_dump()).cast[0].called_as,'先輩')
        for pace in [0,2]:
            with self.assertRaises(ValueError):Cast(name='A',pace=pace)

    def test_pace_and_simultaneous_group(self):
        p=self.project()
        p.lines=[Line(speaker='a',text='あ'*65,pause=0),Line(speaker='b',text='い'*65,pause=0,simultaneous=True),Line(speaker='a',text='う'*65,pause=0)]
        self.assertAlmostEqual(estimate(p),65/(6.5*.75)+10)
        rows,end=timing(p.lines,[10,15,5])
        self.assertEqual(rows,[(0,10),(0,15),(15,20)])
        self.assertEqual(end,20)
        p.lines[0].simultaneous=True
        self.assertEqual(timing(p.lines,[10,15,5])[1],20)

    def test_duplicate_and_export(self):
        p=self.project();line=Line(speaker='a',text='それでは明日の朝に駅前で集合しましょう。')
        self.assertTrue(repeated([line],[line]))
        self.assertFalse(repeated([line,Line(speaker='b',text=line.text,simultaneous=True)]))
        self.assertFalse(repeated([Line(speaker='b',text='あはは！')]*2))
        p.lines=[line,Line(speaker='b',text='了解！',simultaneous=True)]
        self.assertIn('【前の発言と同時】',script_text(p))

    def test_redundant_lines_removed_but_chorus_kept(self):
        a=Line(speaker='a',text='それでは明日の朝に駅前で集合しましょう。')
        b=Line(speaker='b',text=a.text,simultaneous=True)
        self.assertEqual(len(remove_repeats([a,b])),2)
        self.assertEqual(len(remove_repeats([a,a.model_copy()])),1)
        later=Line(speaker='b',text='別の場所まで探しに行きます。',simultaneous=True)
        result=remove_repeats([a,later],[a])
        self.assertEqual(len(result),1)
        self.assertFalse(result[0].simultaneous)

    def test_writer_alias_and_emotion_normalization(self):
        p=self.project()
        rows=parse_lines({'lines':[{'speaker':'A','text':'見つかったよ。','emotion':'温かく'}]},p.cast,'発見')
        self.assertEqual(rows[0].speaker,'a')
        self.assertEqual(rows[0].emotion,'穏やか')
        with self.assertRaises(ValueError):parse_lines({'lines':{}},p.cast,'発見')

    def test_retries_short_script_without_padding(self):
        p=self.project();p.settings.duration=20;p.cast=p.cast[:1]
        outputs=[{'scenes':['鍵を見つける']},{'lines':[{'speaker':'a','text':'短い'}]}, {'lines':[{'speaker':'a','text':'あ'*128}]}]
        with patch('story.ask',side_effect=outputs) as ask:
            result=generate(p,None,lambda *args:None)
        self.assertEqual(ask.call_count,3)
        self.assertEqual(len(result),2)
        self.assertTrue(18<=estimate(p,result)<=22)

    def test_synopsis_job_accepts_cast_without_voice(self):
        import server
        p=self.project()
        with tempfile.TemporaryDirectory() as folder,patch.object(server,'DATA',Path(folder)),patch.dict(server.processes,{},clear=True),patch.object(server.subprocess,'Popen'),patch.object(server.threading,'Thread') as thread:
            (Path(folder)/'jobs').mkdir()
            response=server.start_job('synopsis',p)
            thread.call_args.kwargs['args'][2].close()
            data=json.loads((Path(folder)/'jobs'/response['id']/'input.json').read_text(encoding='utf-8'))
            self.assertEqual(data['kind'],'synopsis')
            p.synopsis=''
            with self.assertRaises(server.HTTPException) as error:server.start_job('script',p)
            self.assertEqual(error.exception.status_code,400)

    def test_failed_quality_keeps_original(self):
        p=self.project();p.settings.duration=20;p.lines=[Line(speaker='a',text='元の台本')]
        with patch('story.ask',side_effect=[{'scenes':['場面']}] + [{'lines':[{'speaker':'a','text':'短い'}]}]*4):
            with self.assertRaises(ValueError):generate(p,None,lambda *args:None)
        self.assertEqual(p.lines[0].text,'元の台本')

    def test_render_overlaps_audio_and_exports_timing(self):
        import numpy as np
        import soundfile as sf
        import worker
        from audio_engine import SR
        p=self.project();p.settings.fit_duration=False;p.settings.bgm='none';p.settings.se_enabled=False
        p.lines=[Line(speaker='a',text='はい',pause=0),Line(speaker='b',text='はい',simultaneous=True,pause=0),Line(speaker='a',text='続き',pause=0)]
        clips=[(np.ones(SR,dtype=np.float32)*.1,False),(np.ones(SR*2,dtype=np.float32)*.1,False),(np.ones(SR,dtype=np.float32)*.1,False)]
        with tempfile.TemporaryDirectory() as folder, patch.object(worker,'FOLDER',Path(folder)), patch.object(worker,'ollama'),patch.object(worker,'progress'),patch('audio_engine.Synthesizer') as synth:
            synth.return_value.line.side_effect=clips
            worker.render(p,'render','')
            a,sr=sf.read(Path(folder)/'voices.wav')
            self.assertEqual(len(a),3*SR)
            self.assertAlmostEqual(float(a[100,0]),2*float(a[SR+100,0]),places=3)
            manifest=json.loads((Path(folder)/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual([r['start'] for r in manifest['timeline']],[0,0,2])
            self.assertIn('【前の発言と同時】',(Path(folder)/'script.txt').read_text(encoding='utf-8'))

if __name__=='__main__':unittest.main()
