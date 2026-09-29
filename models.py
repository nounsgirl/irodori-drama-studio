from typing import Literal
from pydantic import BaseModel, Field, model_validator
import uuid

def uid(): return uuid.uuid4().hex

Emotion = Literal['自然','楽しい','穏やか','驚き','悲しい','怒り','緊張','ささやき']
Effect = Literal['auto','none','chime','bell','waves','rain','steps']

class Cast(BaseModel):
    id: str = Field(default_factory=uid, pattern=r'^[a-zA-Z0-9_-]{1,64}$')
    name: str = Field(min_length=1,max_length=40)
    personality: str = Field(default='',max_length=1200)
    role: str = Field(default='ゲスト',max_length=100)
    style: str = Field(default='自然で親しみやすい話し方',max_length=600)
    pace: float = Field(default=1,ge=.6,le=1.5)
    first_person: str = Field(default='私',max_length=40)
    called_as: str = Field(default='',max_length=200)
    addressing: str = Field(default='',max_length=1200)
    voice: str = Field(default='',max_length=64)
    enabled: bool = True
    gain: float = Field(default=0,ge=-18,le=12)
    pan: float = Field(default=0,ge=-1,le=1)

class Line(BaseModel):
    id: str = Field(default_factory=uid,pattern=r'^[a-zA-Z0-9_-]{1,64}$')
    speaker: str = Field(max_length=64)
    text: str = Field(min_length=1,max_length=500)
    emotion: Emotion = '自然'
    scene: str = Field(default='シーン1',max_length=100)
    pause: float = Field(default=.25,ge=0,le=4)
    simultaneous: bool = False
    se: Effect = 'auto'

class Settings(BaseModel):
    writer_model: Literal['qwen3:8b','qwen3:4b'] = 'qwen3:8b'
    duration: int = Field(default=300,ge=20,le=900)
    direction: str = Field(default='',max_length=2000)
    emotion: Emotion = '楽しい'
    keywords: str = Field(default='',max_length=1500)
    key_lines: str = Field(default='',max_length=2000)
    temperature: float = Field(default=.75,ge=.1,le=1.5)
    steps: int = Field(default=32,ge=10,le=60)
    seed: int = Field(default=42,ge=0,le=2147483647)
    voice_strength: float = Field(default=5,ge=1,le=8)
    speed: float = Field(default=1,ge=.75,le=1.3)
    reference_seconds: int = Field(default=20,ge=5,le=40)
    bgm: Literal['auto','warm','calm','tension','none','uploaded'] = 'auto'
    bgm_asset: str = Field(default='',max_length=64)
    bgm_db: float = Field(default=-25,ge=-45,le=-8)
    se_enabled: bool = True
    se_db: float = Field(default=-18,ge=-36,le=-6)
    ducking: bool = True
    fit_duration: bool = True

class Project(BaseModel):
    id: str = Field(default_factory=uid,pattern=r'^[a-f0-9]{32}$')
    title: str = Field(default='新しい音声ドラマ',min_length=1,max_length=100)
    summary: str = Field(default='',max_length=5000)
    synopsis: str = Field(default='',max_length=8000)
    settings: Settings = Field(default_factory=Settings)
    cast: list[Cast] = Field(default_factory=list,max_length=12)
    lines: list[Line] = Field(default_factory=list,max_length=300)
    @model_validator(mode='after')
    def unique(self):
        for seq in [self.cast,self.lines]:
            ids=[x.id for x in seq]
            if len(ids)!=len(set(ids)): raise ValueError('識別子が重複しています')
        valid={c.id for c in self.cast}
        if any(l.speaker not in valid for l in self.lines): raise ValueError('台本に未登録の話者がいます')
        return self
