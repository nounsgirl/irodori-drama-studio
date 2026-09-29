"""Story planning, duration estimates and editable script exports."""
import json
import math
import re
from difflib import SequenceMatcher
from models import Line


def timing(lines, durations):
    rows = []
    cursor = 0.0
    group_start = 0.0
    for line, duration in zip(lines, durations):
        if not line.simultaneous or not rows:
            group_start = cursor
        start = group_start
        end = start + duration
        cursor = max(cursor, end + line.pause)
        rows.append((start, end))
    return rows, cursor


def estimate(p, lines=None):
    cast = {c.id: c for c in p.cast}
    chosen = p.lines if lines is None else lines
    durations = [len(l.text) / (6.5 * p.settings.speed * cast[l.speaker].pace) for l in chosen]
    return timing(chosen, durations)[1]


def repeated(lines, previous=()):
    seen = [(re.sub(r'\W', '', l.text), l.speaker, -1) for l in previous]
    group = 0
    for line in lines:
        if not line.simultaneous:
            group += 1
        text = re.sub(r'\W', '', line.text)
        if len(text) >= 12 and any(SequenceMatcher(None, text, old).ratio() > .83
                                  and not (line.simultaneous and old_group == group and speaker != line.speaker)
                                  for old, speaker, old_group in seen):
            return True
        seen.append((text, line.speaker, group))
    return False


def premise(p):
    return {'title': p.title, 'idea': p.summary, 'synopsis': p.synopsis,
            'direction': p.settings.direction, 'emotion': p.settings.emotion,
            'keywords': p.settings.keywords, 'required_dialogue': p.settings.key_lines,
            'cast': [{k: getattr(c, k) for k in ('id', 'name', 'personality', 'role', 'style', 'pace', 'first_person', 'called_as', 'addressing')} for c in p.cast if c.enabled]}


def remove_repeats(lines, previous=()):
    """Cut redundant dialogue before asking the writer to fill any duration gap."""
    result = []
    needs_anchor = False
    for original in lines:
        line = original.model_copy()
        if not line.simultaneous:
            needs_anchor = False
        if needs_anchor:
            line.simultaneous = False
        if repeated(result + [line], previous):
            if not line.simultaneous:
                needs_anchor = True
            continue
        result.append(line)
        needs_anchor = False
    return result


def parse_lines(parsed, cast, scene):
    aliases = {c.name: c.id for c in cast}
    aliases.update({c.called_as: c.id for c in cast if c.called_as})
    allowed = {'自然','楽しい','穏やか','驚き','悲しい','怒り','緊張','ささやき'}
    synonyms = {'温かく':'穏やか','温かい':'穏やか','喜び':'楽しい','嬉しい':'楽しい','不安':'緊張','驚いた':'驚き'}
    rows = parsed.get('lines', [])
    if not isinstance(rows, list):
        raise ValueError('linesにはセリフの一覧を指定してください。')
    result = []
    for item in rows:
        emotion = synonyms.get(item.get('emotion'), item.get('emotion', '自然'))
        speaker = item.get('speaker')
        result.append(Line(speaker=aliases.get(speaker, speaker), text=item.get('text'),
                           emotion=emotion if emotion in allowed else '自然', scene=scene,
                           simultaneous=item.get('simultaneous', False)))
    return result


def ask(p, ollama, prompt, schema, seed=0):
    if 'lines' in schema['properties']:
        example = {'lines': [{'speaker': c.id, 'text': 'ここには発声するセリフを書きます。', 'emotion': '自然', 'simultaneous': False} for c in p.cast if c.enabled][:2]}
    elif 'scenes' in schema['properties']:
        example = {'scenes': ['最初の場面の具体的な出来事', '次の場面の具体的な出来事']}
    else:
        example = {'synopsis': 'ここには導入から結末までのあらすじを書きます。'}
    response = ollama({'model': p.settings.writer_model, 'messages': [
        {'role': 'system', 'content': 'あなたは日本語の会話劇の脚本家です。各発言で出来事を進め、前の発言の言い換えや同意だけで尺を埋めません。登場人物の一人称と呼び方を厳守します。回答は指定JSON。'},
        {'role': 'user', 'content': prompt + '\n出力形式の例（内容は今回の場面に書き換える）：' + json.dumps(example, ensure_ascii=False)}], 'format': 'json', 'think': True, 'stream': False,
        'keep_alive': '5m', 'options': {'temperature': p.settings.temperature, 'num_ctx': 16384, 'num_predict': 6000, 'seed': p.settings.seed + seed}})
    return json.loads(response['message']['content'])


def synopsis(p, ollama):
    schema = {'type': 'object', 'properties': {'synopsis': {'type': 'string'}}, 'required': ['synopsis']}
    result = ask(p, ollama, f'{p.settings.duration}秒の作品のあらすじを日本語で作成。導入、具体的な出来事、転機、結末を順に描く。話題の反復で尺を埋めない。キャラクターの関係と変化を含める。300〜1200文字。\n' + json.dumps(premise(p), ensure_ascii=False), schema)
    text = result['synopsis'].strip()
    if not text or len(text) > 8000:
        raise ValueError('あらすじを取得できませんでした。再生成してください。')
    return text


def generate(p, ollama, progress):
    if not p.synopsis.strip():
        raise ValueError('先にあらすじを作成・編集してください。')
    count = max(1, math.ceil(p.settings.duration / 20))
    scene_schema = {'type': 'object', 'properties': {'scenes': {'type': 'array', 'minItems': count, 'maxItems': count, 'items': {'type': 'string'}}}, 'required': ['scenes']}
    data = json.dumps(premise(p), ensure_ascii=False)
    progress(.03, 'あらすじから場面と展開を設計中')
    plan = ask(p, ollama, f'以下の編集済みあらすじを、約20秒ずつの重複しない{count}個の展開に細分化してください。scenes配列の要素数は必ず{count}個。各要素に異なる発見・障害・行動とその具体的な結果を一つずつ書く。新しい事実を引き継ぐ一本の物語にする。「調べる、出発の準備、感謝、笑顔で帰る」等の同じ内容を複数の要素で言い換えることは禁止。固有名・物品・数値を使い、会話で描ける具体的な展開にする。結末は最終要素だけで達成し、それ以前に話を終わらせない。\n' + data, scene_schema)['scenes']
    if len(plan) != count:
        raise ValueError('場面構成が不足しています。再生成してください。')
    cast = [c for c in p.cast if c.enabled]
    schema = {'type': 'object', 'properties': {'lines': {'type': 'array', 'minItems': 2, 'maxItems': 35, 'items': {'type': 'object', 'properties': {
        'speaker': {'type': 'string', 'enum': [c.id for c in cast]}, 'text': {'type': 'string', 'minLength': 1, 'maxLength': 500},
        'emotion': {'type': 'string', 'enum': ['自然','楽しい','穏やか','驚き','悲しい','怒り','緊張','ささやき']},
        'simultaneous': {'type': 'boolean'}}, 'required': ['speaker', 'text', 'emotion', 'simultaneous']}}}, 'required': ['lines']}
    lines = []
    for index, scene in enumerate(plan):
        target = (p.settings.duration - estimate(p, lines)) / (count - index)
        rate = 6.5 * p.settings.speed * sum(c.pace for c in cast) / len(cast)
        chars = round(max(10, target - target / 8 * .25) * rate)
        correction = ''
        scene_lines = []
        for attempt in range(4):
            remaining = max(1, target - estimate(p, scene_lines))
            observed = estimate(p, scene_lines) / len(scene_lines) if scene_lines else 2.5
            turns = max(2, min(35, round(remaining / max(1.5, min(7, observed)))))
            per_line = max(8, round(chars / turns))
            schema['properties']['lines'].update(minItems=2, maxItems=40)
            schema['properties']['lines']['items']['properties']['text'] = {'type': 'string'}
            progress(.08 + .85 * index / count, f'台本 {index + 1}/{count}の展開・分量と重複を確認中（{attempt + 1}/4）')
            prompt = (f'作品「{p.title}」の第{index+1}/{count}の展開。日本語の自然な会話劇を書いてください。\n今起こっている出来事（過去の出来事の説明ではなく、その場の会話で描く）：{scene}\n'
                      f'今回の出力は約{turns}発言、合計約{chars}文字（約{remaining:.0f}秒）。一発言は約{per_line}文字を目安に、短い反応と具体的な会話を組み合わせてください。'
                      'セリフtextに人物名や説明、コード記号を入れないでください。今回の出来事にある事実を具体的に語り、行動の結果を必ず描いてください。「調べよう・確認しよう・そうだね」の反復は禁止。沈黙を表す…を連発しません。既に済んだ話は繰り返しません。'
                      + ('最後にこの物語の結末を描いてください。' if index==count-1 else '次の場面へ続くため、物語全体を締めないでください。') +
                      '\n演出：' + p.settings.direction + ' 全体の感情：' + p.settings.emotion + '。emotionは自然・楽しい・穏やか・驚き・悲しい・怒り・緊張・ささやきから選択。' + '\n必要な語句とセリフ：' + p.settings.keywords + '\n' + p.settings.key_lines +
                      '\n登場人物：' + '\n'.join(f'話者ID={c.id}、名前={c.name}、一人称は必ず「{c.first_person}」、呼ばれ方={c.called_as or c.name}、相手の呼び方={c.addressing}、性格={c.personality}、役割={c.role}、話し方={c.style}' for c in cast) +
                      '\nここまでの会話（今回の出力には再掲載しない）：' + json.dumps([{'speaker': l.speaker, 'text': l.text} for l in lines + scene_lines], ensure_ascii=False) +
                      '\n通常simultaneousはfalse。声を揃える場面だけ、2人目以降をtrueにします。' + correction)
            aliases = {c.name: c.id for c in cast}
            aliases.update({c.called_as: c.id for c in cast if c.called_as})
            try:
                parsed = ask(p, ollama, prompt, schema, index * 10 + attempt)
                new = parse_lines(parsed, cast, f'シーン{index//3+1}')
            except (ValueError, TypeError, AttributeError, KeyError):
                correction = '\n前案の形式に誤りがありました。lines配列にspeakerとtextを持つセリフを出力してください。'
                continue
            for line in new:
                for c in cast:
                    for name in (c.name, c.called_as):
                        if name:
                            line.text = re.sub(r'^' + re.escape(name) + r'\s*[：:]\s*', '', line.text).strip()
            if any(not l.text or l.text in aliases or re.search(r'[{}]|\b(?:false|true|null)\b', l.text) for l in new):
                correction = '\n前案のtextにコード・不要な記号が混入しました。textには日本語の自然なセリフだけを入れ、人物名や属性値は書かないでください。'
                continue
            if not new or any(l.speaker not in {c.id for c in cast} for l in new):
                correction = '\n有効な話者のセリフが不足。指定の話者IDを使用して書き直す。'
                continue
            new[0].simultaneous = False
            original_count = len(new)
            raw_lines = new
            new = remove_repeats(new, lines + scene_lines)
            candidate = scene_lines + new
            seconds = estimate(p, candidate)
            duplicate = repeated(candidate, lines)
            progress(.08 + .85 * index / count, f'第{index+1}場面：推定{seconds:.1f}秒 / 目標{target:.1f}秒 / {len(candidate)}発言' + (f'・重複{original_count-len(new)}件を除去' if original_count != len(new) else ''))
            lower, upper = (.9, 1.1) if count == 1 else (.7, 1.3)
            total_ok = index < count - 1 or .9 * p.settings.duration <= estimate(p, lines + candidate) <= 1.1 * p.settings.duration
            if lower * target <= seconds <= upper * target and not duplicate and total_ok:
                lines.extend(candidate)
                break
            if attempt == 3 and index < count - 1 and .65 * target <= seconds < .9 * target and not duplicate:
                # Scene lengths can vary; redistribute the deficit over the remaining scenes.
                lines.extend(candidate)
                break
            if seconds < .9 * target:
                scene_lines = candidate
                chars = max(10, round((target-seconds)*rate))
                correction = '\n場面の長さが不足しています。上記の続きとして、この場面の具体的な行動・新しい情報・反応を追加してください。'
                continue
            factor = target / max(seconds, 1)
            chars = max(10, round(sum(len(l.text) for l in candidate) * factor)) if candidate else round(target * rate)
            scene_lines = []
            correction = f'\n前案は{seconds:.1f}秒で目標{target:.1f}秒。合計約{chars}文字に書き直す。' + ('重複した発言があるため、新しい出来事と反応で書き直す。' if duplicate else '')
            correction += '\n修正対象の前案（この場面全体を短く書き直す）：' + json.dumps([{'speaker': l.speaker, 'text': l.text, 'simultaneous': l.simultaneous} for l in candidate], ensure_ascii=False)
            if len(new) != original_count:
                conflicts = []
                seen = list(lines)
                for l in raw_lines:
                    normalized = re.sub(r'\W', '', l.text)
                    if len(normalized) >= 12 and not l.simultaneous:
                        for old in seen:
                            if SequenceMatcher(None, normalized, re.sub(r'\W', '', old.text)).ratio() > .83:
                                conflicts.append(l.text)
                                break
                    seen.append(l)
                correction += '\n以下の重複セリフは再使用・言い換え禁止。別の出来事に対する具体的な新しい発言へ置き換える：' + json.dumps(conflicts, ensure_ascii=False)
        else:
            raise ValueError(f'第{index+1}場面の重複・長さの検査に合格しませんでした。あらすじに具体的な展開を追加するか、再生成してください。現在の台本は保持されます。')
    if len(lines) > 300 or not .9 * p.settings.duration <= estimate(p, lines) <= 1.1 * p.settings.duration:
        raise ValueError('指定時間に必要な台本量を満たしませんでした。あらすじを調整して再生成してください。')
    return lines


def script_text(p):
    cast = {c.id: c for c in p.cast}
    parts = [p.title, 'あらすじ', p.synopsis, '登場人物']
    parts.extend(f'{c.name} / 一人称：{c.first_person} / 呼ばれ方：{c.called_as} / 話すペース：{c.pace}倍 / 相手の呼び方：{c.addressing}' for c in p.cast if c.enabled)
    parts.append('台本')
    parts.extend(f'{l.scene} / {cast[l.speaker].name}（{l.emotion}）' + ('【前の発言と同時】' if l.simultaneous else '') + f'\n{l.text}\n［間 {l.pause}秒 / SE {l.se}］' for l in p.lines)
    return '\n\n'.join(parts)
