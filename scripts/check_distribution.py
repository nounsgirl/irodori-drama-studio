"""Reject unexpected tracked files and common sensitive-content patterns."""
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={
    '.gitignore','.gitattributes','.env.example',
    '.github/workflows/check.yml','README.md','LICENSE','requirements.txt',
    'audio_engine.py','models.py','server.py','storage.py','worker.py',
    'start.ps1','start.cmd','test_core.py',
    'static/index.html','static/style.css','static/app.js',
    'scripts/check_distribution.py',
}
PATTERNS={
    'credential':r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,})',
    'private key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'machine-specific path':r'(?i)(?:\b[A-Z]:[/\\]|/(?:Users|home)/[A-Za-z0-9_.-]+/)',
    'embedded binary payload':r'[A-Za-z0-9+/]{350,}={0,2}',
}

def main():
    result=subprocess.run(['git','ls-files','--cached','-z'],cwd=ROOT,check=True,capture_output=True)
    paths=[p for p in result.stdout.decode('utf-8').split('\0') if p]
    if not paths:raise SystemExit('No staged or tracked files to inspect.')
    errors=[]
    for name in paths:
        if name not in ALLOWED:errors.append(f'Unexpected file: {name}');continue
        # Read the index, not the working tree, so staged content is authoritative.
        data=subprocess.run(['git','show',f':{name}'],cwd=ROOT,check=True,capture_output=True).stdout
        if len(data)>200_000:errors.append(f'Oversized source file: {name}')
        try:text=data.decode('utf-8')
        except UnicodeDecodeError:errors.append(f'Non-text file: {name}');continue
        if '\0' in text:errors.append(f'Binary content: {name}')
        for label,pattern in PATTERNS.items():
            if re.search(pattern,text):errors.append(f'{label}: {name}')
    if errors:
        print('\n'.join(errors));return 1
    print(f'Distribution check passed: {len(paths)} source/configuration files; no media or runtime data.')
    return 0

if __name__=='__main__':sys.exit(main())
