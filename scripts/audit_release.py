"""Scan candidate public files without printing sensitive matching text.

Use --git-index after staging to inspect exactly what a Git commit would add.
This is a conservative static check, not proof that arbitrary files are safe.
"""
import argparse
from pathlib import Path
import re
import subprocess

PATTERNS = {
    'private absolute path': r'/(?:home|Users|scratch|mnt|data/users)/[\w.-]+',
    'Windows user path': r'[A-Za-z]:\\Users\\',
    'email address': r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b',
    'server alias': r'\bhpc-\d+\b',
    'IP address': r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',
    'credential token': r'\b(?:sk|hf|ghp|github_pat)[_-][A-Za-z0-9_-]{16,}',
    'private key': r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
    'literal credential': r'''(?i)(?:api_key|access_token|password|secret_key)\s*[=:]\s*["'][^"'\s]{8,}["']''',
}
ALLOWED_SUFFIXES = {'.py', '.md', '.json', '.toml', '.yml', '.yaml', '.txt'}
ALLOWED_NAMES = {'LICENSE', 'NOTICE', 'lmms-eval-LICENSE', '.gitignore', '.gitattributes'}


def audit(root, paths):
    findings = []
    for relative in paths:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            findings.append((str(relative), 'symlink or missing file', None)); continue
        if path.suffix not in ALLOWED_SUFFIXES and path.name not in ALLOWED_NAMES:
            findings.append((str(relative), 'unexpected artifact type', None)); continue
        raw = path.read_bytes()
        if len(raw) > 1024 * 1024:
            findings.append((str(relative), 'oversized file', None)); continue
        try:
            content = raw.decode('utf-8')
        except UnicodeDecodeError:
            findings.append((str(relative), 'binary content', None)); continue
        for label, pattern in PATTERNS.items():
            for match in re.finditer(pattern, content):
                findings.append((str(relative), label, content.count('\n', 0, match.start()) + 1))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--git-index', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.git_index:
        result = subprocess.run(['git', 'ls-files', '-z'], cwd=root, check=True, capture_output=True)
        paths = [Path(value) for value in result.stdout.decode().split('\x00') if value]
    else:
        paths = [p.relative_to(root) for p in root.rglob('*')
                 if p.is_file() and '.git' not in p.relative_to(root).parts
                 and '__pycache__' not in p.relative_to(root).parts]
    findings = audit(root, paths)
    for path, label, line in findings:
        print(f'{path}' + (f':{line}' if line else '') + f': {label} (value withheld)')
    print(f'Scanned {len(paths)} public candidates; findings: {len(findings)}')
    raise SystemExit(bool(findings))


if __name__ == '__main__':
    main()
