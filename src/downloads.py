"""Verified, resumable downloads; receipts survive removal of staged raw data."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
UA = 'curl/7.68.0'


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def json_url(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def download(url, dest, size=None, md5=None, tries=5):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    receipt = ROOT / 'data/provenance' / (hashlib.sha256(str(dest.resolve()).encode()).hexdigest() + '.json')
    if dest.exists():
        valid = (size is None or dest.stat().st_size == size) and (md5 is None or digest(dest, 'md5') == md5)
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            valid &= previous['url'] == url and previous['sha256'] == digest(dest)
        elif size is None and md5 is None:
            raise ValueError(f'{dest}: existing input has no checksum or download receipt')
        if valid:
            print(f'Using verified input {dest}', flush=True)
            return dest
        raise ValueError(f'{dest}: cached input failed validation; remove it explicitly before retrying')
    if size is None:
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD', headers={'User-Agent': UA}),
                                    timeout=120) as response:
            value = response.headers.get('Content-Length')
            if value is not None:
                size = int(value)
    if size is None and md5 is None:
        raise ValueError(f'{url}: neither upstream size nor checksum is available')
    part = dest.with_name(dest.name + '.part')
    aria2 = shutil.which('aria2c')
    for attempt in range(tries):
        if aria2:
            cmd = [aria2, '--quiet=true', '--continue=true', '--split=16', '--max-connection-per-server=16',
                   '--min-split-size=16M', '--file-allocation=none', '--max-tries=20', '--retry-wait=5',
                   '--timeout=60', '--no-want-digest-header=true', f'--user-agent={UA}', '--auto-file-renaming=false',
                   '--allow-overwrite=false', '--dir', str(part.parent), '--out', part.name, url]
        else:
            cmd = ['curl', '-fsSL', '--retry', '3', '--connect-timeout', '60', '-C', '-', '-o', str(part), url]
        result = subprocess.run(cmd)
        if result.returncode == 0 and part.exists() and (size is None or part.stat().st_size == size):
            break
        print(f'WARNING download {attempt + 1}/{tries} failed or incomplete: {url}', flush=True)
    else:
        raise RuntimeError(f'Download incomplete: {url}; partial file retained for resume')
    if md5 is not None and digest(part, 'md5') != md5:
        raise ValueError(f'{url}: MD5 mismatch; partial file retained for inspection')
    actual = digest(part)
    part.replace(dest)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temp = receipt.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(url=url, path=str(dest.relative_to(ROOT)),
                                  bytes=dest.stat().st_size, sha256=actual, upstream_md5=md5), indent=2) + '\n')
    os.replace(temp, receipt)
    return dest


def figshare_file(article, name, dest):
    meta = json_url(f'https://api.figshare.com/v2/articles/{article}')
    hits = [f for f in meta['files'] if f['name'] == name]
    if len(hits) != 1:
        raise ValueError(f'Figshare {article}: expected one {name}, found {len(hits)}')
    f = hits[0]
    return download(f'https://ndownloader.figshare.com/files/{f["id"]}', dest,
                    size=f['size'], md5=f.get('computed_md5'))


def zenodo_file(record, name, dest):
    meta = json_url(f'https://zenodo.org/api/records/{record}')
    hits = [f for f in meta['files'] if f['key'] == name]
    if len(hits) != 1:
        raise ValueError(f'Zenodo {record}: expected one {name}, found {len(hits)}')
    f = hits[0]
    algorithm, checksum = f['checksum'].split(':', 1)
    if algorithm != 'md5':
        raise ValueError(f'Unsupported upstream checksum {algorithm}')
    return download(f'https://zenodo.org/api/records/{record}/files/{name}/content', dest,
                    size=f['size'], md5=checksum)
