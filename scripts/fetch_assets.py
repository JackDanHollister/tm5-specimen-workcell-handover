"""Download authenticated private release archives and verify/extract their contents."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--without-models',action='store_true');p.add_argument('--from-dir',type=Path)
    p.add_argument('--experiments-only',action='store_true',help='Install only the new diagnostic supplement into an existing checkout');args=p.parse_args()
    release=json.loads((ROOT/'configs/release_assets.json').read_text())
    downloads=ROOT/'.downloads';downloads.mkdir(exist_ok=True)
    for bundle in release['bundles']:
        if args.experiments_only and bundle['id']!='experiments':continue
        if args.without_models and bundle['id']=='models':continue
        parts=[]
        for item in bundle['parts']:
            target=downloads/item['name']
            if not target.exists():
                if args.from_dir:shutil.copyfile(args.from_dir/item['name'],target)
                else:subprocess.run(['gh','release','download',bundle.get('tag',release['tag']),'--repo',release['repository'],
                                     '--pattern',item['name'],'--dir',str(downloads)],check=True)
            if target.stat().st_size!=item['bytes'] or sha(target)!=item['sha256']:
                raise ValueError('Downloaded asset differs: '+item['name'])
            parts.append(target)
        archive=downloads/(bundle['id']+'.tar.gz')
        if len(parts)==1:archive=parts[0]
        elif not archive.exists():
            with archive.open('wb') as output:
                for path in parts:
                    with path.open('rb') as source:shutil.copyfileobj(source,output,8*1024*1024)
        if sha(archive)!=bundle['archive_sha256']:raise ValueError('Joined archive differs')
        with tarfile.open(archive,'r:*') as tar:
            for member in tar.getmembers():
                target=(ROOT/member.name).resolve()
                if ROOT.resolve() not in target.parents or not (member.isfile() or member.isdir()):
                    raise ValueError('Archive entry escapes package or is a link')
            if hasattr(tarfile,'data_filter'):tar.extractall(ROOT,filter='data')
            else:tar.extractall(ROOT)
        print('Installed',bundle['id'],flush=True)
    command=[str(Path(__import__('sys').executable)),str(ROOT/'scripts/replay_experiment_evidence.py' if args.experiments_only else ROOT/'scripts/verify_assets.py')]
    if args.without_models and not args.experiments_only:command.append('--without-models')
    subprocess.run(command,check=True)


if __name__=='__main__':main()
