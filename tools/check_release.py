"""Fail on run artifacts, private endpoints, credential assignments or unknown files."""
import ast,json,re,ipaddress
from pathlib import Path
from urllib.parse import urlparse

PUBLIC_DOMAINS={'github.com','raw.githubusercontent.com','huggingface.co','download.pytorch.org','www.apache.org','apache.org','arxiv.org'}


def check(root):
    root=Path(root);files=[]
    for path in root.rglob('*'):
        if '.git' in path.relative_to(root).parts:raise ValueError('Do not distribute repository history in the source folder')
        if path.is_symlink():raise ValueError('Symlink in release: '+str(path.relative_to(root)))
        if not path.is_file():continue
        files.append(path)
        if path.name.startswith('._') or path.name=='.DS_Store':raise ValueError('OS metadata file')
        if path.name not in ('.gitignore','MAMBAOUT_LICENSE') and path.suffix not in {'.py','.md','.json','.txt'}:
            raise ValueError('Unexpected artifact: '+str(path.relative_to(root)))
        text=path.read_text(encoding='utf-8')
        if path.suffix=='.py':
            tree=ast.parse(text,filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node,(ast.Assign,ast.AnnAssign)):
                    value=node.value
                    targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                    names=[t.id for t in targets if isinstance(t,ast.Name)]
                    if isinstance(value,ast.Constant) and isinstance(value.value,str) and value.value and any(re.search(r'(password|passwd|secret|api_key|access_token)',n,re.I) for n in names):
                        raise ValueError('Potential embedded credential: '+str(path.relative_to(root)))
        if path.suffix=='.json':
            if path.parent.name!='configs':raise ValueError('JSON outside configs')
            json.loads(text)
        if re.search(r'(?:/Users|/Volumes|/20220018|/202421633187)/',text):raise ValueError('Private filesystem path')
        if re.search(r'(?:'+'seeta'+'cloud|root'+'@|ssh'+'pass|BEGIN [A-Z ]*PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})',text):raise ValueError('Private endpoint or credential')
        for address in re.findall(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])',text):
            try:ipaddress.ip_address(address)
            except ValueError:continue
            raise ValueError('Embedded IP address: '+str(path.relative_to(root)))
        for url in re.findall(r'https?://[^\s\"\'<>`)]+',text):
            parsed=urlparse(url.rstrip('.,;'))
            if parsed.hostname not in PUBLIC_DOMAINS or parsed.username or parsed.password or parsed.query:
                raise ValueError('Unapproved URL: '+str(path.relative_to(root)))
    return len(files)

if __name__=='__main__':
    count=check(Path(__file__).resolve().parents[1]);print(f'PASS: {count} source/configuration/documentation files')
