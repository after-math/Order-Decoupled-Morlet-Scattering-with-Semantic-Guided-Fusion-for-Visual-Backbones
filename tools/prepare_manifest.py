"""Import archived dataset partitions without copying images or experiment results."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import load_manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,help='Existing saved manifest.json with train/validation/test records')
    p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();m=load_manifest(args.source,args.data_root)
    portable={k:m[k] for k in ['dataset','classes','data_root','splits']}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(portable,ensure_ascii=False,indent=2)+'\n')
