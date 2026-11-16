"""Build four KTH-TIPS2-b folds with physical specimens kept intact."""
import argparse,json
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();root=Path(a.data_root).expanduser().resolve();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    categories=sorted(p for p in root.iterdir() if p.is_dir() and all((p/('sample_'+s)).is_dir() for s in 'abcd'))
    if len(categories)!=11:raise ValueError('Expected 11 categories with sample_a, sample_b, sample_c, sample_d subdirectories')
    groups={}
    for label,category in enumerate(categories):
        for specimen in 'abcd':
            files=sorted(p for p in (category/('sample_'+specimen)).rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.ppm'))
            if len(files)!=108:raise ValueError(f'Expected 108 views: {category.name}/sample_{specimen}')
            groups[label,specimen]=[dict(relative_path=str(p.relative_to(root)),label=label) for p in files]
    for fold,(train,val,test) in enumerate([('ab','c','d'),('bc','d','a'),('cd','a','b'),('da','b','c')],1):
        splits={name:[row for label in range(11) for specimen in specimens for row in groups[label,specimen]]
                for name,specimens in [('train',train),('validation',val),('test',test)]}
        manifest=dict(dataset=f'KTHb_f{fold}',classes=[p.name for p in categories],data_root=str(root),splits=splits)
        (out/f'KTHb_f{fold}.json').write_text(json.dumps(manifest,indent=2)+'\n')
