"""Read newly generated run outputs; calculate paired gains and KTH fold means."""
import argparse,csv,json,sys
from pathlib import Path
from collections import defaultdict
import statistics

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',required=True);a=p.parse_args()
    grouped=defaultdict(list)
    for metrics_file in Path(a.runs).rglob('metrics.json'):
        import torch
        checkpoint=torch.load(metrics_file.parent/'best.pt',map_location='cpu',weights_only=False)
        contract=checkpoint['contract'];spec=checkpoint['spec'];dataset=contract['dataset']
        group='KTH-TIPS2-b' if dataset.startswith('KTHb_f') else dataset
        key=(group,spec['backbone'],spec['variant'],spec['width'],contract['shots'],spec['frozen'])
        grouped[key].append((dataset,json.loads(metrics_file.read_text())['test']['accuracy']))
    writer=csv.writer(sys.stdout);writer.writerow(['dataset','backbone','variant','width','shots','frozen','test_accuracy','gain_pp','folds'])
    means={}
    for key,rows in grouped.items():
        if len({ds for ds,_ in rows})!=len(rows):raise ValueError('Duplicate runs for '+str(key))
        if key[0]=='KTH-TIPS2-b' and len(rows)!=4:raise ValueError('KTH aggregation requires all four folds')
        means[key]=statistics.mean(v for _,v in rows)
    for key,accuracy in sorted(means.items(),key=lambda x:str(x[0])):
        baseline=(key[0],key[1],'Native',*key[3:])
        gain=accuracy-means[baseline] if baseline in means and key[2]!='Native' else ''
        writer.writerow([*key,f'{accuracy:.2f}',f'{gain:+.2f}' if gain!='' else '',len(grouped[key])])
