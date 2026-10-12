"""Development orchestration only. Jury reproduction uses train.py, not this sweep."""
import json
import os
import sys
import subprocess
import numpy as np
import torch
sys.path.insert(0, '.')
from scripts.experiment import load_full, run_config
from mtc_lib.common import stratified_split

torch.set_num_threads(4)
seed = json.load(open('final_config.json'))['seed']
X, y = load_full()
tr, va = stratified_split(y, .1, seed)
configs = [
    dict(name='linear', hidden=[], epochs=30),
    dict(name='compact', hidden=[256], epochs=30),
    dict(name='mlp', hidden=[512,256], epochs=30),
]
results = []
for cfg in configs:
    cfg.update(seed=seed, batch_size=512, lr=.003, dropout=.3)
    results.append(run_config(cfg, X,y,tr,va,out_dir='reports/final'))
best = max(results, key=lambda r:r['val_f1'])
cfg = {k:v for k,v in best['cfg'].items() if k not in ('name','seed')}
config = {'seed':seed,'training':cfg, 'selected_by':'uncalibrated validation macro-F1', 'selected':best['name']}
json.dump(config,open('reports/final/selected_config.json','w'),indent=2)
del X,y,tr,va
import gc;gc.collect()
weights=f'models/final_seed{seed}'
subprocess.run([sys.executable,'train.py','--config','reports/final/selected_config.json','--seed',str(seed),'--out',weights,'--threads','4','--ensemble','2'],check=True)
os.makedirs('submissions',exist_ok=True)
subprocess.run([sys.executable,'predict.py','--weights',weights,'--out',f'submissions/submission_seed_{seed}.npz'],check=True)
subprocess.run([sys.executable,'-m','pip','freeze'],stdout=open('reports/final/environment.txt','w'),check=True)
