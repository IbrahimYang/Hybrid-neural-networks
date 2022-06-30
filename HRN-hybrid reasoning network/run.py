import os
import pickle
from tqdm import tqdm

from executor import Executor
from simulation import Simulation

from SNN import SRN
device = 'cpu'

with open('./data/programs.pkl', 'rb') as f:
    Q = pickle.load(f)

preds_dir = './data/HU_final'
error = {
    'descriptive':0,
    'explanatory':0,
    'predictive':0,
    'counterfactual':0
}
total = {
    'descriptive':0,
    'explanatory':0,
    'predictive':0,
    'counterfactual':0
}

pbar = tqdm(range(5000))
for s_idx in pbar:
    scene = os.path.join(preds_dir, 'sim_%05d.pkl' % (s_idx + 10000))
    net = SRN(scene, debug=False).to(device)
    for q_idx,q in enumerate(Q[s_idx]):
        if q['type'] == 'descriptive': # oe
            pred = net(q['program'])
            ans = q['answer']
            if pred != ans:
                error[q['type']] += 1                
            total[q['type']] += 1

        else: # mc
            for c in q['choices']:
                program = c['program'] + q['program']
                pred = net(program)
                ans = c['answer']
                if pred != ans:
                    error[q['type']] += 1
                total[q['type']] += 1

    bb = total['descriptive']
    aa = bb - error['descriptive']
    pbar.set_description('descriptive acc: {:.3f}%({:d}/{:d})'\
            .format(float(aa)*100/bb, aa, bb))
    
    
for key in error:
    a = error[key]
    b = total[key]
    print('{:s} \t {:.2f}% ({:d}/{:d})'.format(key,100*(b-a)/b,b-a,b))
