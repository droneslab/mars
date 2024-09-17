import pandas as pd
import sys

df = pd.read_csv('/mnt/d/Downloads/corl23_results.csv')
names = df['Name'].tolist()
h_names = [x for x in names if x[:2] == 'h_']
l_names = [x for x in names if x[:2] == 'l_']
h_names_noComps = [x for x in h_names if 'matchnet' not in x and 'pairloss' not in x and 'pie' not in x and 'ride' not in x]
l_names_noComps = [x for x in l_names if 'matchnet' not in x and 'pairloss' not in x and 'pie' not in x and 'ride' not in x]
h_idxs = [ df.index[df['Name'] == x].tolist()[0] for x in h_names_noComps ]
l_idxs = [ df.index[df['Name'] == x].tolist()[0] for x in l_names_noComps ]
h_vals = df.iloc[h_idxs]
l_vals = df.iloc[l_idxs]

t = sys.argv[1]
if t == 'h':
    ns = h_names_noComps
    vss = h_vals
else:
    ns = l_names_noComps
    vss = l_vals

names = [x for x in ns if '_caml_aws' not in x]
diffs = {}
for h in names:
    diffs[h] = {}    
n = 0
for i in range(2,len(h_idxs)+2,2):
    vals = vss[i-2:i]
    cols = list(vals.keys())
    cols.remove('Name')
    cols.remove('ID')
    for c in cols:
        vs = list(vals[c])
        name = names[n]
        stock = vs[0]
        caml = vs[1]
        diff = float(caml) - float(stock)
        diffs[name][c] = diff
    n += 1

test_diffs = {}

for k in list(diffs.keys()):
    for k2 in list(diffs[k].keys()):
        if k2 in list(test_diffs.keys()):
            test_diffs[k2].append((k, diffs[k][k2]))
        else:
            test_diffs[k2] = []
            test_diffs[k2].append((k, diffs[k][k2]))

loss_counts = {}     
exlude = ['recall_8', 'navTest']
for test in list(test_diffs.keys()):
    if 'recall' in test or 'navTest' in test:
        continue
    
    
    vs = test_diffs[test]
    vs.sort(key = lambda x: x[1])
    if 'rmse' not in test:
        vs.reverse()
        vs = [v for v in vs if v[1] > 0]
    else:
        vs = [v for v in vs if v[1] < 0]
    print('\n---------- ' + test + ' ----------')
    for n,v in vs:
        if n in list(loss_counts.keys()):
            loss_counts[n] += 1
        else:
            loss_counts[n] = 1
            
        print(n.split(t + '_')[1], v)
        
print(sorted(loss_counts.items(), key=lambda x: x[1], reverse=True))