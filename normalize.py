import pandas as pd
import os
import math

for root, dirs, files in os.walk('./processed'):
    for filename in files:
        if filename.endswith('.h5'):
            fullpath = os.path.join(root, filename)
            print('=== Normalize h5 file {}'.format(fullpath))
            with pd.HDFStore(fullpath, complevel=9, complib='blosc') as store:
                for k in store:
                    print(' -- Processing table {}'.format(k))
                    t = store[k]
                    t = t[t['AskVolume1'] < 100000][t['Turnover'] >= 0][t['LastTurnover'] >= 0]
                    s_min = t.min()
                    s_max = t.max()
                    if s_max['Turnover'] > 10000:
                        print('  - Normalizing columns')
                        t[['Turnover', 'LastTurnover']] = t[['Turnover', 'LastTurnover']].applymap(
                            lambda x: math.log(x+1))
                        s_min = t.min()
                        s_max = t.max()

                    for col in t.columns[4:]:
                        print('    -- processing col {} min={} max={}'.format(col, s_min[col], s_max[col]))
                        if s_min[col] != s_max[col]:
                            t[col] = (t[col] - s_min[col]) / (s_max[col] - s_min[col])
                    store[k] = t