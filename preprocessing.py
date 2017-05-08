import pandas as pd
import os

for root, dirs, files in os.walk('data'):
    for filename in files:
        if filename.endswith('.csv'):
            fullpath = os.path.join(root, filename)
            print('Processing data from csv={}'.format(fullpath))
            frame = pd.read_csv(fullpath)
            output_filename = os.path.join('processed', os.path.split(root)[-1] + '.h5')

            chars_to_replace = ('[', ']', '.', ' ')
            str_to_replace = filename.split('.')[0]
            for c in chars_to_replace:
                str_to_replace = str_to_replace.replace(c, '_')
            output_table = str_to_replace
            print('Saving to {} with table={}'.format(output_filename, output_table))
            frame.to_hdf(output_filename, output_table, complevel=9, complib='blosc')