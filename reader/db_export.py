import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class DBExportReader(Dataset):
    def __init__(self, db_h5_file, future_time = 20, lookback = 100, read_first_k_table = -1):
        self._db = pd.HDFStore(db_h5_file)
        self._future_time = future_time
        self._lookback = lookback
        self._db_len = 0

        self._tables = []
        for k in self._db:
            self._db_len += len(self._db[k]) - future_time - lookback
            self._tables.append(self._db[k])
            if read_first_k_table != -1 and len(self._tables) == read_first_k_table:
                break

    def __len__(self):
        return self._db_len

    def _read_from_table(self, t, idx):
        input_start_idx = idx
        input = t.iloc[input_start_idx:input_start_idx+self._lookback, 4:]
        input['AveragePrice'] = input.apply(lambda x: (x['AskPrice1'] + x['BidPrice1']) / 2, axis = 1)
        last_average_price = input.iloc[-1]['AveragePrice']

        result_idx = idx + self._lookback + self._future_time - 1
        result_row = t.iloc[result_idx, 4:]
        result_avgprice = (result_row['AskPrice1'] + result_row['BidPrice1']) / 2

        # print('last_avg={} result_Avg={}'.format(last_average_price, result_avgprice))

        # -1, 0, 1
        cmp_result = int(result_avgprice > last_average_price) - int(result_avgprice < last_average_price)
        # print('cmp_result=', cmp_result)
        # 0, 1, 2
        onehot_idx = cmp_result + 1

        result_arr = np.zeros(3)
        result_arr[onehot_idx] = 1
        return input.as_matrix(), result_arr

    def __getitem__(self, idx):
        for t in self._tables:
            l = len(t)
            valid_l = l - self._future_time - self._lookback
            if idx < valid_l:
                return self._read_from_table(t, idx)
            else:
                idx -= valid_l
        raise IndexError("DBExportReader out of range. idx={}, while db_len={}".format(idx, self._db_len))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._db.close()


if __name__ == '__main__':
    with DBExportReader('./processed/DBExport.h5', read_first_k_table=1) as db:
        for i in range(len(db)):
            db[i]