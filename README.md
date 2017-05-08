# future-price-predictor
Toy price predictor of future market

## Requirements
Python 3.6, CUDA 8.0

```
virtualenv -p python3.6 --system-site-packages .env
source .env/bin/activate
source ./install.sh
```

You should place raw data at `data/DBExport` and `data/m0000`. Processed data is stored at `processed/DBExport.h5` and `processed/m0000.h5`.

Processed data could be read as:
```
import pandas as pd
with pd.HDFStore('processed/DBExport.h5') as store:
	for k in store.keys():
		frame = pd.read_hdf('processed/DBExport.h5', k)
		# do sth to frame
		frame.to_hdf('processed/DBExport.h5', k, mode='w')
```
