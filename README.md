# Stock Trading with Machine Learning

## Overview

A stock trading bot that uses machine learning to make price predictions.

## Requirements

-   Python 3.5+
-   alpha_vantage
-   pandas
-   numpy
-   sklearn
-   keras
-   tensorflow
-   matplotlib

## Documentation

[Blog Post](https://yacoubahmed.me/blog/stock-prediction-ml)

[Medium Article](https://medium.com/towards-data-science/getting-rich-quick-with-machine-learning-and-stock-market-predictions-696802da94fe)

## Install

1. Clone the repo
2. Create a new env `conda create -n STOCK python==3.7 --no-default-packages -y`
3. Activate the env `conda activate STOCK`
4. Pip install the requirements `pip install -r requirements.txt`

## Setup

Create an [Alpha Vantage](https://www.alphavantage.co/) account and get an API key

Make a creds.json file with your key
```
{
  "av_api_key":"XXXXXXXXXXXXXXXX"
}
```

run save_data_to_csv.py
```
python save_data_to_csv.py KO daily
```

train your model
```
python tech_ind_model.py
```

run evaluation
```
python trading_algo.py
```

## License

[GPL-3.0](https://www.gnu.org/licenses/quick-guide-gplv3.html)
