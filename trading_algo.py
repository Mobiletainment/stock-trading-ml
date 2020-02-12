import numpy as np
from keras.models import load_model
from util import csv_to_dataset, history_points
import matplotlib.pyplot as plt

model = load_model('technical_model.h5')

ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('KO_daily.csv')

# test_split = 0.9
# n = int(ohlcv_histories.shape[0] * test_split)
n = -365

ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

buys = []
sells = []
thresh = .11

start = 0
end = -1

x = -1
for ohlcv, ind in zip(ohlcv_test[start: end], tech_ind_test[start: end]):
    normalised_price_today = ohlcv[-1][0]
    normalised_price_today = np.array([[normalised_price_today]])
    price_today = y_normaliser.inverse_transform(normalised_price_today)
    predicted_price_tomorrow = np.squeeze(y_normaliser.inverse_transform(model.predict([[ohlcv], [ind]])))
    delta = predicted_price_tomorrow - price_today
    if delta > thresh:
        buys.append((x, price_today[0][0]))
    elif delta < -thresh:
        sells.append((x, price_today[0][0]))
    x += 1

print(f"buys: {len(buys)}")
print(f"sells: {len(sells)}")


def compute_earnings(buys_, sells_):
    last_price = 0
    stock = 0
    balance = 5000
    start_balance = int(balance)
    highest_balance = int(balance)

    while len(buys_) > 0 and len(sells_) > 0:
        if buys_[0][0] < sells_[0][0]:
            # Test policy: Leave some in the bank
            num_stocks = (balance * .8) // buys_[0][1]
            if num_stocks > 0:
                stock += num_stocks
                cost = round(num_stocks * buys_[0][1], 2)
                balance -= cost
                print(f'buy {num_stocks} stocks for ${cost} at ${buys_[0][1]:.2f} each')
                last_price = buys_[0][1]
            buys_.pop(0)
        else:
            # time to sell all of our stock
            retur = stock * sells_[0][1]
            balance += retur
            if stock > 0:
                print(f'sell all {stock} stocks for ${retur:.2f} at ${sells_[0][1]:.2f} each')
            stock = 0
            if balance > highest_balance:
                highest_balance = int(balance)
            last_price = sells_[0][1]
            sells_.pop(0)

    # sell any leftover stock
    if stock > 0:
        balance += stock * last_price

    print(f"\nTotal Balance: ${balance:.2f}\n"
          f"Highest balance: ${highest_balance}\n"
          f"Earnings: ${balance - start_balance:.2f}\n"
          f"Gain %{((balance/ start_balance) - 1) * 100:.2f}\n")


# we create new lists so we dont modify the original
compute_earnings([b for b in buys], [s for s in sells])

plt.gcf().set_size_inches(22, 15, forward=True)

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

if len(buys) > 0:
    plt.scatter(list(list(zip(*buys))[0]), list(list(zip(*buys))[1]), c='#00ff00', s=50)
if len(sells) > 0:
    plt.scatter(list(list(zip(*sells))[0]), list(list(zip(*sells))[1]), c='#ff0000', s=50)

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted', 'Buy', 'Sell'])
plt.show()
