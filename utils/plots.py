
import matplotlib.pyplot as plt

def plt_long_series(row, x, y):
    plt.style.use('seaborn')

    def my_plot(ax, data1, data2, param_dict):
        
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation(90)

        out = ax.plot_date(data1, data2, **param_dict)
        return out


    fig, ax = plt.subplots(row,1,figsize=(10, 3 * row))

    for i,a in enumerate(ax):
        start = -20 * (len(ax)-i)
        if start+20 == 0:
            end = None
        else :
            end = start+20

        fig.tight_layout()
        my_plot(a, 
                x[start:end], 
                y[start:end],
                {'linestyle':'solid'})

# data_eg = data_eg[::-1]
# row = 5
# x = data_eg.index
# y = data_eg.individual_buy_count
# plt_long_series(row, x, y)