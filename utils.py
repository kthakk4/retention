import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## helper function to compare predicted to actual
def retention_plotter(actual,predicted,title='',train_period=''): #assumes actual and predicted start from the same point and are sorted
    #normalize the starting values to 1.0
    actual = [1.0*x/actual[0] for x in actual]
    predicted = [1.0*x/predicted[0] for x in predicted]

    df = pd.DataFrame({'actual':pd.Series(actual), 'predicted':pd.Series(predicted)})
    df['t'] = df.index
    plt.plot('t','actual',data=df,color='skyblue')
    plt.plot('t','predicted',data=df,color='olive',linestyle='dashed')
    plt.legend()
    plt.title(title)

    # draw a vertical line indicating training period if train_period is provided to the helper function
    if train_period != '':
        plt.axvline(x=train_period)