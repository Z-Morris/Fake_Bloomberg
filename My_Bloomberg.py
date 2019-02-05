from dateutil.relativedelta import relativedelta
import matplotlib.font_manager as font_manager
import matplotlib.ticker as mticker
import pandas_datareader as pdr
#import fix_yahoo_finance as yf
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.style as style
from matplotlib.pyplot import figure, show, cm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels import regression
from matplotlib.gridspec import GridSpec
from scipy import stats
from tkinter import *
import requests
import re
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
rcParams['figure.figsize'] = 9, 9

style.use('dark_background')


pd.set_option('display.max_columns', None)
#yf.pdr_override()


def gui_data():
    plt.close()
    plt.close()
    ticker2 = "SPY"
    # start and end date - (Ex: "10/19/2018")
    start_date = e2.get()
    end_date = e3.get()

    ticker1 = e1.get()
    start_data = e2.get()
    startdate = start_date.split("/")
    start_date = datetime(int(startdate[2]), int(startdate[0]), int(startdate[1]))
    enddate = end_date.split("/")
    end_date = datetime(int(enddate[2]), int(enddate[0]), int(enddate[1]))

    df1 = pdr.get_data_yahoo(symbols=ticker1, start= start_date, end=end_date)
    df2 = pdr.get_data_yahoo(symbols=ticker2, start= start_date, end=end_date)

    data1 = df1['Adj Close']
    data2 = df2['Adj Close']

    data1 = data1.pct_change()[1:]
    data2 = data2.pct_change()[1:]

    X = data2.values
    Y = data1.values

    xmin = min(X)
    ymax = max(Y)

    # used for formatting purposes
    numyear = (int(enddate[2]) - int(startdate[2]))
    nummonth = (int(enddate[0]) - int(startdate[0]))
        
    # position of the rsq and beta on the plot
    ymax2 = (sorted(Y)[-3])

    # linear regression returning alpha and beta
    def linear_regression(x,y):
        x = sm.add_constant(x)
        model = regression.linear_model.OLS(y,x).fit()
        x = x[:, 1]
        return model.params[0], model.params[1]
    alpha, beta = linear_regression(X,Y)

    # initialize x
    x1 = np.linspace(X.min(), X.max(), 100)
    y1 = alpha + beta * x1

    # plot the data
    fig = plt.figure(figsize=(5.4,4))
    thismanager = plt.get_current_fig_manager()
    thismanager.window.wm_geometry("+0-1000")

    gs = GridSpec(2, 2, width_ratios=[1,10], height_ratios=[7,1])
    ax_main = plt.subplot(gs[0,1])
    ax_yDist = plt.subplot(gs[0,0],sharey=ax_main)
    ax_xDist = plt.subplot(gs[1,1],sharex=ax_main)

    # add Y histogram
    ax_yDist.hist(Y,data = Y,bins=50,align='mid', orientation='horizontal',color = "#F39F41",edgecolor='black', alpha = 0.7)
    ax_yDist.set(ylabel='{} Returns'.format(ticker1))

    # add X histogram
    ax_xDist.hist(X,data = X,bins=50,orientation='vertical',align='mid',color = "#F39F41",edgecolor='black', alpha = 0.7)
    ax_xDist.set(xlabel='{} Returns'.format(ticker2))

    # add best-fit line
    m, s = stats.norm.fit(X) # get mean and standard deviation
    m2, s2 = stats.norm.fit(Y) # get mean and standard deviation
    lnspc = np.linspace(xmin, max(X), len(X))
    lnspc2 = np.linspace(min(Y), max(Y), len(Y))
    pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval
    pdf_g2 = stats.norm.pdf(lnspc2, m2, s2) # now get theoretical values in our interval  

    # blot best-fit line
    ax_yDist.plot(pdf_g2,lnspc2, linewidth='0.3')

    # blot bet-fite line
    ax_xDist.plot(lnspc, pdf_g, linewidth='0.3')

    # x-axis standard deviation lines
    x_axis_std1v=(m+s)
    x_axis_std2v=(m+(s*2))
    x_axis_std3v=(m+(s*3))
    x_axis_std_1v=(m-s)
    x_axis_std_2v=(m-(s*2))
    x_axis_std_3v=(m-(s*3))

    # y-axis standard deviation lines
    y_axis_std1v=(m2+s2)
    y_axis_std2v=(m2+((s2)*2))
    y_axis_std3v=(m2+((s2)*3))
    y_axis_std_1v=(m2-s2)
    y_axis_std_2v=(m2-((s2)*2))
    y_axis_std_3v=(m2-((s2)*3))

    # add standard deviations
    # x-axis meanline
    ax_xDist.axvline(x=m, linewidth='0.3', ymin=0, ymax=0.95, color='lightgreen').set_linestyle('-')

    # positive standard deviations
    ax_xDist.axvline(x=x_axis_std1v,ymin=0, ymax=0.55, linewidth='0.3', color='lightgreen').set_linestyle('--')
    ax_xDist.axvline(x=x_axis_std2v, ymax = 0.15,  linewidth='0.3', color='lightgreen').set_linestyle('--')
    ax_xDist.axvline(x=x_axis_std3v, ymax= 0, linewidth='0.3', color='lightgreen').set_linestyle('--')
    # negative standard deviations
    ax_xDist.axvline(x=x_axis_std_1v, ymax = 0.55, linewidth='0.3', color='lightgreen').set_linestyle('--')
    ax_xDist.axvline(x=x_axis_std_2v, ymax = 0.15, linewidth='0.3', color='lightgreen').set_linestyle('--')
    ax_xDist.axvline(x=x_axis_std_3v, ymax = 0, linewidth='0.3', color='lightgreen').set_linestyle('--')

    # y-axis mean line
    ax_yDist.axhline(y=m2, linewidth='0.3', xmin=0, xmax=0.95, color='lightgreen').set_linestyle('-')

    # positive standard deviations
    ax_yDist.axhline(y=y_axis_std1v, xmax = 0.55, linewidth='0.3', color='lightgreen').set_linestyle('--')
    ax_yDist.axhline(y=y_axis_std2v, xmax = 0.15, linewidth='0.3', color='lightgreen').set_linestyle('--')
    ax_yDist.axhline(y=y_axis_std3v, xmax = 0, linewidth='0.3', color='lightgreen').set_linestyle('--')
    # negative standard deviations
    ax_yDist.axhline(y=y_axis_std_1v, xmax = 0.55, linewidth='0.3', color='lightgreen').set_linestyle('--')
    ax_yDist.axhline(y=y_axis_std_2v, xmax = 0.15, linewidth='0.3', color='lightgreen').set_linestyle('--')
    ax_yDist.axhline(y=y_axis_std_3v, xmax = 0, linewidth='0.3', color='lightgreen').set_linestyle('--')
        
    # set axis tick labels
    ax_xDist.set_yticklabels([])
    ax_yDist.set_xticklabels([])

    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.setp(ax_main.get_yticklabels(), visible=False)

    # Plot the regression line
    ax_main.plot(x1, y1, 'r', alpha = 0.9, linewidth=2.5, zorder=3)

    # Plot the data (bloomberg style plot)
    ax_main.scatter(X, Y, color = "#F39F41", marker = 'D', edgecolor='black', zorder=2)

    # calculate and plot beta & rsq
    rs = np.corrcoef(data2, data1)
    rs = (rs[1])
    rs = (rs[0])
    rsq = rs **2

    sqformat = '\u00b2'
    rsq = '{:0.2f}'.format(rsq)
    beta ='{:0.2f}'.format(beta)

    # add beta and rsq to the plot
    ax_main.text(xmin, ymax2, "R{0} = ".format(sqformat)+rsq+"\nBeta = "+beta)

    # additional formatting (red markers)
    if float(beta) >=0:
        ax_main.scatter(x=min(x1),y=min(y1), marker='D', color='red', edgecolor='black',linewidth='2', zorder=4, s=25)
        ax_main.scatter(x=max(x1),y=max(y1), marker='D', color='red', edgecolor='black',linewidth='2', zorder=4, s=25)
    else:
        ax_main.scatter(x=max(x1),y=min(y1), marker='D', color='red', edgecolor='black',linewidth='2', zorder=4, s=25)
        ax_main.scatter(x=min(x1),y=max(y1), marker='D', color='red', edgecolor='black',linewidth='2', zorder=4, s=25)

    # add most recent datapoint
    ax_main.scatter(x=(X[-1]),y=(Y[-1]), marker='x', color='red', zorder=5, s=200, alpha = 0.5)
    ax_main.scatter(x=(X[-1]),y=(Y[-1]), marker='+', color='red', zorder=5, s=200, alpha = 0.5)


    # add title and formatting
    if numyear >=1:
        ax_main.set_title("{0} {1}-Year Daily Beta".format(ticker1, numyear))
    else:
        ax_main.set_title("{0} {1}-Month Daily Beta".format(ticker1, nummonth))

    # Customize the plot
    #bloomberg style grid
    ax_main.axhline(0, linewidth =0.5 , linestyle= '-',zorder=1)
    ax_main.axvline(0, linewidth =0.5 , linestyle= '-',zorder=1)
    ax_main.grid(True, color='grey', linewidth = 0.5, linestyle = "--")

    plt.axis('tight')
 #   plt.tight_layout()
    
    # Show the plot

    ## TA
    url = 'https://finance.yahoo.com/quote/{0}?p={0}&.tsrc=fin-srch'.format(ticker1)
    url_data = requests.get(url)
    full_data = (url_data.text)
    full_data_new = re.findall('stock chart, (.*?) stock chart', full_data)
    full_title = full_data_new[0]
    if "&amp;" in full_title:
        full_title = re.sub("&amp;", "&", full_title)
    else:
        pass


    r = pdr.get_data_yahoo(symbols=ticker1, start=start_date, end=end_date)
    def moving_average(x, n, type='simple'):
        """
        compute an n period moving average.
        type is 'simple' | 'exponential'
        """
        x = np.asarray(x)
        if type == 'simple':
            weights = np.ones(n)
        else:
            weights = np.exp(np.linspace(-1., 0., n))

        weights /= weights.sum()

        a = np.convolve(x, weights, mode='full')[:len(x)]
        a[:n] = a[n]
        return a


    def relative_strength(prices, n=14):
        """
        compute the n period relative strength indicator
        http://stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
        http://www.investopedia.com/terms/r/rsi.asp
        """

        deltas = np.diff(prices)
        seed = deltas[:n + 1]
        up = seed[seed >= 0].sum() / n
        down = -seed[seed < 0].sum() / n
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:n] = 100. - 100. / (1. + rs)

        for i in range(n, len(prices)):
            delta = deltas[i - 1]  # cause the diff is 1 shorter

            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (n - 1) + upval) / n
            down = (down * (n - 1) + downval) / n

            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
        return rsi


    def moving_average_convergence(x, nslow=26, nfast=12):
        """
        compute the MACD (Moving Average Convergence/Divergence) using a fast and
        slow exponential moving avg
        return value is emaslow, emafast, macd which are len(x) arrays
        """
        emaslow = moving_average(x, nslow, type='exponential')
        emafast = moving_average(x, nfast, type='exponential')
        return emaslow, emafast, emafast - emaslow


    plt.rc('axes', grid=True)
    plt.rc('grid', color='0.75', linestyle='--', linewidth=0.5)

    textsize = 9
    left, width = 0.1, 0.8
    rect1 = [left, 0.7, width, 0.2]
    rect2 = [left, 0.3, width, 0.4]
    rect3 = [left, 0.1, width, 0.2]


    ## Black Background for technicals chart
    fig = plt.figure(facecolor='black')
    axescolor = 'black'
    ax1 = fig.add_axes(rect1, facecolor=axescolor)  # left, bottom, width, height
    ax2 = fig.add_axes(rect2, facecolor=axescolor, sharex=ax1)
    ax2t = ax2.twinx()
    ax3 = fig.add_axes(rect3, facecolor=axescolor, sharex=ax1)


    # plot the relative strength indicator
    prices = r["Adj Close"]

    rsi = relative_strength(prices)
    fillcolor = 'green'

    ax1.plot(r.index, rsi, color='black')
    ax1.axhline(70, color='red')
    ax1.axhline(30, color='green')
    ax1.fill_between(r.index, rsi, 70, where=(rsi >= 70),
                     facecolor='#ff7f7f', edgecolor='#ff7f7f')
    ax1.fill_between(r.index, rsi, 30, where=(rsi <= 30),
                     facecolor='#7fff7f', edgecolor='#7fff7f')
    ax1.text(0.6, 0.9, '>70 = overbought', va='top',
             transform=ax1.transAxes, fontsize=textsize)
    ax1.text(0.6, 0.1, '<30 = oversold',
             transform=ax1.transAxes, fontsize=textsize)
    ax1.set_ylim(0, 100)
    ax1.set_yticks([30, 70])
    ax1.text(0.025, 0.95, 'RSI (14)', va='top',
             transform=ax1.transAxes, fontsize=textsize)
    ax1.set_title('%s - (%s)' % (full_title, ticker1))

    # plot the price and volume data
    dx = r["Adj Close"] - r.Close
    low = r.Low + dx
    high = r.High + dx

    deltas = np.zeros_like(prices)
    deltas[1:] = np.diff(prices)
    up = deltas > 0
    ax2.vlines(r.index[up], low[up], high[up], color='green', label='_nolegend_')
    ax2.vlines(r.index[~up], low[~up], high[~up], color='red', label='_nolegend_')
    ax2.get_cursor_data(r.index)
    ma50 = moving_average(prices, 50, type='simple')
    ma200 = moving_average(prices, 200, type='simple')

    linema50, = ax2.plot(r.index, ma50, color='#7699bc', lw=2, label='MA (50)')
    linema200, = ax2.plot(r.index, ma200, color='tan', lw=2, label='MA (200)')

    last = r.tail(1)
    s = '%s  O:%1.2f  H:%1.2f  L:%1.2f  C:%1.2f  V:%1.1fM  Chg:%+1.2f' % (
        last.index.strftime('%Y.%m.%d')[0],
        last.Open, last.High,
        last.Low, last.Close,
        last.Volume * 1e-6,
        last.Close - last.Open)
    t4 = ax2.text(0.3, 0.9, s, transform=ax2.transAxes, fontsize=textsize)

    props = font_manager.FontProperties(size=10)
    leg = ax2.legend(loc='center left', shadow=True, fancybox=True, prop=props)
    leg.get_frame().set_alpha(0.5)



    volume = (r.Close * r.Volume) / 1e6  # dollar volume in millions
    vmax = volume.max()
    ax2t.bar(r.index[up],volume[up], width = 1, color='green', label='_nolegend_')
    ax2t.bar(r.index[~up],volume[~up], width = 1, color='red', label='_nolegend_')
    #poly = ax2t.fill_between(r.index, volume, 0, label='Volume',
    #                      facecolor='silver', edgecolor='silver')
    ax2t.set_ylim(0, 5 * vmax)
    ax2t.set_yticks([])

    # compute the MACD indicator
    fillcolor = 'darkslategrey'
    nslow = 26
    nfast = 12
    nema = 9
    emaslow, emafast, macd = moving_average_convergence(
        prices, nslow=nslow, nfast=nfast)
    ema9 = moving_average(macd, nema, type='exponential')
    ax3.plot(r.index, macd, color='black', lw=2)
    ax3.plot(r.index, ema9, color='blue', lw=1)

    macdsign = macd-ema9
    macdpos = macdsign > 0
    ax3.bar(r.index[macdpos], macdsign[macdpos], width=1, color='#7fff7f')
    ax3.bar(r.index[~macdpos], macdsign[~macdpos], width=1, color='#ff7f7f')
    ax3.text(0.025, 0.95, 'MACD (%d, %d, %d)' % (nfast, nslow, nema), va='top', transform=ax3.transAxes, fontsize=textsize)

    #ax3.fill_between(r.index, macd-ema9,where=(macd-ema9>0),
    #                 facecolor='#7fff7f', edgecolor='#7fff7f')
    #ax3.fill_between(r.index, macd-ema9,where=(macd-ema9<0),
    #                 facecolor='#ff7f7f', edgecolor='#ff7f7f')
    #ax3.text(0.025, 0.95, 'MACD (%d, %d, %d)' % (nfast, nslow, nema), va='top',
    #         transform=ax3.transAxes, fontsize=textsize)

    # MACD bull and bear signals
    #ax3.fill_between(r.index, macd, ema9, where=(macd > ema9),
    #                facecolor='#7fff7f', edgecolor='#7fff7f')
    #ax3.fill_between(r.index, macd, ema9, where=(macd < ema9),
    #                facecolor='#ff7f7f', edgecolor='#ff7f7f')

    # ax3.set_yticks([])
    # turn off upper axis tick labels, rotate the lower ones, etc


    for ax in ax1, ax2, ax2t, ax3:
        if ax != ax3:
            for label in ax.get_xticklabels():
                label.set_visible(False)
        else:
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_horizontalalignment('right')

        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    class MyLocator(mticker.MaxNLocator):
        def __init__(self, *args, **kwargs):
            mticker.MaxNLocator.__init__(self, *args, **kwargs)

        def __call__(self, *args, **kwargs):
            return mticker.MaxNLocator.__call__(self, *args, **kwargs)

    # at most 5 ticks, pruning the upper and lower so they don't overlap
    # with other ticks
    # ax2.yaxis.set_major_locator(mticker.MaxNLocator(5, prune='both'))
    # ax3.yaxis.set_major_locator(mticker.MaxNLocator(5, prune='both'))

    ax2.yaxis.set_major_locator(MyLocator(5, prune='both'))
    ax3.yaxis.set_major_locator(MyLocator(5, prune='both'))

    ## black background formatting for stock chart
    ############################################################
    ax1.plot(r.index, rsi, color='white')
    ax3.plot(r.index, macd, color='white', lw=2)
    t4 = ax2.text(0.3, 0.9, s, transform=ax2.transAxes, fontsize=textsize, color='white')
    ax1.set_title('%s - (%s)' % (full_title, ticker1), color='white')
    ax1.text(0.6, 0.9, '>70 = overbought', va='top',transform=ax1.transAxes, fontsize=textsize, color='grey')
    ax1.text(0.6, 0.1, '<30 = oversold',transform=ax1.transAxes, fontsize=textsize, color='grey')
    ax1.text(0.025, 0.95, 'RSI (14)', va='top', transform=ax1.transAxes, fontsize=textsize, color='white')
    ax3.text(0.025, 0.95, 'MACD (%d, %d, %d)' % (nfast, nslow, nema), va='top', transform=ax3.transAxes, fontsize=textsize, color='White')
    #ax1.axhline(0, color='grey', linewidth=7)
    #ax2t.axhline(0,color='grey', linewidth=7)
    for ax in ax1, ax2, ax2t, ax3:
        for label in ax.get_xticklabels():
            label.set_color('grey')
        for label in ax.get_yticklabels():
            label.set_color('grey')

    for label in ax_xDist.get_xticklabels():
        label.set_color('grey')
    for label in ax_yDist.get_yticklabels():
        label.set_color('grey')
    ############################################################
    thismanager = plt.get_current_fig_manager()
    thismanager.window.wm_geometry("+1000+0")
    plt.axis('tight')
#    plt.tight_layout()
    plt.gcf().canvas.set_window_title('{}'.format(ticker1))

    # Key Metrics snapshot
    header = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36","X-Requested-With": "XMLHttpRequest"}
    req = requests.get('https://finance.yahoo.com/quote/{}/key-statistics/'.format(ticker1), headers=header, allow_redirects=False)

    ## destroy previous labels
    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=8, column=6)
    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=8, column=7, stick="E")
    
    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=9, column=7, sticky="E")
    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=9, column=6)    

    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=10, column=6)
    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=11, column=6)
    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=11, column=7, sticky="E")
    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=12, column=7, sticky="E")
    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=13, column=7, sticky="E")
    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=14, column=7, sticky="E")
    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=15, column=7, sticky="E")
    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=16, column=7, sticky="E")
    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=17, column=7, sticky="E")

    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=13, column=6)
    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=14, column=6)

    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=16, column=6)
    Label(master, text="---------------",font=('arial', 12), justify=RIGHT, bg="Black", fg="Black").grid(row=17, column=6)

    if req.status_code == 200:
        comps_tables = pd.read_html(req.text)
        key_metrics = (comps_tables[0])
        prof_metrics = (comps_tables[2])
        return_metrics = (comps_tables[3])
        
        df1 = pd.DataFrame(key_metrics)
        df2 = pd.DataFrame(prof_metrics)
        df3 = pd.DataFrame(return_metrics)
        df4 = pd.DataFrame(comps_tables[4])

        Label(master, text= "Revenue:",font=('arial', 12), bg="Black", fg="#F39F41").grid(row=8, column=6, sticky="W")
        Label(master, text= "EPS:",font=('arial', 12), bg="Black", fg="#F39F41").grid(row=9, column=6, sticky="W")

        Label(master, text= "Rev Growth (YoY):",font=('arial', 12), bg="Black", fg="#F39F41").grid(row=8, column=7, sticky="W")
        Label(master, text= "EPS Growth (YoY):",font=('arial', 12), bg="Black", fg="#F39F41").grid(row=9, column=7, sticky="W")

        Label(master, text="Market Cap:",font=('arial', 12), justify=LEFT, bg="Black", fg="#F39F41").grid(row=10, column=6, sticky="W")
        Label(master, text="Enterprise Value:",font=('arial', 12), justify=LEFT, bg="Black", fg="#F39F41").grid(row=11, column=6, sticky="W")
        Label(master, text="Trailing P/E:",font=('arial', 12), justify=LEFT, bg="Black", fg="#F39F41").grid(row=11, column=7, sticky="W")
        Label(master, text="Forward P/E:",font=('arial', 12), justify=LEFT, bg="Black", fg="#F39F41").grid(row=12, column=7, sticky="W")
        Label(master, text="PEG Ratio (5yr):",font=('arial', 12), justify=LEFT, bg="Black", fg="#F39F41").grid(row=13, column=7, sticky="W")
        Label(master, text="Price/Sales:",font=('arial', 12), justify=LEFT, bg="Black", fg="#F39F41").grid(row=14, column=7, sticky="W")
        Label(master, text="Price/Book:",font=('arial', 12), justify=LEFT, bg="Black", fg="#F39F41").grid(row=15, column=7, sticky="W")
        Label(master, text="EV/Revenue:",font=('arial', 12), justify=LEFT, bg="Black", fg="#F39F41").grid(row=16, column=7, sticky="W")
        Label(master, text="EV/EBITDA:",font=('arial', 12), justify=LEFT, bg="Black", fg="#F39F41").grid(row=17, column=7, sticky="W")

        Label(master, text="Profit Margin:",font=('arial', 12), justify=LEFT, bg="Black", fg="#F39F41").grid(row=13, column=6, sticky="W")
        Label(master, text="Operating Margin:",font=('arial', 12), justify=LEFT, bg="Black", fg="#F39F41").grid(row=14, column=6, sticky="W")

        Label(master, text="Return on Assets:",font=('arial', 12), justify=LEFT, bg="Black", fg="#F39F41").grid(row=16, column=6, sticky="W")
        Label(master, text="Return on Equity:",font=('arial', 12), justify=LEFT, bg="Black", fg="#F39F41").grid(row=17, column=6, sticky="W")
        
        ## create new labels
        Label(master, text=df4.iloc[0][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=8, column=6)
        Label(master, text=df4.iloc[2][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=8, column=7, stick="E")
        
        Label(master, text=df4.iloc[7][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=9, column=7, sticky="E")
        Label(master, text=df4.iloc[6][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=9, column=6)    

        Label(master, text=df1.iloc[0][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=10, column=6)
        Label(master, text=df1.iloc[1][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=11, column=6)
        Label(master, text=df1.iloc[2][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=11, column=7, sticky="E")
        Label(master, text=df1.iloc[3][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=12, column=7, sticky="E")
        Label(master, text=df1.iloc[4][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=13, column=7, sticky="E")
        Label(master, text=df1.iloc[5][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=14, column=7, sticky="E")
        Label(master, text=df1.iloc[6][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=15, column=7, sticky="E")
        Label(master, text=df1.iloc[7][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=16, column=7, sticky="E")
        Label(master, text=df1.iloc[8][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=17, column=7, sticky="E")

        Label(master, text=df2.iloc[0][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=13, column=6)
        Label(master, text=df2.iloc[1][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=14, column=6)

        Label(master, text=df3.iloc[0][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=16, column=6)
        Label(master, text=df3.iloc[1][1],font=('arial', 12), justify=RIGHT, bg="Black", fg="#F39F41").grid(row=17, column=6)
    else:
        pass
    plt.show()
    
master = Tk()
master.title("Homemade Bloomberg")
master.geometry("555x375+0+0")
Label(master, text="Ticker:", font=('arial'), bg="Black",fg="#F39F41").grid(row=2, column=3)
Label(master, text="(Ex: GOOG)", font=('arial', 10, 'italic'), bg="Black", fg="#F39F41").grid(row=3, column=3)
Label(master, text="Start/End Date:", font=('arial'), bg="Black", fg="#F39F41").grid(row=5, column=3)
Label(master, text="(Ex: 10/19/2017)", font=('arial', 10, 'italic'),bg="Black", fg="#F39F41").grid(row=6, column=3)
#Label(master, text="End Date:", font=('arial'), bg="#F39F41").grid(row=5, column=6)
#Label(master, text="(Ex: 10/19/2018)", font=('arial', 11, 'italic'),bg="#F39F41").grid(row=9, column=3)

master.grid_rowconfigure(1, weight=1)
master.grid_rowconfigure(2, weight=0)
master.grid_rowconfigure(3, weight=0)
master.grid_rowconfigure(4, weight=1)
master.grid_rowconfigure(5, weight=0)
master.grid_rowconfigure(7, weight=1)
master.grid_rowconfigure(8, weight=0)
master.grid_rowconfigure(18, weight=1)

e1 = Entry(master,bg="#F39F41", font=('arial', 22, 'bold'))
e2 = Entry(master,bg="#F39F41", font=('arial'))
e3 = Entry(master,bg="#F39F41", font=('arial'), justify=LEFT)

e1.grid(row=2, column=6)
e2.grid(row=5, column=6)
e3.grid(row=5, column=7)

e1.config(highlightbackground="#F39F41")
e2.config(highlightbackground="#F39F41")
e3.config(highlightbackground="#F39F41")

x = Button(master, text='Search', command = gui_data)
x.config(highlightbackground="black")
x.grid(row=2, column=7)
master.lift()
master.attributes('-topmost',True)
master.configure(background='black')

mainloop()
