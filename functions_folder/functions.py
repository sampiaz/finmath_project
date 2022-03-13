import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def DisplaySizing():
    def logistic(x, L=1, x_0=5, k=1):
        return L / (1 + np.exp(-k * (x - x_0)))

    df = pd.DataFrame()
    df['x'] = range(0, 10, 1)
    df['x'] = df['x']
    df['linearSizing'] = df['x']
    df['sigmoidSizing'] = df['x'].apply(logistic) * 10
    df.set_index('x', inplace = True)
    df.plot()

def DisplayCorrelations(test_set):
    print('\n Correlations between the spread between implied and realized volatility and change in straddle price \n')

    for tickerName in test_set['ticker'].unique():
        df = test_set[test_set['ticker'] == tickerName]

        model = sm.OLS(df['volatility_edge'], df['straddle_change'])
        results = model.fit()

        plt.scatter(df['volatility_edge'], df['straddle_change'], label = tickerName)
        print('{}: Correlation {:.4f}, Beta {:.4f}'.format(tickerName, df['volatility_edge'].corr(df['straddle_change']), results.params[0]))

    print('Overall Strategy correlation: {:.4f}'.format(test_set['volatility_edge'].corr(test_set['straddle_change'])))
    model = sm.OLS(test_set['volatility_edge'], test_set['straddle_change'])
    results = model.fit()
    print('Overall Beta: {:.5f}'.format(results.params[0]))
    plt.title('Vol Edge and Straddle Change Correlations')
    plt.legend(loc="upper center", bbox_to_anchor=(1.25, 1.15), ncol=1)
    plt.show()

def Backtest(df):
    df['shortPnL'] = 0
    df['longPnL'] = 0
    for index, row in df.iterrows():
        ### implied exceeds realized, we sell the straddle
        if row['volatility_edge'] > 0:
            df.at[index, 'shortPnL'] += -1 * df.at[index, 'straddle_change'] * 100  
        else:
            ##realized exceeds implied, we buy the straddle
            df.at[index, 'longPnL'] += df.at[index, 'straddle_change'] * 100  
        df.at[index, 'totalPnL'] = df.at[index, 'shortPnL'] + df.at[index, 'longPnL']
    table = pd.DataFrame()
    for tickerName in df['ticker'].unique():
            sub = df[df['ticker'] == tickerName]
            table = table.append({'ticker': tickerName, 'Long PnL': sub['longPnL'].sum(), 'Short PnL': sub['shortPnL'].sum(), 'Total PnL': sub['longPnL'].sum() + sub['shortPnL'].sum()}, ignore_index = True)

    display(table.set_index('ticker').sort_values('Total PnL', ascending=False))

    print('PnL from shorts: {:.2f}'.format(df['shortPnL'].sum()))
    print('PnL from longs {:.2f}'.format(df['longPnL'].sum()))
    print('Overall Sharpe Ratio {:.2f}'.format(df['totalPnL'].mean()/ df['totalPnL'].std()))

class Earnings_strat():
    
    def __init__(self, filing_dates_df, option_data_df, adj_close_df, vola_window):
        self.filing_dates_df = filing_dates_df
        self.option_data_df = option_data_df
        self.adj_close_df = adj_close_df
        self.vola_window = vola_window
        self.data_cleaned = False
        
    def __calc_rolling_vola(self):
        
        close_vola_df = pd.DataFrame()
        
        for t in self.adj_close_df.ticker.unique():
            df = self.adj_close_df[self.adj_close_df['ticker'] == t]
            df = (df['adj_close'].rolling(self.vola_window).std() * np.sqrt(252)) / df['adj_close']
            close_vola_df = pd.concat([close_vola_df, df], axis = 0)
            
        close_vola_df=close_vola_df.rename(columns={0:"vola_{}day".format(self.vola_window)})
        close_vola_df = pd.concat([self.adj_close_df, close_vola_df], axis = 1)
        
        return close_vola_df
    
    # this function returns the 50 delta calls and puts for exdate immediately after filing_date
    def __filter_opts(self):
                
        self.opt_expiry = {}
        result = pd.DataFrame(columns = ['ticker', 'filing_date', 'exdate'])
        
        # find option expiring after earnings date
        for s in self.option_data_df['ticker'].unique():
        
            filing = pd.DataFrame(self.filing_dates_df[self.filing_dates_df['ticker'] == s])
            options = pd.DataFrame(self.option_data_df[self.option_data_df['ticker'] == s])
    
            options['exdate'] = pd.to_datetime(options['exdate'])
            filing['filing_date'] = pd.to_datetime(filing['filing_date'])
        
            # want options that expire after filing date
            filing['filing_date'] = filing['filing_date'] + pd.Timedelta('1 days')
        
            filing_exdate = []
        
            for d in filing['filing_date']:
                try:
                    date = options[options['exdate'] > d].head(1).exdate.item()
                    
                    result = result.append({'ticker': s, 'filing_date': d, 'exdate': date},
                                          ignore_index = True)
                except ValueError: # error raised if no option expiry after final filing date
                    continue

        result['filing_date'] = pd.to_datetime(result['filing_date'])
        result['exdate'] = pd.to_datetime(result['exdate'])
        self.filing_dates_df = pd.DataFrame(result)
        
        self.option_data_df['exdate'] = pd.to_datetime(self.option_data_df['exdate'])
        
        # merge on exdate just after filing_dates
        self.option_data_df = pd.merge(self.option_data_df, self.filing_dates_df,
                                       on = ['ticker', 'exdate'])
        
        # get options closest to delta=50 for each expiry
        res = pd.DataFrame()
        for s in self.option_data_df['ticker'].unique():
            
            call_d, put_d = {}, {}
            
            df = self.option_data_df[self.option_data_df['ticker'] == s]
        
            df = df.set_index('date')
        
            for d in df.index.unique():
                call_i = (abs(df.loc[d]['delta'] - 0.5)).argmin()
                put_i = (abs(df.loc[d]['delta'] + 0.5)).argmin()
                call_d[d] = df.loc[d].iloc[call_i]
                put_d[d] = df.loc[d].iloc[put_i]
            
            calls = pd.DataFrame(call_d).transpose()
            puts = pd.DataFrame(put_d).transpose()
        
            calls.index = pd.to_datetime(calls.index)
            puts.index = pd.to_datetime(puts.index)
            
            temp = pd.concat([calls, puts], axis = 0)
            res = pd.concat([res, temp], axis = 0)
        
        return res
    
    # clean_data returns the daily 50 delta call and put for each symbol
    # and includes the next filing_date, adj_close, and rolling realized vol
    def clean_data(self):
        close_vola_df = self.__calc_rolling_vola()
        cleaned_opts = self.__filter_opts()
        
        cleaned_opts = cleaned_opts.reset_index()
        cleaned_opts = cleaned_opts.rename(columns = {'index':'date'})
        
        self.atm_opts = pd.merge(cleaned_opts, close_vola_df, on=['date', 'ticker'])
        
        self.data_cleaned = True
        return self.atm_opts
    
    def __is_weekend(self, day):
    
        if (day.weekday() == 6):
            return day + pd.Timedelta(days = 1)
        elif (day.weekday() == 5):
            return day + pd.Timedelta(days = 2)
        else:
            return day
        
    def __is_expiry(self, expiry, exit_date):
        if (exit_date == expiry):
            return (exit_date - pd.Timedelta('1 days'))
        else:
            return exit_date
    
    # returns df with bid/ask 'days' days before earnings and 1 day after earnings
    # as well as implied and realized vol 'days' days before earnings
    # so we can use this df to find a correlation between the two
    def vol_vs_straddle(self, days):
        
        if (not self.data_cleaned):
            print("must run clean_data function first")
            return
        
        opts = pd.DataFrame(self.atm_opts)
        
        opts['days_to_filing'] = opts['filing_date'] - opts['date']
        opts['filing_day_of_week'] = opts['filing_date'].dt.weekday
        opts = opts[opts['days_to_filing'] == '{} days'.format(days)]
        opts['day_after_filing'] = opts['filing_date'] + pd.Timedelta(days=1)
        
        opts = opts.rename(columns = {'date': 'today', 'day_after_filing': 'date'})
        opts['adj_date'] = opts.apply(lambda row: self.__is_weekend(row['date']), axis = 1)
        
        opt_prices = pd.DataFrame(self.option_data_df[['date', 'optionid', 'best_bid',
                                                       'best_offer', 'volume']])
        opt_prices = opt_prices.rename(columns = {'best_bid': 'post_bid', 
                                                  'best_offer': 'post_offer'
                                                 ,'volume': 'post_volume'})
        opts = opts.drop(['date'], axis=1)
        opts = opts.rename(columns={'adj_date':'date'})
        opts['date'] = pd.to_datetime(opts['date'])
        opt_prices['date'] = pd.to_datetime(opt_prices['date'])
        
        # NOTE: here I make the assumption that companies whose options expire the day after
        # the earnings report announce earnings before the open. so if data doesn't look great,
        # comment these lines and try again (you'll have less data)
        opts['adj_date'] = opts.apply(lambda row: self.__is_expiry(row.exdate, row.date), axis = 1)
        opts = opts.drop(['date'], axis=1)
        opts = opts.rename(columns={'adj_date':'date'})

        #---------------------------------------------------------------

        o = pd.merge(opts, opt_prices, on = ['optionid', 'date'])
        
        agg_f = {'exdate': 'first', 'strike_price': 'first', 'best_bid': 'sum', 'best_offer': 'sum',
        'volume': 'mean', 'impl_volatility': 'mean', 'filing_date': 'first', 'adj_close': 'first',
        'post_bid': 'sum', 'post_offer': 'sum', 
        'post_volume': 'mean', 'vola_{}day'.format(self.vola_window): 'mean', 'date':'first'}
        
        out = o.groupby(['today', 'ticker']).aggregate(agg_f)
        out['volatility_edge'] = out['impl_volatility'] - out['vola_{}day'.format(self.vola_window)]
        
        out = out.rename(columns={'date': 'date_after_filing'})
        
        return out
        
    
    