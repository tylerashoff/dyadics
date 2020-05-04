import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from scipy.stats import pearsonr, spearmanr
from ripser import Rips
from dtw import dtw, accelerated_dtw
from datetime import timedelta, date
import pickle
import os


class dyads():

    def __init__(self, dfi, dfp):
        self.dfi = dfi
        self.dfp = dfp
        self.start_date = pd.to_datetime('1955-1-1', format='%Y-%m-%d')
        self.end_date = pd.to_datetime('1978-12-31', format='%Y-%m-%d')

        self.country_codes_i = {
            'USA': ['USA'],
            'USSR': ['USR'],
            'China': ['CHN'],
            'East-Germany': ['GME'],
            'West-Germany': ['GMW'],
            'Canada': ['CAD']
        }

        self.country_codes_p = {
            'USA': ['USA'],
            'USSR': ['SUN', 'RUS'],
            'China': ['CHN'],
            'East-Germany': ['DDR'],
            'West-Germany': ['DEU'],
            'Canada': ['CAN']
        }

        self.complete_dyads = None  # track data with complete data in cor
        pass

    def filter_dates(self, dfi=None, dfp=None):
        '''
        filter by selected date range
        '''

        if dfi is None:
            dfi_filt = self.dfi
            dfp_filt = self.dfp

            # convert to datetime
            dfi_filt['date'] = pd.to_datetime(
                dfi_filt['date'], format='%Y-%m-%d')
            dfp_filt['date'] = pd.to_datetime(
                dfp_filt['story_date'], format='%m/%d/%Y')
            start = self.start_date
            end = self.end_date
            pass

        else:
            dfi_filt = dfi
            dfp_filt = dfp

            start = max(min(dfi_filt.date), min(dfp_filt.date))

            end = min(max(dfi_filt.date), max(dfp_filt.date))
            pass

        # filter by start and end dates
        dfi_filt = dfi_filt[(dfi_filt.date >= start) & (dfi_filt.date <= end)]
        dfp_filt = dfp_filt[(dfp_filt.date >= start) & (dfp_filt.date <= end)]

        return (dfi_filt, dfp_filt)

    def initial_manupulations(self):
        '''
        rename and select columns
        '''
        self.dfp = self.dfp.rename(
            columns={
                'source_root': 'actor',
                'target': 'something_else',
                'target_root': 'target'
            })
        self.dfp = self.dfp[['date', 'actor', 'target', 'goldstein']]

        self.dfi = self.dfi[['date', 'actor', 'target', 'scale']]

        # implement azar weights
        azar_weighting = [
            92, 47, 31, 27, 14, 10, 6, 0, -6, -16, -29, -44, -50, -65, -102
        ]

        self.dfp['score'] = self.dfp['goldstein']
        self.dfi['score'] = [
            azar_weighting[ind - 1] for ind in self.dfi['scale'].to_list()
        ]

        # create time frame designations
        self.dfp['year'] = pd.DatetimeIndex(self.dfp.date).to_period('Y')
        self.dfi['year'] = pd.DatetimeIndex(self.dfi.date).to_period('Y')

        self.dfp['month'] = pd.DatetimeIndex(self.dfp.date).to_period('M')
        self.dfi['month'] = pd.DatetimeIndex(self.dfi.date).to_period('M')

        self.dfp['week'] = pd.DatetimeIndex(self.dfp.date).to_period('W')
        self.dfi['week'] = pd.DatetimeIndex(self.dfi.date).to_period('W')

        pass

    def create_state_pairs(self, states, state_pairs=None):
        '''
        build state pairs for the selected countries
        '''

        for state1 in states:
            states = np.delete(states, np.where(states == state1))
            for state2 in states:
                if state_pairs is None:
                    state_pairs = np.array([[state1, state2]])
                    pass
                else:
                    state_pairs = np.vstack([state_pairs, [state1, state2]])
                    pass
                pass
            pass

        return (state_pairs)

    def dyad_select(self, df, dyad, diction):
        '''
        get the associated country codes for the selected states
        '''
        state1, state2 = [diction[state] for state in dyad]

        selected = [
            df.loc[(df.actor.isin(state1)) & (df.target.isin(state2))],
            df.loc[(df.actor.isin(state2)) & (df.target.isin(state1))]
        ]
        return (pd.concat(selected))

    def backfill(self, scores):
        '''
        support function for backfill_dates method
        '''
        scores = np.array(scores)
        for ind in range(len(scores)):
            if not isinstance(scores[ind], float):
                # back fill with 0 for unseen dates
                scores[ind] = 0  #if ind == 0 else scores[ind-1]
                pass
            pass
        return (scores)

    def backfill_dates(self, dates, scores):
        '''
        fill dates that were not seen with NA
        '''

        for ind in scores.index:
            if ind not in dates:
                scores = scores.drop(ind)
                pass
            pass

        for d in dates:
            if d in scores.index:
                continue
            else:
                scores.at[d] = 'NA'
                pass
            pass
        scores = self.backfill(scores.sort_index())
        return (scores)

    def daterange(self, date1, date2, unit='w'):
        '''
        method to return list of dates at level unit in a range
        '''
        days = []
        for n in range(int((date2 - date1).days) + 1):
            days.append(date1 + timedelta(n))
            pass
        return np.unique([day.to_period(unit[0]) for day in days])

    def extract_scores(self, dyad, unit, check=False):
        '''
        pull scores from raw data for a given dyad
        '''
        dfi_dyad = self.dyad_select(self.dfi, dyad, self.country_codes_i)
        dfp_dyad = self.dyad_select(self.dfp, dyad, self.country_codes_p)

        dfi_filt, dfp_filt = self.filter_dates(dfi=dfi_dyad, dfp=dfp_dyad)

        dates = self.daterange(self.start_date, self.end_date, unit)

        # get directional dyadic scores
        scores_i = dfi_filt.groupby(unit).score.mean()
        scores_p = dfp_filt.groupby(unit).score.mean()

        if check and (unit[0] == 'y' and (len(scores_i) != len(dates) or
                                          len(scores_p) != len(dates))):
            return None

        # scale scores before backfilling
        scores_i = pd.Series(scale(scores_i), index=scores_i.index)
        scores_p = pd.Series(scale(scores_p), index=scores_p.index)

        # backfill scores
        scores_i = self.backfill_dates(dates, scores_i)
        scores_p = self.backfill_dates(dates, scores_p)

        return (scores_i, scores_p)

    def sma(self, scores, wind=3):
        '''
        compute simple moving average for given window
        '''
        sma = []
        for ind in range(len(scores)):
            if ind in range(wind + 1):
                if len(scores[0:ind]) > 0:
                    sma.append(np.mean(scores[0:ind]))
                    pass
                else:
                    sma.append(0)
                pass
            else:
                sma.append(np.mean(scores[ind - (wind - 1):ind + 1]))
                pass
            pass
        return (np.array(sma))

    def clean_correlations(self,
                           dyads,
                           unit,
                           shift=None,
                           sma_wind=None,
                           verbose=False):
        '''
        wrapper for cor method
        '''
        correlations = self.cor(dyads, unit, shift, sma_wind, verbose=verbose)

        dyad_str = np.array(
            [dyad[0] + '-' + dyad[1] for dyad in self.complete_dyads])

        pearson_cor = correlations[:, :, 0][:, 0]
        pearson_p = correlations[:, :, 1][:, 0]
        spearman_cor = correlations[:, :, 0][:, 1]
        spearman_p = correlations[:, :, 1][:, 1]

        correlations_clean = pd.DataFrame({
            'Dyad': dyad_str,
            'Pearson Correlation': pearson_cor,
            'Pearson p-value': pearson_p,
            'Spearman Correlation': spearman_cor,
            'Spearman p-value': spearman_p
        })
        return (correlations_clean)

    def cor(self,
            dyads,
            unit,
            sma_wind=None,
            shift=None,
            correlation_coefs=None,
            verbose=False):
        '''
        recursively find correlations
        '''

        # pass correlation list back up when finished with dyad pairs
        if dyads is None or len(dyads) == 0:
            return correlation_coefs

        if verbose:
            print('\n ---Running correlation on dyad: ', dyads[0])
            pass

        # shift data if necessary
        #dfi, dfp = self.cor_shift(shift, unit)

        extract = self.extract_scores(dyads[0], unit, check=True)

        if extract is None:
            correlation_coefs = self.cor(
                dyads[1:],
                unit,
                shift,
                sma_wind,
                correlation_coefs,
                verbose=verbose)
            return (correlation_coefs)

        scores_i, scores_p = extract

        if sma_wind is not None:
            scores_i = self.sma(scores_i, wind=sma_wind)
            scores_p = self.sma(scores_p, wind=sma_wind)
            pass

        # calculate pearson and spearman correlations and associated p-values
        pr = pearsonr(scores_i, scores_p)
        sr = spearmanr(scores_i, scores_p)

        new_coefs = np.array([[pr, sr]])

        if correlation_coefs is None:
            correlation_coefs = new_coefs
            self.complete_dyads = np.array([dyads[0]])
            pass
        else:
            correlation_coefs = np.vstack([correlation_coefs, new_coefs])
            self.complete_dyads = np.vstack([self.complete_dyads, dyads[0]])
            pass

        # recursive step to continue with dyads
        correlation_coefs = self.cor(
            dyads[1:],
            unit,
            shift,
            sma_wind,
            correlation_coefs,
            verbose=verbose)

        return (correlation_coefs)

    def dyn_timewarp(self, dyads, unit, dtw_vals=None, sma_wind=None):
        '''
        compute dynamic time warping distance
        '''
        if dyads is None or len(dyads) == 0:
            dyad_str = np.array(
                [dyad[0] + '-' + dyad[1] for dyad in self.complete_dyads])
            dtws = pd.DataFrame({'Dyad': dyad_str, 'dtw': dtw_vals})
            return (dtws)

        dfi_score, dfp_score = self.extract_scores(dyads[0], unit)

        if sma_wind is not None:
            dfi_score = self.sma(dfi_score, sma_wind)
            dfp_score = self.sma(dfp_score, sma_wind)
            pass

        dtw = accelerated_dtw(dfi_score, dfp_score, dist='euclidean')

        dtw_val = dtw[0] / len(dfi_score)  # normalize distance

        if dtw_vals is None:
            dtw_vals = [dtw_val]
            pass
        else:
            dtw_vals.append(dtw_val)
            pass

        dtws = self.dyn_timewarp(dyads[1:], unit, dtw_vals, sma_wind=sma_wind)
        return (dtws)

    def run_corrs(self, state_pairs):
        '''
        find correlations at all time levels
        '''

        print('Years', end='\r')
        self.corrs_year = self.clean_correlations(
            state_pairs, 'year', verbose=False)
        print('Months', end='\r')
        self.corrs_month = self.clean_correlations(self.complete_dyads, 'month')

        self.corrs_month_2 = self.clean_correlations(
            self.complete_dyads, 'month', sma_wind=2)
        self.corrs_month_3 = self.clean_correlations(
            self.complete_dyads, 'month', sma_wind=3)

        self.corrs_month_6 = self.clean_correlations(
            self.complete_dyads, 'month', sma_wind=6)
        print('Weeks', end='\r')
        self.corrs_week = self.clean_correlations(self.complete_dyads, 'week')
        self.corrs_week_2 = self.clean_correlations(
            self.complete_dyads, 'week', sma_wind=2)
        self.corrs_week_3 = self.clean_correlations(
            self.complete_dyads, 'week', sma_wind=3)

        print('Corrs done')
        pass

    def run_dtws(self):
        '''
        find dtw distance at all time levels
        '''

        print('Years', end='\r')
        self.dtws_year = self.dyn_timewarp(self.complete_dyads, 'year')

        print('Months', end='\r')
        self.dtws_month = self.dyn_timewarp(self.complete_dyads, 'month')
        self.dtws_month_2 = self.dyn_timewarp(
            self.complete_dyads, 'month', sma_wind=2)
        self.dtws_month_3 = self.dyn_timewarp(
            self.complete_dyads, 'month', sma_wind=3)
        self.dtws_month_6 = self.dyn_timewarp(
            self.complete_dyads, 'month', sma_wind=6)

        print('Weeks', end='\r')
        self.dtws_week = self.dyn_timewarp(self.complete_dyads, 'week')
        self.dtws_week_2 = self.dyn_timewarp(
            self.complete_dyads, 'week', sma_wind=2)
        self.dtws_week_3 = self.dyn_timewarp(
            self.complete_dyads, 'week', sma_wind=3)
        print('DTW done')
        pass

    def plotting_by_unit(self,
                         dyads,
                         unit,
                         thresh=None,
                         sma_wind=None,
                         selfs=None,
                         plotted=None,
                         show_plots=True):

        if dyads is None or len(dyads) == 0:
            return (plotted)

        dfi_score, dfp_score = self.extract_scores(dyads[0], unit)

        index_i = np.arange(len(dfi_score))
        index_p = np.arange(len(dfp_score))

        if unit[0] == 'y':
            index_i += 1955
            index_p += 1955
            pass

        x_i, y_i = index_i, dfi_score
        x_p, y_p = index_p, dfp_score

        if thresh is not None:
            y_i, x_i = self.topological_filtering(dfi_score, thresh=thresh[0])
            y_p, x_p = self.topological_filtering(dfp_score, thresh=thresh[1])
            pass
        elif sma_wind is not None:
            y_i = self.sma(dfi_score, wind=sma_wind)
            y_p = self.sma(dfp_score, wind=sma_wind)
            pass

        if selfs is not None:
            if selfs[0] == 'i':
                plotted_coords = [[index_i, dfi_score], [x_i, y_i]]
                if show_plots:
                    plt.plot(index_i, dfi_score)
                    plt.plot(x_i, y_i)
                    pass
                pass
            elif selfs[0] == 'p':
                plotted_coords = [[index_p, dfp_score], [x_p, y_p]]
                if show_plots:
                    plt.plot(index_p, dfp_score)
                    plt.plot(x_p, y_p)
                    pass
                pass
            pass
        else:
            plotted_coords = [[x_i, y_i], [x_p, y_p]]
            if show_plots:
                plt.plot(x_i, y_i)
                plt.plot(x_p, y_p)
                pass
            pass

        if show_plots:
            plt.title(str(dyads[0]))
            plt.show()
            pass

        if plotted is None:
            plotted = {str(dyads[0]): plotted_coords}
            pass
        else:
            plotted[str(dyads[0])] = plotted_coords
            pass

        plotted = self.plotting_by_unit(
            dyads[1:],
            unit,
            thresh,
            sma_wind,
            selfs,
            plotted,
            show_plots=show_plots)

        return (plotted)

    def run_plotting(self, show=True):

        print('Years', end='\r')
        self.plot_year = self.plotting_by_unit(
            self.complete_dyads, 'year', show_plots=show)

        print('Months', end='\r')
        self.plot_month = self.plotting_by_unit(
            self.complete_dyads, 'month', show_plots=show)
        self.plot_month_2 = self.plotting_by_unit(
            self.complete_dyads, 'month', sma_wind=2, show_plots=show)
        self.plot_month_3 = self.plotting_by_unit(
            self.complete_dyads, 'month', sma_wind=3, show_plots=show)
        self.plot_month_6 = self.plotting_by_unit(
            self.complete_dyads, 'month', sma_wind=6, show_plots=show)

        print('Weeks', end='\r')
        self.plot_week = self.plotting_by_unit(
            self.complete_dyads, 'week', show_plots=show)
        self.plot_week_2 = self.plotting_by_unit(
            self.complete_dyads, 'week', sma_wind=2, show_plots=show)
        self.plot_week_3 = self.plotting_by_unit(
            self.complete_dyads, 'week', sma_wind=3, show_plots=show)
        print('Plots done')

        pass

    def outputs(self):
        '''
        format output variables
        '''
        outputs = {
            'complete_dyads': self.complete_dyads,
            # corrs
            'corrs_year': self.corrs_year,
            'corrs_month': self.corrs_month,
            'corrs_month_2': self.corrs_month_2,
            'corrs_month_3': self.corrs_month_3,
            'corrs_month_6': self.corrs_month_6,
            'corrs_week': self.corrs_week,
            'corrs_week_2': self.corrs_week_2,
            'corrs_week_3': self.corrs_week_3,
            # dtws
            'dtws_year': self.dtws_year,
            'dtws_month': self.dtws_month,
            'dtws_month_2': self.dtws_month_2,
            'dtws_month_3': self.dtws_month_3,
            'dtws_month_6': self.dtws_month_6,
            'dtws_week': self.dtws_week,
            'dtws_week_2': self.dtws_week_2,
            'dtws_week_3': self.dtws_week_3,
            # plotting
            'plot_year': self.plot_year,
            'plot_month': self.plot_month,
            'plot_month_2': self.plot_month_2,
            'plot_month_3': self.plot_month_3,
            'plot_month_6': self.plot_month_6,
            'plot_week': self.plot_week,
            'plot_week_2': self.plot_week_2,
            'plot_week_3': self.plot_week_3
        }
        return (outputs)

    def main(self, tell_me_whe_done=False):

        # Select specific dyads
        states = np.array(
            ['USA', 'USSR', 'China', 'East-Germany', 'West-Germany', 'Canada'])
        state_pairs = self.create_state_pairs(states)

        self.dfi, self.dfp = self.filter_dates()
        self.initial_manupulations()

        self.run_corrs(state_pairs)
        self.run_dtws()
        self.run_plotting(False)
        """
        # topological filtering
        thresh = np.array([[0.01, 0.3], [0.1, 0.5]])
        #self.plotting_by_unit(self.complete_dyads, 'month')
        #self.plotting_by_unit(self.complete_dyads, 'month', thresh = thresh)
        """

        outputs = self.outputs()

        if tell_me_whe_done:
            os.system('say "hey, you are done you frickin genius"')
            pass

        return (outputs)

    pass


if not os.getcwd()[-4:] == 'code':
    os.chdir('./code')
    from rework import clean_icpsr
    os.chdir('..')
    pass
else:
    from rework import clean_icpsr
    pass

# clean data
#clean_icpsr()

# read data
dfi = pd.read_csv('data/icpsr.csv')
dfp = pd.read_csv('data/phoenix.csv')

dys = dyads(dfi, dfp)
output = dys.main(tell_me_whe_done=True)

pickle.dump(output, open('./code/dyads.p', 'wb'))
pickle.dump(output, open('./writing/final/dyads.p', 'wb'))
