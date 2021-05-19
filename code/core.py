from scipy import stats
import pandas as pd
import plotnine as gg
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
from tqdm import tqdm
import matplotlib as mpl


def setup_plots(font='Arial', font_size=7):
    gg.theme_set(gg.theme_classic() +
                 gg.theme(text=gg.element_text(family=font, size=font_size),
                          strip_background=gg.element_blank()))
    mpl.rcParams.update({'font.size': font_size,
                         'font.family': font,
                         'pdf.fonttype': 42})


def get_predictive_performance(prediction_list, activity_col):
    """From a list of dataframes of predictions for each fold for each datasetset for each guide,
    get spearman correlation of each fold/dataset

    :param prediction_list: list of DataFrame
    :param activity_col: str, column which we're trying to predict
    :return: DataFrame
    """
    test_prediction_df = (pd.concat(prediction_list)
                          .rename({'dataset': 'testing_set'}, axis=1))
    predictive_performance = (test_prediction_df.groupby(['model_name', 'testing_set', 'fold'])
                              .apply(lambda df: stats.spearmanr(df[activity_col], df['prediction'])[0])
                              .reset_index(name='spearman_r'))
    test_set_count = (test_prediction_df[['sgRNA Context Sequence', 'testing_set', 'fold']]
                      .drop_duplicates()
                      .groupby(['testing_set', 'fold'])
                      .agg(n_guides=('sgRNA Context Sequence', 'count'))
                      .reset_index())
    predictive_performance = (predictive_performance.merge(test_set_count, how='inner',
                                                           on=['testing_set', 'fold']))
    predictive_performance['fold'] = predictive_performance['fold'].astype('category')
    predictive_performance['fold_name'] = ('Fold ' + (predictive_performance['fold'].astype(int) + 1).astype(str))
    agg_performance = (predictive_performance.groupby('model_name')
                       .agg(mean_spearman=('spearman_r', 'mean'),
                            std_spearman=('spearman_r', 'std'),
                            median_spearman=('spearman_r', 'median'))
                       .reset_index()
                       .sort_values('mean_spearman', ascending=False)
                       .reset_index(drop=True))
    predictive_performance['model_name'] = pd.Categorical(predictive_performance['model_name'],
                                                            categories=reversed(agg_performance['model_name']))
    return predictive_performance, agg_performance


def plot_spearman_heatmap(predictive_performance, title='',
                         cbar_title='Spearman r',
                         xlabel='Test Set', ylabel='Model',
                         cbar_width=10, cbar_height=10):
    """Plot heatmap of spearman correlations

    :param predictive_performance: DataFrame from `get_predictive_performance`
    :param title: str
    :param cbar_title: str
    :param xlabel: str
    :param ylabel: str
    :param cbar_width: int, in pixels
    :param cbar_height: int, in pixels
    :return: plotnine figure
    """
    g = (gg.ggplot(predictive_performance) +
         gg.aes(x='testing_set', y='model_name', fill='spearman_r') +
         gg.geom_tile(color='black') +
         gg.scale_fill_cmap('RdBu_r', limits=(-1, 1)) +
         gg.guides(fill=gg.guide_colorbar(barwidth=cbar_width, barheight=cbar_height,
                                          title=cbar_title, raster=True)) +
         gg.theme(axis_text_x=gg.element_text(angle=90, hjust=0.5, vjust=1)) +
         gg.facet_wrap('fold_name') +
         gg.xlab(xlabel) +
         gg.ylab(ylabel))
    if title:
        g = g + gg.ggtitle(title)
    return g


def plot_model_performance(predictive_performance, title='',
                           legend_title='Fold',
                           xlabel='Model', ylabel='Spearman',
                           point_size=1, wspace=0.3):
    test_set_mean_performance = (predictive_performance.groupby(['testing_set', 'model_name'])
                                 .agg({'spearman_r': 'mean'})
                                 .reset_index())
    g = (gg.ggplot(predictive_performance) +
         gg.aes(x='model_name', y='spearman_r') +
         gg.geom_point(gg.aes(color='fold_name'), size=point_size) +
         gg.geom_point(data=test_set_mean_performance, shape='_', size=5) +
         gg.scale_color_brewer(type='qual', palette='Set2') +
         gg.facet_wrap('testing_set', scales='free_y') +
         gg.theme(axis_text_x=gg.element_text(angle=90, hjust=0.5, vjust=1),
                  subplots_adjust={'wspace': wspace}) +
         gg.xlab(xlabel) +
         gg.ylab(ylabel) +
         gg.guides(color=gg.guide_legend(title=legend_title)))
    if title:
        g = g + gg.ggtitle(title)
    return g


def get_tidy_cv_df(sg_df, random_state=7, y_col='dataset', group_col='target'):
    """Get dataframe where training and testing sets for each fold are concatenated

    :param sg_df: DataFrame with y_col and group_col
    :param random_state: int, for StratifiedGroupKFold
    :param y_col: str
    :param group_col: str
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    tidy_cv_list = []
    for i, (train_index, test_index) in enumerate(sgkf.split(sg_df, sg_df[y_col],
                                                             sg_df[group_col])):
        train_df = sg_df.iloc[train_index, :].copy()
        train_df['cv'] = i
        train_df['train'] = True
        tidy_cv_list.append(train_df)
        test_df = sg_df.iloc[test_index, :].copy()
        test_df['cv'] = i
        test_df['train'] = False
        tidy_cv_list.append(test_df)
    tidy_cv_df = pd.concat(tidy_cv_list)
    return tidy_cv_df


def point_range_plot(df, x, y, ymin, ymax):
    """Create a pointrange plot

    :param df: DataFrame with columns x, y, ymin, ymax
    :param x: str
    :param y: str
    :param ymin: str
    :param ymax: str
    :return: plotnine plot
    """
    g = (gg.ggplot(data=df) +
         gg.aes(x=x, y=y,
                ymin=ymin, ymax=ymax) +
         gg.geom_pointrange() +
         gg.facet_wrap('dataset', scales='free_y') +
         gg.theme_classic() +
         gg.theme(subplots_adjust={'wspace': 0.25},
                  axis_text_x=gg.element_text(angle=45, hjust=1, vjust=1)))
    return g


def calculate_prediction_stats(prediction_df):
    """Calculate statistics from prediction data

    :param prediction_df: DataFrame
    :return: DataFrame
    """
    prediction_stats = (prediction_df
                        .groupby(['dataset', 'model'])
                        .apply(lambda df: pd.Series(
        {'spearman': stats.spearmanr(df['prediction'], df['sgRNA Activity'])[0],
         'pearson': stats.pearsonr(df['prediction'], df['sgRNA Activity'])[0],
         'size': df.shape[0]}))
                        .reset_index())
    return prediction_stats


def get_model_performance(prediction_df, nboots=1000):
    """Evaluate the predictive performance of model

    :param prediction_df: DataFrame, predictions with columns ['sgRNA Activity', 'dataset', 'model', 'prediction']
    :param nboots: int, number of bootstrap resamples
    :return: DataFrame,
    """
    performance_point_estimates =  calculate_prediction_stats(prediction_df)
    bootstrap_prediction_list = []
    for i in tqdm(range(nboots)):
        resampled_predictions = (prediction_df.groupby(['dataset', 'model'])
                                 .sample(frac=1, replace=True,
                                         random_state=i))
        predictive_performance = calculate_prediction_stats(resampled_predictions)
        bootstrap_prediction_list.append(predictive_performance)
    predictive_performance_ci = (pd.concat(bootstrap_prediction_list)
                                 .reset_index(drop=True)
                                 .groupby(['dataset', 'model'])
                                 .agg(pearson_975=('pearson', lambda x: np.percentile(x, 97.5)),
                                      pearson_025=('pearson', lambda x: np.percentile(x, 2.5)),
                                      spearman_975=('spearman', lambda x: np.percentile(x, 97.5)),
                                      spearman_025=('spearman', lambda x: np.percentile(x, 2.5)))
                                 .reset_index()
                                 .merge(performance_point_estimates, how='inner',
                                        on=['dataset', 'model']))
    predictive_performance_ci['dataset_name'] = (predictive_performance_ci['dataset'] + '\n(n = ' +
                                                 predictive_performance_ci['size'].astype(int).astype(str) + ')')
    dataset_size = (predictive_performance_ci[['dataset_name', 'size']].drop_duplicates()
                    .sort_values('size', ascending=False))
    predictive_performance_ci['dataset_name'] = pd.Categorical(predictive_performance_ci['dataset_name'],
                                                               categories=dataset_size['dataset_name'])
    predictive_performance_ci['spearman_rank'] = (predictive_performance_ci.groupby('dataset')
                                                  ['spearman']
                                                  .rank())
    model_avg_rank = (predictive_performance_ci.groupby('model')
                      .agg({'spearman_rank': 'mean'})
                      .reset_index()
                      .sort_values('spearman_rank'))
    predictive_performance_ci['model'] = pd.Categorical(predictive_performance_ci['model'],
                                                        categories=model_avg_rank['model'])
    return predictive_performance_ci
