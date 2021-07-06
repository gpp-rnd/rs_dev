from scipy import stats
import pandas as pd
import plotnine as gg
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
from sklearn import preprocessing
from scipy import stats
from sklearn.base import clone
import gpplot
import matplotlib.pyplot as plt
import seaborn as sns
import sglearn
from datasets import tracr_list


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


def point_range_plot(df, x, y, ymin, ymax, wspace=0.25, xlabel=None, ylabel=None):
    """Create a pointrange plot

    :param df: DataFrame with columns x, y, ymin, ymax
    :param x: str
    :param y: str
    :param ymin: str
    :param ymax: str
    :return: plotnine plot
    """
    if xlabel is None:
        xlabel = x
    if ylabel is None:
        ylabel = y
    g = (gg.ggplot(data=df) +
         gg.aes(x=x, y=y,
                ymin=ymin, ymax=ymax) +
         gg.geom_pointrange() +
         gg.facet_wrap('dataset', scales='free_y') +
         gg.theme(subplots_adjust={'wspace': wspace},
                  axis_text_x=gg.element_text(angle=45, hjust=1, vjust=1)) +
         gg.xlab(xlabel) +
         gg.ylab(ylabel))
    return g


def calculate_prediction_cors(prediction_df):
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


def get_model_correlations(prediction_df, nboots=1000):
    """Evaluate the predictive performance of model

    :param prediction_df: DataFrame, predictions with columns ['sgRNA Activity', 'dataset', 'model', 'prediction']
    :param nboots: int, number of bootstrap resamples
    :return: DataFrame,
    """
    performance_point_estimates = calculate_prediction_cors(prediction_df)
    bootstrap_prediction_list = []
    for i in tqdm(range(nboots)):
        resampled_predictions = (prediction_df.groupby(['dataset', 'model'])
                                 .sample(frac=1, replace=True,
                                         random_state=i))
        predictive_performance = calculate_prediction_cors(resampled_predictions)
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
    model_avg_spearman = (predictive_performance_ci.groupby('model')
                          .agg({'spearman': 'mean'})
                          .reset_index()
                          .sort_values('spearman'))
    predictive_performance_ci['model'] = pd.Categorical(predictive_performance_ci['model'],
                                                        categories=model_avg_spearman['model'])
    return predictive_performance_ci


def calculate_model_rank_loss(prediction_df):
    prediction_df = prediction_df.copy()
    prediction_df['true_rank'] = (prediction_df.groupby(['dataset', 'model', 'sgRNA Target'])
                                  ['sgRNA Activity']
                                  .rank(ascending=False))
    prediction_df['predicted_rank'] = (prediction_df.groupby(['dataset', 'model', 'sgRNA Target'])
                                       ['prediction']
                                       .rank(ascending=False))
    prediction_df['abs_rank_difference'] = (prediction_df['true_rank'] -
                                            prediction_df['predicted_rank']).abs()
    absolute_rank_difference = (prediction_df.groupby(['dataset', 'model'])
                                .agg(avg_abs_rank_diff=('abs_rank_difference', 'mean'),
                                     size=('sgRNA Sequence', 'count'))
                                .reset_index())
    return absolute_rank_difference


def get_model_rank_loss(prediction_df, nboots=1000):
    point_estimate = calculate_model_rank_loss(prediction_df)
    bootstrap_prediction_list = []
    for i in tqdm(range(nboots)):
        resampled_genes = (prediction_df[['sgRNA Target']]
                           .drop_duplicates()
                           .sample(frac=0.8, replace=False,
                                   random_state=i))
        resampled_predictions = prediction_df.merge(resampled_genes, how='inner',
                                                    on='sgRNA Target')
        predictive_performance = calculate_model_rank_loss(resampled_predictions)
        bootstrap_prediction_list.append(predictive_performance)
    predictive_performance_ci = (pd.concat(bootstrap_prediction_list)
                                 .reset_index(drop=True)
                                 .groupby(['dataset', 'model'])
                                 .agg(avg_abs_rank_diff_975=('avg_abs_rank_diff', lambda x: np.percentile(x, 97.5)),
                                      avg_abs_rank_diff_025=('avg_abs_rank_diff', lambda x: np.percentile(x, 2.5)))
                                 .reset_index()
                                 .merge(point_estimate, how='inner',
                                        on=['dataset', 'model']))
    predictive_performance_ci['dataset_name'] = (predictive_performance_ci['dataset'] + '\n(n = ' +
                                                 predictive_performance_ci['size'].astype(int).astype(str) + ')')
    dataset_size = (predictive_performance_ci[['dataset_name', 'size']].drop_duplicates()
                    .sort_values('size', ascending=False))
    predictive_performance_ci['dataset_name'] = pd.Categorical(predictive_performance_ci['dataset_name'],
                                                               categories=dataset_size['dataset_name'])
    model_avg_rank_diff = (predictive_performance_ci.groupby('model')
                           .agg({'avg_abs_rank_diff': 'mean'})
                           .reset_index()
                           .sort_values('avg_abs_rank_diff', ascending=False))
    predictive_performance_ci['model'] = pd.Categorical(predictive_performance_ci['model'],
                                                        categories=model_avg_rank_diff['model'])
    return predictive_performance_ci



def lollipop_plot(data, cat, val, xlabel=None, ylabel=None):
    if xlabel is None:
        xlabel = val
    if ylabel is None:
        ylabel = cat
    g = (gg.ggplot(data) +
         gg.aes(y=val, ymin=0, ymax=val, x=cat, xend=cat) +
         gg.geom_point(size=4, shape='.') +
         gg.geom_linerange() +
         gg.xlab(ylabel) +  # Flipping before coord flip
         gg.ylab(xlabel) +
         gg.coord_flip())
    return g


def add_xy_line(slope=1, intercept=0, ax=None, linestyle='dashed', color='black', **kwargs):
    """Add line with specified slope and intercept to a scatter plot; Default: y=x line

    Parameters
    ----------
    slope: float
        Value of slope of line to be drawn
    intercept: float
        Value of intercept of line to be drawn
    ax: Axis object, optional
        Plot to add line to
    linestyle: str, optional
        Style of line
    color: str, optional
        Color of line

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()
    x = np.array(ax.get_xlim())
    y = intercept + slope * x
    ax = ax.plot(x, y, linestyle=linestyle, color=color, **kwargs)
    return ax


def add_transform_col(df, col, transform_name, transformed_name, group_col=None):
    df = df.copy()
    transforms = {'z-score': preprocessing.StandardScaler(),
                  'min-max-scaler': preprocessing.MinMaxScaler(),
                  'robust-scaler': preprocessing.RobustScaler(),
                  'quantile-uniform': preprocessing.QuantileTransformer(random_state=0),
                  'quantile-normal': preprocessing.QuantileTransformer(random_state=0, output_distribution='normal'),
                  'yeo-johnson': preprocessing.PowerTransformer(method='yeo-johnson')}
    transform = transforms[transform_name]
    if group_col is None:
        df[transformed_name] = clone(transform).fit_transform(df[[col]])[:, 0]
    else:
        df[transformed_name] = (df.groupby(group_col)
                                [col]
                                .transform(lambda x: clone(transform).fit_transform(x.to_frame())[:, 0]))
    return df


def compare_dataset_transforms(df1, df1_score_col, df1_name,
                               df2, df2_score_col, df2_name,
                               merge_col,
                               df1_group_col=None, df1_avg_col=None,
                               df2_group_col=None, df2_avg_col=None):

    transform_performance = []
    for name in ['z-score', 'min-max-scaler', 'robust-scaler',
                 'quantile-uniform', 'quantile-normal', 'yeo-johnson']:
        print(name)
        df1_score = df1.copy()
        df2_score = df2.copy()
        df1_score = add_transform_col(df1_score, df1_score_col, name, 'score', group_col=df1_group_col)
        df2_score = add_transform_col(df2_score, df2_score_col, name, 'score', group_col=df2_group_col)
        if df1_avg_col:
            df1_score = (df1_score.groupby(df1_avg_col)
                         .agg(score=('score', 'mean'))
                         .reset_index())
        if df2_avg_col:
            df2_score = (df2_score.groupby(df2_avg_col)
                         .agg(score=('score', 'mean'))
                         .reset_index())
        merged_dfs = (df1_score.merge(df2_score, how='inner',
                                      on=merge_col,
                                      suffixes=(df1_name, df2_name)))
        merged_dfs['difference'] = merged_dfs['score' + df1_name] - merged_dfs['score' + df2_name]
        favor_df1 = (merged_dfs['difference'] > 0).sum()
        favor_df2 = (merged_dfs['difference'] < 0).sum()
        absolute_favorability = abs(favor_df1 - favor_df2)
        x = 'score' + df1_name
        y = 'score' + df2_name
        plt.subplots(figsize=(4, 4))
        gpplot.point_densityplot(merged_dfs, x=x,
                                 y=y)
        gpplot.add_correlation(merged_dfs, x=x,
                               y=y, method='pearson')
        gpplot.add_correlation(merged_dfs, x=x,
                               y=y, method='spearman',
                               loc='lower right')
        add_xy_line()
        plt.title(name)
        sns.despine()
        transform_performance.append({'transform': name,
                                      'pearsonr': stats.pearsonr(merged_dfs[x],
                                                                 merged_dfs[y])[0],
                                      'spearmanr': stats.spearmanr(merged_dfs[x],
                                                                   merged_dfs[y])[0],
                                      'absolute_skew': absolute_favorability,
                                      'absolute_difference': merged_dfs['difference'].abs().mean()})
    transform_performance_df = (pd.DataFrame(transform_performance))
    transform_performance_df = transform_performance_df.sort_values('pearsonr', ascending=False)
    return transform_performance_df


def no_clip(ax):
    "Turn off all clipping in axes ax; call immediately before drawing"
    ax.set_clip_on(False)
    artists = []
    artists.extend(ax.collections)
    artists.extend(ax.patches)
    artists.extend(ax.lines)
    artists.extend(ax.texts)
    artists.extend(ax.artists)
    for a in artists:
        a.set_clip_on(False)


def get_feature_df(guide_df, tracrs=None):
    if tracrs is None:
        tracrs = tracr_list
    X = sglearn.featurize_guides(guide_df['sgRNA Context Sequence'])
    for tracr in tracrs:
        X[tracr + ' tracr'] = ((guide_df['tracr'] == tracr)
                               .astype(int)
                               .to_list())
    return X
