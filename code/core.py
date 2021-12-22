import pandas as pd
import plotnine as gg
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
from sklearn import preprocessing
from sklearn.model_selection import StratifiedGroupKFold
from scipy import stats
from sklearn.base import clone
import gpplot
import matplotlib.pyplot as plt
import seaborn as sns
import sglearn
from datasets import tracr_list
from scipy.stats import gaussian_kde
from shap.plots._labels import labels
from shap.plots import colors
from shap.utils._general import encode_array_if_needed
from shap.utils import approximate_interactions, convert_name
import warnings
import networkx as nx
from networkx.drawing.layout import _process_params, rescale_layout


def setup_plots(font='Arial', font_size=7, title_size=8.2):
    gg.theme_set(gg.theme_classic() +
                 gg.theme(text=gg.element_text(family=font, size=font_size),
                          plot_title=gg.element_text(family=font, size=title_size),
                          axis_text=gg.element_text(color='black'),
                          strip_background=gg.element_blank(),
                          plot_background=gg.element_blank()))
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
         gg.scale_fill_cmap('plasma', limits=(0, 1)) +
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


def point_range_plot(df, x, y, ymin, ymax, wspace=0.25, xlabel=None, ylabel=None, facet='dataset',
                     color=None):
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
    if color is not None:
        g = (gg.ggplot(data=df) +
             gg.aes(x=x, y=y,
                    ymin=ymin, ymax=ymax, color=color) +
             gg.scale_color_brewer(type='qual', palette='Set2') +
             gg.geom_pointrange() +
             gg.facet_wrap(facet, scales='free_y') +
             gg.theme(subplots_adjust={'wspace': wspace},
                      axis_text_x=gg.element_text(angle=45, hjust=1, vjust=1)) +
             gg.xlab(xlabel) +
             gg.ylab(ylabel))
    else:
        g = (gg.ggplot(data=df) +
             gg.aes(x=x, y=y,
                    ymin=ymin, ymax=ymax) +
             gg.geom_pointrange() +
             gg.facet_wrap(facet, scales='free_y') +
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



def lollipop_plot(data, cat, val, val_label=None, cat_label=None):
    if val_label is None:
        val_label = val
    if cat_label is None:
        cat_label = cat
    g = (gg.ggplot(data) +
         gg.aes(y=val, ymin=0, ymax=val, x=cat, xend=cat) +
         gg.geom_point(size=4, shape='.') +
         gg.geom_linerange() +
         gg.xlab(cat_label) +  # Flipping before coord flip
         gg.ylab(val_label) +
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


def get_performance_type_df(training_group_dict, predictive_performance_df,
                            agg_performance_df):
    training_group_df = pd.DataFrame(training_group_dict)
    all_performance_type = predictive_performance_df.merge(training_group_df,
                                                            how='inner', on='model_name')
    all_performance_type['model_name'] = pd.Categorical(all_performance_type['model_name'],
                                                        categories=agg_performance_df['model_name'])
    return all_performance_type


def plot_performance_type_df(performance_type_df, legend_title,
                             figsize=(2.3, 2.3),
                             ylabel='', xlabel='Spearman r',
                             title='Model Cross-Validation Performance'):
    plt.subplots(figsize=figsize)
    sns.pointplot(data=performance_type_df, x='spearman_r', y='model_name',
                  hue='model_type', join=False, palette='Set2',
                  scale=0.5, hue_order=['all', 'single',
                                        'leave one out'])
    sns.despine()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend(loc="upper left", bbox_to_anchor=(0, 1),
               title=legend_title)


def shorten_text(text, length_limit):
    if len(text) > length_limit:
        return text[:length_limit - 3] + "..."
    else:
        return text


def summary_legacy(shap_values, features=None, feature_names=None, max_display=None, plot_type=None,
                   color=None, axis_color="#333333", title=None, alpha=1, show=True, sort=True,
                   color_bar=True, plot_size="auto", layered_violin_max_num_bins=20, class_names=None,
                   class_inds=None,
                   color_bar_label=labels["FEATURE_VALUE"], color_bar_shrink=1,
                   cmap=colors.red_blue, point_size=16, text_size=13,
                   # depreciated
                   auto_size_plot=None,
                   use_log_scale=False,
                   legend_aspect=20):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.
    Parameters
    ----------
    shap_values : numpy.array
        For single output explanations this is a matrix of SHAP values (# samples x # features).
        For multi-output explanations this is a list of such matrices of SHAP values.
    features : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a feature_names list as shorthand
    feature_names : list
        Names of the features (length # features)
    max_display : int
        How many top features to include in the plot (default is 20, or 7 for interaction plots)
    plot_type : "dot" (default for single output), "bar" (default for multi-output), "violin",
        or "compact_dot".
        What type of summary plot to produce. Note that "compact_dot" is only used for
        SHAP interaction values.
    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default the size is auto-scaled based on the number of
        features that are being displayed. Passing a single float will cause each row to be that
        many inches high. Passing a pair of floats will scale the plot by that
        number of inches. If None is passed then the size of the current figure will be left
        unchanged.
    """

    # support passing an explanation object
    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        base_value = shap_exp.base_values
        shap_values = shap_exp.values
        if features is None:
            features = shap_exp.data
        if feature_names is None:
            feature_names = shap_exp.feature_names
        # if out_names is None: # TODO: waiting for slicer support of this
        #     out_names = shap_exp.output_names

    # deprecation warnings
    if auto_size_plot is not None:
        warnings.warn("auto_size_plot=False is deprecated and is now ignored! Use plot_size=None instead.")

    multi_class = False
    if isinstance(shap_values, list):
        multi_class = True
        if plot_type is None:
            plot_type = "bar" # default for multi-output explanations
        assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
    else:
        if plot_type is None:
            plot_type = "dot" # default for single output explanations
        assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

    # default color:
    if color is None:
        if plot_type == 'layered_violin':
            color = "coolwarm"
        elif multi_class:
            color = lambda i: colors.red_blue_circle(i/len(shap_values))
        else:
            color = colors.blue_rgb

    idx2cat = None
    # convert from a DataFrame or other types
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns
        # feature index to category flag
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])

    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the " \
                    "provided data matrix."
        if num_features - 1 == features.shape[1]:
            assert False, shape_msg + " Perhaps the extra column in the shap_values matrix is the " \
                          "constant offset? Of so just pass shap_values[:,:-1]."
        else:
            assert num_features == features.shape[1], shape_msg

    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

    if use_log_scale:
        plt.xscale('symlog')

    # plotting SHAP interaction values
    if not multi_class and len(shap_values.shape) == 3:

        if plot_type == "compact_dot":
            new_shap_values = shap_values.reshape(shap_values.shape[0], -1)
            new_features = np.tile(features, (1, 1, features.shape[1])).reshape(features.shape[0], -1)

            new_feature_names = []
            for c1 in feature_names:
                for c2 in feature_names:
                    if c1 == c2:
                        new_feature_names.append(c1)
                    else:
                        new_feature_names.append(c1 + "* - " + c2)

            return summary_legacy(
                new_shap_values, new_features, new_feature_names,
                max_display=max_display, plot_type="dot", color=color, axis_color=axis_color,
                title=title, alpha=alpha, show=show, sort=sort,
                color_bar=color_bar, plot_size=plot_size, class_names=class_names,
                color_bar_label="*" + color_bar_label
            )

        if max_display is None:
            max_display = 7
        else:
            max_display = min(len(feature_names), max_display)

        sort_inds = np.argsort(-np.abs(shap_values.sum(1)).sum(0))

        # get plotting limits
        delta = 1.0 / (shap_values.shape[1] ** 2)
        slow = np.nanpercentile(shap_values, delta)
        shigh = np.nanpercentile(shap_values, 100 - delta)
        v = max(abs(slow), abs(shigh))
        slow = -v
        shigh = v

        plt.figure(figsize=(1.5 * max_display + 1, 0.8 * max_display + 1))
        plt.subplot(1, max_display, 1)
        proj_shap_values = shap_values[:, sort_inds[0], sort_inds]
        proj_shap_values[:, 1:] *= 2  # because off diag effects are split in half
        summary_legacy(
            proj_shap_values, features[:, sort_inds] if features is not None else None,
            feature_names=feature_names[sort_inds],
            sort=False, show=False, color_bar=False,
            plot_size=None,
            max_display=max_display
        )
        plt.xlim((slow, shigh))
        plt.xlabel("")
        title_length_limit = 11
        plt.title(shorten_text(feature_names[sort_inds[0]], title_length_limit))
        for i in range(1, min(len(sort_inds), max_display)):
            ind = sort_inds[i]
            plt.subplot(1, max_display, i + 1)
            proj_shap_values = shap_values[:, ind, sort_inds]
            proj_shap_values *= 2
            proj_shap_values[:, i] /= 2  # because only off diag effects are split in half
            summary_legacy(
                proj_shap_values, features[:, sort_inds] if features is not None else None,
                sort=False,
                feature_names=["" for i in range(len(feature_names))],
                show=False,
                color_bar=False,
                plot_size=None,
                max_display=max_display
            )
            plt.xlim((slow, shigh))
            plt.xlabel("")
            if i == min(len(sort_inds), max_display) // 2:
                plt.xlabel(labels['INTERACTION_VALUE'])
            plt.title(shorten_text(feature_names[ind], title_length_limit))
        plt.tight_layout(pad=0, w_pad=0, h_pad=0.0)
        plt.subplots_adjust(hspace=0, wspace=0.1)
        if show:
            plt.show()
        return

    if max_display is None:
        max_display = 20

    if sort:
        # order features by the sum of their effect magnitudes
        if multi_class:
            feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
        else:
            feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)):]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    row_height = 0.4
    if plot_size == "auto":
        plt.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        plt.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        plt.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)
    plt.axvline(x=0, color="#999999", zorder=-1)

    if plot_type == "dot":
        for pos, i in enumerate(feature_order):
            plt.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
            shaps = shap_values[:, i]
            values = None if features is None else features[:, i]
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            if values is not None:
                values = values[inds]
            shaps = shaps[inds]
            colored_feature = True
            try:
                if idx2cat is not None and idx2cat[i]: # check categorical feature
                    colored_feature = False
                else:
                    values = np.array(values, dtype=np.float64)  # make sure this can be numeric
            except:
                colored_feature = False
            N = len(shaps)
            # hspacing = (np.max(shaps) - np.min(shaps)) / 200
            # curr_bin = []
            nbins = 100
            quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (row_height / np.max(ys + 1))

            if features is not None and colored_feature:
                # trim the color range, but prevent the color range from collapsing
                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)
                if vmin > vmax: # fixes rare numerical precision issues
                    vmin = vmax

                assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

                # plot the nan values in the interaction feature as grey
                nan_mask = np.isnan(values)
                plt.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                           vmax=vmax, s=point_size, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)

                # plot the non-nan values colored by the trimmed feature value
                cvals = values[np.invert(nan_mask)].astype(np.float64)
                cvals_imp = cvals.copy()
                cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                cvals[cvals_imp > vmax] = vmax
                cvals[cvals_imp < vmin] = vmin
                plt.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                           cmap=cmap, vmin=vmin, vmax=vmax, s=point_size,
                           c=cvals, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)
            else:

                plt.scatter(shaps, pos + ys, s=point_size, alpha=alpha, linewidth=0, zorder=3,
                           color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)

    elif plot_type == "violin":
        for pos, i in enumerate(feature_order):
            plt.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

        if features is not None:
            global_low = np.nanpercentile(shap_values[:, :len(feature_names)].flatten(), 1)
            global_high = np.nanpercentile(shap_values[:, :len(feature_names)].flatten(), 99)
            for pos, i in enumerate(feature_order):
                shaps = shap_values[:, i]
                shap_min, shap_max = np.min(shaps), np.max(shaps)
                rng = shap_max - shap_min
                xs = np.linspace(np.min(shaps) - rng * 0.2, np.max(shaps) + rng * 0.2, 100)
                if np.std(shaps) < (global_high - global_low) / 100:
                    ds = gaussian_kde(shaps + np.random.randn(len(shaps)) * (global_high - global_low) / 100)(xs)
                else:
                    ds = gaussian_kde(shaps)(xs)
                ds /= np.max(ds) * 3

                values = features[:, i]
                window_size = max(10, len(values) // 20)
                smooth_values = np.zeros(len(xs) - 1)
                sort_inds = np.argsort(shaps)
                trailing_pos = 0
                leading_pos = 0
                running_sum = 0
                back_fill = 0
                for j in range(len(xs) - 1):

                    while leading_pos < len(shaps) and xs[j] >= shaps[sort_inds[leading_pos]]:
                        running_sum += values[sort_inds[leading_pos]]
                        leading_pos += 1
                        if leading_pos - trailing_pos > 20:
                            running_sum -= values[sort_inds[trailing_pos]]
                            trailing_pos += 1
                    if leading_pos - trailing_pos > 0:
                        smooth_values[j] = running_sum / (leading_pos - trailing_pos)
                        for k in range(back_fill):
                            smooth_values[j - k - 1] = smooth_values[j]
                    else:
                        back_fill += 1

                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)

                # plot the nan values in the interaction feature as grey
                nan_mask = np.isnan(values)
                plt.scatter(shaps[nan_mask], np.ones(shap_values[nan_mask].shape[0]) * pos,
                           color="#777777", vmin=vmin, vmax=vmax, s=9,
                           alpha=alpha, linewidth=0, zorder=1)
                # plot the non-nan values colored by the trimmed feature value
                cvals = values[np.invert(nan_mask)].astype(np.float64)
                cvals_imp = cvals.copy()
                cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                cvals[cvals_imp > vmax] = vmax
                cvals[cvals_imp < vmin] = vmin
                plt.scatter(shaps[np.invert(nan_mask)], np.ones(shap_values[np.invert(nan_mask)].shape[0]) * pos,
                           cmap=cmap, vmin=vmin, vmax=vmax, s=9,
                           c=cvals, alpha=alpha, linewidth=0, zorder=1)
                # smooth_values -= nxp.nanpercentile(smooth_values, 5)
                # smooth_values /= np.nanpercentile(smooth_values, 95)
                smooth_values -= vmin
                if vmax - vmin > 0:
                    smooth_values /= vmax - vmin
                for i in range(len(xs) - 1):
                    if ds[i] > 0.05 or ds[i + 1] > 0.05:
                        plt.fill_between([xs[i], xs[i + 1]], [pos + ds[i], pos + ds[i + 1]],
                                        [pos - ds[i], pos - ds[i + 1]], color=colors.red_blue_no_bounds(smooth_values[i]),
                                        zorder=2)

        else:
            parts = plt.violinplot(shap_values[:, feature_order], range(len(feature_order)), points=200, vert=False,
                                  widths=0.7,
                                  showmeans=False, showextrema=False, showmedians=False)

            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor('none')
                pc.set_alpha(alpha)

    elif plot_type == "layered_violin":  # courtesy of @kodonnell
        num_x_points = 200
        bins = np.linspace(0, features.shape[0], layered_violin_max_num_bins + 1).round(0).astype(
            'int')  # the indices of the feature data corresponding to each bin
        shap_min, shap_max = np.min(shap_values), np.max(shap_values)
        x_points = np.linspace(shap_min, shap_max, num_x_points)

        # loop through each feature and plot:
        for pos, ind in enumerate(feature_order):
            # decide how to handle: if #unique < layered_violin_max_num_bins then split by unique value, otherwise use bins/percentiles.
            # to keep simpler code, in the case of uniques, we just adjust the bins to align with the unique counts.
            feature = features[:, ind]
            unique, counts = np.unique(feature, return_counts=True)
            if unique.shape[0] <= layered_violin_max_num_bins:
                order = np.argsort(unique)
                thesebins = np.cumsum(counts[order])
                thesebins = np.insert(thesebins, 0, 0)
            else:
                thesebins = bins
            nbins = thesebins.shape[0] - 1
            # order the feature data so we can apply percentiling
            order = np.argsort(feature)
            # x axis is located at y0 = pos, with pos being there for offset
            y0 = np.ones(num_x_points) * pos
            # calculate kdes:
            ys = np.zeros((nbins, num_x_points))
            for i in range(nbins):
                # get shap values in this bin:
                shaps = shap_values[order[thesebins[i]:thesebins[i + 1]], ind]
                # if there's only one element, then we can't
                if shaps.shape[0] == 1:
                    warnings.warn(
                        "not enough data in bin #%d for feature %s, so it'll be ignored. Try increasing the number of records to plot."
                        % (i, feature_names[ind]))
                    # to ignore it, just set it to the previous y-values (so the area between them will be zero). Not ys is already 0, so there's
                    # nothing to do if i == 0
                    if i > 0:
                        ys[i, :] = ys[i - 1, :]
                    continue
                # save kde of them: note that we add a tiny bit of gaussian noise to avoid singular matrix errors
                ys[i, :] = gaussian_kde(shaps + np.random.normal(loc=0, scale=0.001, size=shaps.shape[0]))(x_points)
                # scale it up so that the 'size' of each y represents the size of the bin. For continuous data this will
                # do nothing, but when we've gone with the unqique option, this will matter - e.g. if 99% are male and 1%
                # female, we want the 1% to appear a lot smaller.
                size = thesebins[i + 1] - thesebins[i]
                bin_size_if_even = features.shape[0] / nbins
                relative_bin_size = size / bin_size_if_even
                ys[i, :] *= relative_bin_size
            # now plot 'em. We don't plot the individual strips, as this can leave whitespace between them.
            # instead, we plot the full kde, then remove outer strip and plot over it, etc., to ensure no
            # whitespace
            ys = np.cumsum(ys, axis=0)
            width = 0.8
            scale = ys.max() * 2 / width  # 2 is here as we plot both sides of x axis
            for i in range(nbins - 1, -1, -1):
                y = ys[i, :] / scale
                c = plt.get_cmap(color)(i / (
                        nbins - 1)) if color in plt.cm.datad else color  # if color is a cmap, use it, otherwise use a color
                plt.fill_between(x_points, pos - y, pos + y, facecolor=c)
        plt.xlim(shap_min, shap_max)

    elif not multi_class and plot_type == "bar":
        feature_inds = feature_order[:max_display]
        y_pos = np.arange(len(feature_inds))
        global_shap_values = np.abs(shap_values).mean(0)
        plt.barh(y_pos, global_shap_values[feature_inds], 0.7, align='center', color=color)
        plt.yticks(y_pos, fontsize=text_size)
        plt.gca().set_yticklabels([feature_names[i] for i in feature_inds])

    elif multi_class and plot_type == "bar":
        if class_names is None:
            class_names = ["Class "+str(i) for i in range(len(shap_values))]
        feature_inds = feature_order[:max_display]
        y_pos = np.arange(len(feature_inds))
        left_pos = np.zeros(len(feature_inds))

        if class_inds is None:
            class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
        elif class_inds == "original":
            class_inds = range(len(shap_values))
        for i, ind in enumerate(class_inds):
            global_shap_values = np.abs(shap_values[ind]).mean(0)
            plt.barh(
                y_pos, global_shap_values[feature_inds], 0.7, left=left_pos, align='center',
                color=color(i), label=class_names[ind]
            )
            left_pos += global_shap_values[feature_inds]
        plt.yticks(y_pos, fontsize=text_size)
        plt.gca().set_yticklabels([feature_names[i] for i in feature_inds])
        plt.legend(frameon=False, fontsize=text_size)

    # draw the color bar
    if color_bar and features is not None and plot_type != "bar" and \
            (plot_type != "layered_violin" or color in plt.cm.datad):
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap=cmap if plot_type != "layered_violin" else plt.get_cmap(color))
        m.set_array([0, 1])
        cb = plt.colorbar(m, ticks=[0, 1], aspect=1000, shrink=color_bar_shrink)
        cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
        cb.set_label(color_bar_label, size=text_size, labelpad=0)
        cb.ax.tick_params(labelsize=text_size, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        cb.ax.set_aspect(bbox.height * legend_aspect)
        cb.draw_all()

    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().tick_params(color=axis_color, labelcolor=axis_color)
    plt.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=text_size)
    if plot_type != "bar":
        plt.gca().tick_params('y', length=20, width=0.5, which='major')
    plt.gca().tick_params('x', labelsize=text_size)
    plt.ylim(-1, len(feature_order))
    if plot_type == "bar":
        plt.xlabel(labels['GLOBAL_VALUE'], fontsize=text_size)
    else:
        plt.xlabel(labels['VALUE'], fontsize=text_size)
    if show:
        plt.show()


def dependence_legacy(ind, shap_values=None, features=None, feature_names=None, display_features=None,
                      interaction_index="auto",
                      color="#1E88E5", axis_color="#333333", cmap=None,
                      dot_size=16, x_jitter=0, alpha=1, title=None, xmin=None, xmax=None, ax=None, show=True,
                      text_size=None, default_fig_size=(7.5, 5), legend_aspect=20, nan_width=0):
    """ Create a SHAP dependence plot, colored by an interaction feature.
    Plots the value of the feature on the x-axis and the SHAP value of the same feature
    on the y-axis. This shows how the model depends on the given feature, and is like a
    richer extenstion of the classical parital dependence plots. Vertical dispersion of the
    data points represents interaction effects. Grey ticks along the y-axis are data
    points where the feature's value was NaN.
    Parameters
    ----------
    ind : int or string
        If this is an int it is the index of the feature to plot. If this is a string it is
        either the name of the feature to plot, or it can have the form "rank(int)" to specify
        the feature with that rank (ordered by mean absolute SHAP value over all the samples).
    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features).
    features : numpy.array or pandas.DataFrame
        Matrix of feature values (# samples x # features).
    feature_names : list
        Names of the features (length # features).
    display_features : numpy.array or pandas.DataFrame
        Matrix of feature values for visual display (such as strings instead of coded values).
    interaction_index : "auto", None, int, or string
        The index of the feature used to color the plot. The name of a feature can also be passed
        as a string. If "auto" then shap.common.approximate_interactions is used to pick what
        seems to be the strongest interaction (note that to find to true stongest interaction you
        need to compute the SHAP interaction values).
    x_jitter : float (0 - 1)
        Adds random jitter to feature values. May increase plot readability when feature
        is discrete.
    alpha : float
        The transparency of the data points (between 0 and 1). This can be useful to the
        show density of the data points when using a large dataset.
    xmin : float or string
        Represents the lower bound of the plot's x-axis. It can be a string of the format
        "percentile(float)" to denote that percentile of the feature's value used on the x-axis.
    xmax : float or string
        Represents the upper bound of the plot's x-axis. It can be a string of the format
        "percentile(float)" to denote that percentile of the feature's value used on the x-axis.
    ax : matplotlib Axes object
         Optionally specify an existing matplotlib Axes object, into which the plot will be placed.
         In this case we do not create a Figure, otherwise we do.
    """

    if cmap is None:
        cmap = colors.red_blue

    if type(shap_values) is list:
        raise TypeError("The passed shap_values are a list not an array! If you have a list of explanations try " \
                        "passing shap_values[0] instead to explain the first output class of a multi-output model.")

    # convert from DataFrames if we got any
    if str(type(features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = features.columns
        features = features.values
    if str(type(display_features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = display_features.columns
        display_features = display_features.values
    elif display_features is None:
        display_features = features

    if feature_names is None:
        feature_names = [labels['FEATURE'] % str(i) for i in range(shap_values.shape[1])]

    # allow vectors to be passed
    if len(shap_values.shape) == 1:
        shap_values = np.reshape(shap_values, len(shap_values), 1)
    if len(features.shape) == 1:
        features = np.reshape(features, len(features), 1)

    ind = convert_name(ind, shap_values, feature_names)

    # guess what other feature as the stongest interaction with the plotted feature
    if not hasattr(ind, "__len__"):
        if interaction_index == "auto":
            interaction_index = approximate_interactions(ind, shap_values, features)[0]
        interaction_index = convert_name(interaction_index, shap_values, feature_names)
    categorical_interaction = False

    # create a matplotlib figure, if `ax` hasn't been specified.
    if not ax:
        figsize = default_fig_size if interaction_index != ind and interaction_index is not None else (6, 5)
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    # plotting SHAP interaction values
    if len(shap_values.shape) == 3 and hasattr(ind, "__len__") and len(ind) == 2:
        ind1 = convert_name(ind[0], shap_values, feature_names)
        ind2 = convert_name(ind[1], shap_values, feature_names)
        if ind1 == ind2:
            proj_shap_values = shap_values[:, ind2, :]
        else:
            proj_shap_values = shap_values[:, ind2, :] * 2  # off-diag values are split in half

        # there is no interaction coloring for the main effect
        if ind1 == ind2:
            fig.set_size_inches(6, 5, forward=True)

        # TODO: remove recursion; generally the functions should be shorter for more maintainable code
        dependence_legacy(
            ind1, proj_shap_values, features, feature_names=feature_names,
            interaction_index=(None if ind1 == ind2 else ind2), display_features=display_features, ax=ax, show=False,
            xmin=xmin, xmax=xmax, x_jitter=x_jitter, alpha=alpha
        )
        if ind1 == ind2:
            ax.set_ylabel(labels['MAIN_EFFECT'] % feature_names[ind1])
        else:
            ax.set_ylabel(labels['INTERACTION_EFFECT'] % (feature_names[ind1], feature_names[ind2]))

        if show:
            plt.show()
        return

    assert shap_values.shape[0] == features.shape[0], \
        "'shap_values' and 'features' values must have the same number of rows!"
    assert shap_values.shape[1] == features.shape[1], \
        "'shap_values' must have the same number of columns as 'features'!"

    # get both the raw and display feature values
    oinds = np.arange(
        shap_values.shape[0])  # we randomize the ordering so plotting overlaps are not related to data ordering
    np.random.shuffle(oinds)

    xv = encode_array_if_needed(features[oinds, ind])

    xd = display_features[oinds, ind]
    s = shap_values[oinds, ind]
    if type(xd[0]) == str:
        name_map = {}
        for i in range(len(xv)):
            name_map[xd[i]] = xv[i]
        xnames = list(name_map.keys())

    # allow a single feature name to be passed alone
    if type(feature_names) == str:
        feature_names = [feature_names]
    name = feature_names[ind]

    # get both the raw and display color values
    color_norm = None
    if interaction_index is not None:
        interaction_feature_values = encode_array_if_needed(features[:, interaction_index])
        cv = interaction_feature_values
        cd = display_features[:, interaction_index]
        clow = np.nanpercentile(cv.astype(np.float), 5)
        chigh = np.nanpercentile(cv.astype(np.float), 95)
        if clow == chigh:
            clow = np.nanmin(cv.astype(np.float))
            chigh = np.nanmax(cv.astype(np.float))
        if type(cd[0]) == str:
            cname_map = {}
            for i in range(len(cv)):
                cname_map[cd[i]] = cv[i]
            cnames = list(cname_map.keys())
            categorical_interaction = True
        elif clow % 1 == 0 and chigh % 1 == 0 and chigh - clow < 10:
            categorical_interaction = True

        # discritize colors for categorical features
        if categorical_interaction and clow != chigh:
            clow = np.nanmin(cv.astype(np.float))
            chigh = np.nanmax(cv.astype(np.float))
            bounds = np.linspace(clow, chigh, min(int(chigh - clow + 2), cmap.N - 1))
            color_norm = mpl.colors.BoundaryNorm(bounds, cmap.N - 1)

    # optionally add jitter to feature values
    if x_jitter > 0:
        if x_jitter > 1: x_jitter = 1
        xvals = xv.copy()
        if isinstance(xvals[0], float):
            xvals = xvals.astype(np.float)
            xvals = xvals[~np.isnan(xvals)]
        xvals = np.unique(xvals)  # returns a sorted array
        if len(xvals) >= 2:
            smallest_diff = np.min(np.diff(xvals))
            jitter_amount = x_jitter * smallest_diff
            xv += (np.random.random_sample(size=len(xv)) * jitter_amount) - (jitter_amount / 2)

    # the actual scatter plot, TODO: adapt the dot_size to the number of data points?
    xv_nan = np.isnan(xv)
    xv_notnan = np.invert(xv_nan)
    if interaction_index is not None:

        # plot the nan values in the interaction feature as grey
        cvals = interaction_feature_values[oinds].astype(np.float64)
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (clow + chigh) / 2.0
        cvals[cvals_imp > chigh] = chigh
        cvals[cvals_imp < clow] = clow
        p = ax.scatter(
            xv[xv_notnan], s[xv_notnan], s=dot_size, linewidth=0, c=cvals[xv_notnan],
            cmap=cmap, alpha=alpha, vmin=clow, vmax=chigh,
            norm=color_norm, rasterized=len(xv) > 500
        )
        p.set_array(cvals[xv_notnan])
    else:
        p = ax.scatter(xv, s, s=dot_size, linewidth=0, color=color,
                       alpha=alpha, rasterized=len(xv) > 500)

    if interaction_index != ind and interaction_index is not None:
        # draw the color bar
        if type(cd[0]) == str:
            tick_positions = [cname_map[n] for n in cnames]
            if len(tick_positions) == 2:
                tick_positions[0] -= 0.25
                tick_positions[1] += 0.25
            cb = plt.colorbar(p, ticks=tick_positions, ax=ax)
            cb.set_ticklabels(cnames)
        else:
            cb = plt.colorbar(p, ax=ax)

        cb.set_label(feature_names[interaction_index], size=text_size)
        cb.ax.tick_params(labelsize=text_size)
        if categorical_interaction:
            cb.ax.tick_params(length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        cb.ax.set_aspect(bbox.height * legend_aspect)

    # handles any setting of xmax and xmin
    # note that we handle None,float, or "percentile(float)" formats
    if xmin is not None or xmax is not None:
        if type(xmin) == str and xmin.startswith("percentile"):
            xmin = np.nanpercentile(xv, float(xmin[11:-1]))
        if type(xmax) == str and xmax.startswith("percentile"):
            xmax = np.nanpercentile(xv, float(xmax[11:-1]))

        if xmin is None or xmin == np.nanmin(xv):
            xmin = np.nanmin(xv) - (xmax - np.nanmin(xv)) / 20
        if xmax is None or xmax == np.nanmax(xv):
            xmax = np.nanmax(xv) + (np.nanmax(xv) - xmin) / 20

        ax.set_xlim(xmin, xmax)

    # plot any nan feature values as tick marks along the y-axis
    xlim = ax.get_xlim()
    if interaction_index is not None:
        p = ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=nan_width, c=cvals_imp[xv_nan], cmap=cmap, alpha=alpha,
            vmin=clow, vmax=chigh
        )
        p.set_array(cvals[xv_nan])
    else:
        ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=nan_width, color=color, alpha=alpha
        )
    ax.set_xlim(xlim)

    # make the plot more readable
    ax.set_xlabel(name, color=axis_color, fontsize=text_size)
    ax.set_ylabel(labels['VALUE_FOR'] % name, color=axis_color, fontsize=text_size)
    if title is not None:
        ax.set_title(title, color=axis_color, fontsize=text_size)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labelsize=text_size)
    for spine in ax.spines.values():
        spine.set_edgecolor(axis_color)
    if type(xd[0]) == str:
        ax.set_xticks([name_map[n] for n in xnames])
        ax.set_xticklabels(xnames, dict(rotation='vertical', fontsize=text_size))
    if show:
        with warnings.catch_warnings():  # ignore expected matplotlib warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            plt.show()


def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return mpl.colors.LinearSegmentedColormap('colormap',cdict,1024)


def bipartite_layout(
    G, nodes, align="vertical", scale=1, center=None, aspect_ratio=4 / 3,
    input_nodes_order=None, other_nodes_order=None,
):
    """Position nodes in two straight lines.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    nodes : list or container
        Nodes in one node set of the bipartite graph.
        This set will be placed on left or top.

    align : string (default='vertical')
        The alignment of nodes. Vertical or horizontal.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    aspect_ratio : number (default=4/3):
        The ratio of the width to the height of the layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.bipartite.gnmk_random_graph(3, 5, 10, seed=123)
    >>> top = nx.bipartite.sets(G)[0]
    >>> pos = nx.bipartite_layout(G, top)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """

    G, center = _process_params(G, center=center, dim=2)
    if len(G) == 0:
        return {}

    height = 1
    width = aspect_ratio * height
    offset = (width / 2, height / 2)
    if input_nodes_order is None and other_nodes_order is None:
        top = set(nodes)
        bottom = set(G) - top
        nodes = list(top) + list(bottom)
    else:
        top = [x for x in input_nodes_order if x in set(G)]
        bottom = [x for x in other_nodes_order if x in set(G)]
        nodes = top + bottom

    if align == "vertical":
        left_xs = np.repeat(0, len(top))
        right_xs = np.repeat(width, len(bottom))
        left_ys = np.linspace(0, height, len(top))
        right_ys = np.linspace(0, height, len(bottom))

        top_pos = np.column_stack([left_xs, left_ys]) - offset
        bottom_pos = np.column_stack([right_xs, right_ys]) - offset

        pos = np.concatenate([top_pos, bottom_pos])
        pos = rescale_layout(pos, scale=scale) + center
        pos = dict(zip(nodes, pos))
        return pos

    if align == "horizontal":
        top_ys = np.repeat(height, len(top))
        bottom_ys = np.repeat(0, len(bottom))
        top_xs = np.linspace(0, width, len(top))
        bottom_xs = np.linspace(0, width, len(bottom))

        top_pos = np.column_stack([top_xs, top_ys]) - offset
        bottom_pos = np.column_stack([bottom_xs, bottom_ys]) - offset

        pos = np.concatenate([top_pos, bottom_pos])
        pos = rescale_layout(pos, scale=scale) + center
        pos = dict(zip(nodes, pos))
        return pos

    msg = "align must be either vertical or horizontal."
    raise ValueError(msg)