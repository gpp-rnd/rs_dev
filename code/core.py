from scipy import stats
import pandas as pd
import plotnine as gg
from sklearn.model_selection import StratifiedGroupKFold


def get_predictive_performance(prediction_list, activity_col):
    """From a list of dataframes of predictions for each fold for each datasetset for each guide,
    get pearson correlation of each fold/dataset

    :param prediction_list: list of DataFrame
    :param activity_col: str, column which we're trying to predict
    :return: DataFrame
    """
    test_prediction_df = (pd.concat(prediction_list)
                          .rename({'dataset': 'testing_set'}, axis=1))
    predictive_performance = (test_prediction_df.groupby(['model_name', 'testing_set', 'fold'])
                              .apply(lambda df: stats.pearsonr(df[activity_col], df['prediction'])[0])
                              .reset_index(name='pearson_r'))
    test_set_count = (test_prediction_df[['sgRNA Context Sequence', 'testing_set', 'fold']]
                      .drop_duplicates()
                      .groupby(['testing_set', 'fold'])
                      .agg(n_guides=('sgRNA Context Sequence', 'count'))
                      .reset_index())
    predictive_performance = (predictive_performance.merge(test_set_count, how='inner',
                                                           on=['testing_set', 'fold']))
    predictive_performance['fold'] = predictive_performance['fold'].astype('category')
    predictive_performance['fold_name'] = ('fold ' + predictive_performance['fold'].astype(str) +
                                           ' (n=' +
                                           predictive_performance['n_guides'].astype(int).astype(str) + ')')
    agg_performance = (predictive_performance.groupby('model_name')
                       .agg(mean_pearson=('pearson_r', 'mean'),
                            std_pearson=('pearson_r', 'std'),
                            median_pearson=('pearson_r', 'median'))
                       .reset_index()
                       .sort_values('mean_pearson', ascending=False))
    predictive_performance['model_name'] = pd.Categorical(predictive_performance['model_name'],
                                                            categories=agg_performance['model_name'])
    if predictive_performance['testing_set'].isin(predictive_performance['model_name']).all():
        predictive_performance['testing_set'] = pd.Categorical(predictive_performance['testing_set'],
                                                               categories=agg_performance['model_name'])
    return predictive_performance, agg_performance


def plot_pearson_heatmap(predictive_performance):
    """Plot heatmap of pearson correlations

    :param predictive_performance: DataFrame from `get_predictive_performance`
    :return: plotnine figure
    """
    g = (gg.ggplot(predictive_performance) +
         gg.aes(x='testing_set', y='model_name', fill='pearson_r') +
         gg.geom_tile(color='black') +
         gg.scale_fill_cmap('RdBu_r', limits=(-1, 1)) +
         gg.theme_classic() +
         gg.theme(axis_text_x=gg.element_text(angle=90, hjust=0.5, vjust=1)) +
         gg.facet_wrap('fold'))
    return g


def plot_pearson_lollipop(predictive_performance):
    g = (gg.ggplot(predictive_performance) +
         gg.aes(y='pearson_r', ymin=0, ymax='pearson_r', x='fold_name', xend='fold_name', color='model_name') +
         gg.scale_color_brewer(type='qual', palette='Set2') +
         gg.geom_point(position=gg.position_dodge(width=0.7), size=4, shape='.') +
         gg.geom_linerange(position=gg.position_dodge(width=0.7)) +
         gg.coord_flip() +
         gg.theme_classic() +
         gg.theme(subplots_adjust={'wspace': 0.6, 'hspace': 0.3}) +
         gg.facet_wrap('testing_set', scales='free') +
         gg.guides(color=gg.guide_legend(reverse=True)))
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
