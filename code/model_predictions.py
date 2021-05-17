import pandas as pd


class ModelPrediction():
    def __init__(self, prediction_file, prediction_col, name, train_sgrna_file, train_20mer_col=None,
                 train_23mer_col=None, train_30mer_col=None, classtype='model', target_col=None):
        self.prediction_file = prediction_file
        self.prediction_col = prediction_col
        self.name = name
        self.train_sgrna_file = train_sgrna_file
        self.train_20mer_col = train_20mer_col
        self.train_23mer_col = train_23mer_col
        self.train_30mer_col = train_30mer_col
        self.classtype = classtype
        self.target_col = target_col
        self.prediction_df = None
        self.sgrnas = None

    def load_model_predictions(self):
        self.prediction_df = pd.read_csv(self.prediction_file)

    def get_sgrnas(self):
        train_df = pd.read_csv(self.train_sgrna_file)
        if self.train_20mer_col is not None:
            self.sgrnas = set(train_df[self.train_20mer_col].to_list())
        if self.train_23mer_col is not None:
            self.sgrnas = set(train_df[self.train_23mer_col]
                              .str[:20]
                              .to_list())
        if self.train_30mer_col is not None:
            self.sgrnas = set(train_df[self.train_30mer_col]
                              .str[4:-6]
                              .to_list())


rule_set2_predictions = ModelPrediction('../data/external/Rule_Set_2_rs_dev_all_sgrnas.csv', 'Rule Set 2', 'Rule Set 2',
                                        'https://raw.githubusercontent.com/MicrosoftResearch/Azimuth/master/azimuth/data/FC_plus_RES_withPredictions.csv',
                                        train_30mer_col='30mer')

deepspcas9_predictions = ModelPrediction('../data/external/DeepSpCas9_rs_dev_all_sgrnas.csv', 'DeepSpCas9', 'DeepSpCas9',
                                         '../data/raw/Kim2019_S1_Train.csv',
                                         train_30mer_col='Target context sequence (4+20+3+3)')

deepcrispr_predictions = ModelPrediction('../data/external/DeepCRISPR_rs_dev_all_sgrnas.csv', 'DeepCRISPR', 'DeepCRISPR',
                                         '../data/external/Chuai2018_tableS5_sgrnas.csv', train_23mer_col='sgRNA')

model_prediction_list = [rule_set2_predictions, deepspcas9_predictions, deepcrispr_predictions]

vbc_predictions = ModelPrediction('../data/external/vbc_scores_rs_dev.csv', 'VBC score', 'VBC score',
                                  '../data/external/munozS4_azimuth_sgrnas.csv', train_20mer_col='sgRNA Sequence',
                                  target_col='gene')
