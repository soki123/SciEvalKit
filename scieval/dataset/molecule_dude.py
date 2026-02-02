import warnings
import pandas as pd
import re
from abc import abstractmethod
from sklearn.metrics import matthews_corrcoef, accuracy_score, r2_score, roc_auc_score, precision_score, recall_score, mean_absolute_error
from ..smp import *
from .text_base import TextBaseDataset

def compute_metrics(scores, labels, alpha=80.5):
    if len(set(labels)) < 2:
        return 0.0, 0.0, 0.0, 0.0

    scores_np = np.array(scores)
    labels_np = np.array(labels)
    sort_idx = np.argsort(-scores_np)
    sorted_scores = scores_np[sort_idx]
    sorted_labels = labels_np[sort_idx]
    score_data = np.column_stack((sorted_scores, sorted_labels.astype(bool)))

    try:
        roc_auc = Scoring.CalcAUC(score_data, col=1)
        bedroc = Scoring.CalcBEDROC(score_data, col=1, alpha=alpha)
        ef_1 = Scoring.CalcEnrichment(score_data, col=1, fractions=[0.01])[0]
        ef_5 = Scoring.CalcEnrichment(score_data, col=1, fractions=[0.05])[0]
    except Exception:
        roc_auc, bedroc, ef_1, ef_5 = 0.0, 0.0, 0.0, 0.0
    return roc_auc, bedroc, ef_1, ef_5


def extract_score(text):
    if not text or not isinstance(text, str):
        return 0.0
    match = re.search(r"(\d+(\.\d+)?)", text)
    if match:
        try:
            val = float(match.group(1))
            if 1.0 < val <= 100:
                return val / 100.0
            print(val)
            return val
        except (ValueError, TypeError):
            return 0.0
    return 0.0



class molecule_DUDE(TextBaseDataset):
    TYPE = 'TEXT'
    DATASET_URL = {
        'dude': '',
    }
    DATASET_MD5 = {
        'dude': '',
    }#MD5码



    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        result_values = []
        label_values = []
        data = load(eval_file)
        data= data[~pd.isna(data["prediction"])]
        assert 'answer' in data and 'prediction' in data 
        #获取dataset_name名

        if data.empty :
            return {'ROC_AUC': 0.0, 'BEDROC': 0.0, 'EF1%': 0.0, 'EF5%': 0.0}
        grouped = defaultdict(list)
        for index,entry in data.iterrows():
            score=extract_score(entry["prediction"])
            ref_data = json.loads(entry["answer"])

            true_label = ref_data.get('label', 0)
            target = ref_data.get('target', 'default')
            q = ref_data.get('q_atoms', 1)
            c = ref_data.get('c_atoms', 1)
            factor = 2 * c / (q + c) if (q + c) > 0 else 1.0
            adj_score = score * factor
            grouped[target].append((adj_score, true_label))

        results = []
        for target, pairs in grouped.items():
            if not pairs:
                continue
            sc, lb = zip(*pairs)
            roc_auc, bedroc, ef1, ef5 = compute_metrics(list(sc), list(lb))
            results.append({
                'target': target,
                'ROC_AUC': roc_auc,
                'BEDROC': bedroc,
                'EF1%': ef1,
                'EF5%': ef5,
            })
        if not results:
            return {'ROC_AUC': 0.0, 'BEDROC': 0.0, 'EF1%': 0.0, 'EF5%': 0.0}
        df = pd.DataFrame(results)
        return {
            'ROC_AUC': round(float(df['ROC_AUC'].mean()), 4),
            'BEDROC': round(float(df['BEDROC'].mean()), 4),
            'EF1%': round(float(df['EF1%'].mean()), 4),
            'EF5%': round(float(df['EF5%'].mean()), 4),
        }
