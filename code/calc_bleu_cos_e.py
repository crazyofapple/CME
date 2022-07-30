from sacrebleu import corpus_bleu
import json
import nltk
import pandas as pd
def computeBLEU(outputs, targets):
    # see https://github.com/mjpost/sacreBLEU
    # targets = [[t[i] for t in targets] for i in range(len(targets[0]))]
    return corpus_bleu(outputs, targets).score

def g_bleu(targets, predictions):
  """Computes BLEU score.
  Args:
    targets: list of strings or list of list of strings if multiple references
      are present.
    predictions: list of strings
  Returns:
    bleu_score across all targets and predictions
  """
  if isinstance(targets[0], list):
    targets = [[x for x in target] for target in targets]
  else:
    # Need to wrap targets in another list for corpus_bleu.
    targets = [targets]

  bleu_score = corpus_bleu(predictions, targets,
                                     smooth_method="exp",
                                     smooth_value=0.0,
                                     force=False,
                                     lowercase=True,
                                     tokenize="intl",
                                     use_effective_order=False)
  return bleu_score.score

ref = "this is a string"
x = g_bleu([[ref, ref, ref], [ref, ref, ref]], [ref, ref, ref])
total = 0
bleu = 0
data = pd.read_csv("$Anonymous_path$/nile/dataset_snli/all/test_data.csv")
golden_es = []
e1s = []
e2s = []
for i in range(len(data)):
    e1 = data.iloc[i]['Explanation_1']
    e2 = data.iloc[i]['Explanation_2']
    
    # e3 = data.iloc[i]['Explanation_3']
    golden_es.append([e1,e2])
    e1s.append(e1)
    e2s.append(e2)

# $Anonymous_path$/nile/dataset_snli/all/predict.json
tpreds = []
tgoldens = []
# $Anonymous_path$/Chinese-GPT/LAS-NL-Explanations/sim_experiments/data/v1.1/predict.json
with open("$Anonymous_path$/Chinese-GPT/LAS-NL-Explanations/sim_experiments/data/v1.1/predict.json", "r") as f:
# with open("$Anonymous_path$/Chinese-GPT/LAS-NL-Explanations/sim_experiments/data/e-SNLI-data/predict.json", "r") as f:
    data = json.load(f)
    for i, item in enumerate(data):
        total += 1
        pred_title = item['predict_explanation']
        # print(pred_title)
        title = item['golden_explanation']
        tpreds.append(pred_title)
        tgoldens.append(title)
        # bleu += computeBLEU(outputs=pred_title, targets=golden_es[i])
    assert len(tpreds) == len(tgoldens)
    # print(tpreds)
    print("generate belu:", g_bleu([tgoldens], tpreds))