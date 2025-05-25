import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

# from evaluator.build import EVALUATOR_REGISTRY
from ngram_metrics.bleu.bleu import Bleu
from ngram_metrics.cider.cider import Cider
from ngram_metrics.meteor.meteor import Meteor
from ngram_metrics.rouge.rouge import Rouge
import json
import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# @EVALUATOR_REGISTRY.register()
class CaptionEvaluator():
    def __init__(self, task_name = ''):
        self.task_name = task_name

        self.cider_scorer = Cider()
        self.bleu_scorer = Bleu()
        self.meteor_scorer = Meteor()
        self.rouge_scorer = Rouge()

        self.best_result = -np.inf
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.reset()

    def reset(self):
        self.eval_dict = {
            'target_metric': [], 'sentence_sim': [],
            'cider': 0, 'bleu': 0, 'meteor': 0, 'rouge': 0,
        }
        self.total_count = 0
        self.save_results = []
        self.init_corpus()

    def init_corpus(self):
        self.gt_sentence_mp = []
        self.pred_sentence_mp = []

    def batch_metrics(self, data_dict):
        metrics = {}
        output_gt = data_dict['output_gt']
        output_pred = data_dict['output_txt']
        batch_size = len(output_gt)

        # consider IoU-based caption metrics
        if 'iou_flag' in data_dict:
            iou_flags = data_dict['iou_flag']
        else:
            iou_flags = [True] * batch_size

        if self.task_name.lower() == 'scan2cap':
            for i in range(batch_size):
                corpus_key = data_dict['corpus_key'][i]
                if iou_flags[i]:
                    self.pred_sentence_mp[corpus_key] = [('sos ' + output_pred[i] + ' eos').replace('. ', ' . ')]
                else:
                    output_pred[i] = ""
                    self.pred_sentence_mp[corpus_key] = ["sos eos"]
        else:
            for i in range(batch_size):
                if iou_flags[i]:
                    self.pred_sentence_mp.append([output_pred[i]])
                else:
                    output_pred[i] = ""
                    self.pred_sentence_mp.append([""])
                self.gt_sentence_mp.append([output_gt[i]])

        # compute sentence similarity
        embed_pred = self.sentence_model.encode(output_pred, convert_to_tensor=True)
        embed_gt = self.sentence_model.encode(output_gt, convert_to_tensor=True)
        sims = pytorch_cos_sim(embed_pred, embed_gt).diag()

        metrics['total_count'] = batch_size
        metrics['sentence_sim'] = sims.mean().item()
        metrics['target_metric'] = metrics['sentence_sim']
        return metrics

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        batch_size = metrics['total_count']
        self.total_count += batch_size

        for i in range(batch_size):
            save_dict = {
                # vision
                # 'source': data_dict['source'][i],
                # 'scene_id': data_dict['scene_id'][i],
                # 'anchor': data_dict['anchor_locs'][i].tolist(),
                # language
                # 'instruction': data_dict['prompt_after_obj'][i],
                'response_gt': data_dict['output_gt'][i],
                'response_pred': data_dict['output_txt'][i],
            }
            if 'iou_flag' in data_dict:
                save_dict['iou_flag'] = data_dict['iou_flag'][i].item()
            self.save_results.append(save_dict)

        for key in self.eval_dict.keys():
            if key not in ['cider', 'bleu', 'meteor', 'rouge']:
                self.eval_dict[key].append(metrics[key] * batch_size)

    def record(self):
        # ngram metrics
        if self.task_name.lower() == 'scan2cap':
            # align gt_sentence_mp to pred_sentence_mp for partial evaluation
            self.gt_sentence_mp = {k: self.gt_sentence_mp[k] for k in self.pred_sentence_mp.keys()}
        else:
            self.gt_sentence_mp = {k: v for k, v in enumerate(self.gt_sentence_mp)}
            self.pred_sentence_mp = {k: v for k, v in enumerate(self.pred_sentence_mp)}

        self.eval_dict['cider'] = self.cider_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]
        self.eval_dict['bleu'] = self.bleu_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0][-1]
        self.eval_dict['meteor'] = self.meteor_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]
        self.eval_dict['rouge'] = self.rouge_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]

        # others
        for k, v in self.eval_dict.items():
            if k not in ['cider', 'bleu', 'meteor', 'rouge']:
                self.eval_dict[k] = sum(v) / self.total_count

        return self.eval_dict

if __name__ == '__main__':
    task = 'rscan_changere'
    epoch = 5
    folder_list = {
                    "SCReasoner": "results/SCReasoner/rscan_changecap/results.json"
                    }
    for folder_name, folder_path in folder_list.items():
        print(f"Evaluating {folder_name}...")
        # Load the JSON file
        js = json.load(open(folder_path, 'r'))
        output_gt = []
        output_txt = []
        for data in tqdm.tqdm(js):
            # if "o'clock" in data['response_pred']:
            #     continue
            output_gt.append(data['response_gt'])
            output_txt.append(data['response_pred'])

        evaluator = CaptionEvaluator()
        data_dict = {
            'output_gt': output_gt,
            'output_txt': output_txt,
        }
        evaluator.update(data_dict)
        result = evaluator.record()
        print(folder_name, result)
