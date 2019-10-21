import torch
from collections import OrderedDict


class Predictor:

	def __init__(self):

		self.idx2label = OrderedDict([(0, 'O'),
                             (1, 'B_EVENT'),
                             (2, 'I_EVENT'),
                             (3, 'B_TIMEX'),
                             (4, 'I_TIMEX')])

	def predict(self, data, model):
		"""
		data[0]: token_idx
		data[1]: pos_idx
		"""
		self.model = model
		self.model.eval()
		scores = self.model(data[0], data[1])

		score, pred = scores.max(dim = 1, keepdim=False)
		pred_list = pred.tolist()
		# pred_list = [self.idx2label[i] for i in pred_list]
		return score, pred_list
