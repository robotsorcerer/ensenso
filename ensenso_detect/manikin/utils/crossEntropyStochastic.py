
"""
Implements the cross-entropy loss coupled with the doubly 
stochastic penalty introduced in Xu et. al. (2015)

The Loss Function is thus defined:

L = - \sum_{t=1}^{T} \sum_{i=1}^{C} [y_{t, i} log \hat{y}_{t,i}] + 
      \lambda \sum_{i=1}^{K^2}(1 - \sum_{t=1}^T l_{t,i})^2 + 
      \gamma \sum_i \sum_j \theta_{i,j}^2
"""

class CrossEntropyStochasticLoss(nn.Module):
	"""
	docstring for CrossEntropyStochasticLoss

	See:
		https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/14
		http://pytorch.org/docs/notes/extending.html#extending-torch-autograd
	"""
	def __init__(self, arg):

		super(CrossEntropyStochasticLoss, self).__init__()

		


	def forward(self, x):
	    x = self.features(x)
	    x = x.view(x.size(0), -1)
	    x = self.classifier(x)
	    return x