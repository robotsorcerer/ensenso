### do not use decaying learning rates seems


### Going for a high number of maxIter upwards of 1000 generates high classification accuracy

On Saturday, Apr 22, I got 97% accuracy with no adaptive learning rate


|                    |
|  Training Sample # |  Testing Sample #  | learning rate  | epochs  | Accuracy on Testing Set (%)
|       100          |         55         |      1e-3      |    100  |       77 %
|       100          |         55         |      1e-2      |    200  |       54 %
==================================================================================================
|       169          |         59         |      1e-3      |    200  |       84 %
|       169          |         59         |      1e-3      |    1000 |       97 %
|       169          |         59         |      1e-3      |    1200 |       77 %
