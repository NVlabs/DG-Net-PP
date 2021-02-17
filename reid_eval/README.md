## Evaluation
The results are slightly different from the paper.

- For market2duke
```bash
python test_2label_duke.py --name best-market2duke --which_epoch 231805
```
The result is `Rank@1:0.7931 Rank@5:0.8793 Rank@10:0.8990 mAP:0.6436`.

`--name` model name 

`--which_epoch` select the i-th model

- For duke2market
```bash
python test_2label_market.py --name best-duke2market --which_epoch 172353

The result is `Rank@1:0.8260 Rank@5:0.9136 Rank@10:0.9388 mAP:0.6400`
