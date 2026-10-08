# spx-vix-neural-sde

Joint calibration of SPX options, VIX options and VIX futures with a neural stochastic differential equation, from my MSFE practicum at the University of Illinois, sponsored by J.P. Morgan Chase.

The VIX is defined off SPX options, so the two markets have to agree, but a model calibrated to the SPX smile usually misprices the VIX smile and the VIX futures curve. The approach, from Guyon and Mustapha (2023), writes the log SPX price X and the VIX-squared factor Y as two coupled diffusions and lets a small neural network learn the volatility of X, the drift and volatility of Y, and their correlation as functions of (t, X, Y). Each epoch simulates the paths, prices SPX and VIX options by Monte Carlo, computes VIX from a regression of realised variance on the state, and backpropagates the implied volatility error through all of it.

## Whose work is what

Five of us worked on the project, with mentors from J.P. Morgan. Two teammates did the compute side: moving the pricing to CuPy, a CUDA root-finder, and multilevel and quasi Monte Carlo. That work cut simulation time by 90% and is not in this repository. My part was how a fit is judged and how the model trains, and this code is the calibration with those three pieces added.

**Coverage and deviation rates** (`src/metrics.py`). Training loss fell from 4.0 to 0.07 over our runs and told us nothing about whether a trader could use the result. Coverage rate is the share of strikes where the model's implied vol lands inside the market bid-ask. Deviation rate is the mean distance from the mid, measured in bid-ask widths. Coverage is what a desk cares about, and deviation is smooth enough to watch while training runs. Both are logged to `metrics.csv` at every checkpoint.

**Dynamic loss weights** (`src/weights.py`). The published method weights the SPX smile, VIX future and VIX smile losses with fixed constants, so once one part dominates, the others stop improving. Each weight is now its base value times exp(share of total loss / sigma²), recomputed every epoch from detached losses, so the weight moves toward whichever part is fitting worst.

**Warm starts** (`checkpoint.warm_start`, `train.py --warm-start`). A new calibration starts from the network of an earlier one, either one with fewer maturities or one from an earlier date, rather than from random weights.

## Results from the project

From our final report, on market data from WRDS, Bloomberg and OptionsDX, which cannot be redistributed:

- The final model held coverage above 95% and deviation below 0.1 on a calm quote date, 1 October 2021, and a volatile one, 2 April 2025.
- Over identical 10,000-epoch runs, dynamic weights reached higher coverage than the published fixed weights from the start of training and stayed ahead.
- A warm start reached the same loss on a new SPX maturity in 600 epochs instead of 30,000, and about 400 epochs when rolling the window forward. It did worse than a cold start on the VIX options and futures, which we left as an open problem.

## Running it without the data

`tests/make_synthetic_market.py` writes a synthetic SPX and VIX option market in the same CSV layout as the real data, so the whole pipeline runs end to end without proprietary quotes. The two surfaces are not jointly consistent, so this checks that the pipeline works rather than how well it fits.

On that market, 600 epochs with dynamic weights and 20,000 paths took 16 minutes on a laptop RTX 4060. The loss fell from 992 to 0.54. Deviation fell from 281 bid-ask widths to 9.5 on the SPX side and from 47.8 to 3.9 on the VIX side, and the weights moved the way they should: the VIX smile weight rose from 2.0 to 4.1 while that part carried most of the loss, then eased back to 2.8 once it fitted. A warm start from the final checkpoint began at a loss of 0.54, against 992 from a cold start.

```
pip install -r requirements.txt
python tests/test_components.py                       # metrics, weights, warm start
python tests/make_synthetic_market.py data/synthetic
cd src
NSDE_DATA_DIR=../data/synthetic python train.py --epochs 600 --batch-size 20000 --weighting dynamic --name demo
NSDE_DATA_DIR=../data/synthetic python train.py --epochs 50 --batch-size 20000 --warm-start checkpoints/demo/<last>.ckpt --name demo_warm
```

With real data, put the option chains, spot series and yield curve in `data/misc/` in the format `src/data.py` reads and drop `NSDE_DATA_DIR`. A GPU is strongly recommended. The default batch is 150,000 paths.

## Reference

Guyon, J. and Mustapha, S. (2023). Neural joint S&P 500/VIX smile calibration. *Risk*. [SSRN 4309576](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4309576).
