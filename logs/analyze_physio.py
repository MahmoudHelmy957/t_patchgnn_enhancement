import glob, os, re, statistics as stats


#   physio_STUD_seeds_233310_3.err
#   physio_TEST_seeds_232999_5.err
ERR_GLOBS = [
    "physio_STUD_seeds_233310_[0-9].err",
    "physio_TEST_seeds_*_[0-9].err",
    "physio_STUD_*_[0-9].err",
    "physio_TEST_*_[0-9].err",
]

files = []
for g in ERR_GLOBS:
    files.extend(glob.glob(g))
files = sorted(set(files))
if not files:
    raise SystemExit("No matching .err files found. Adjust ERR_GLOBS patterns.")

# "Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: 26, 0.00651, 0.00651, 0.08070, 0.03843, 70.12%"
pat = re.compile(
    r"Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE:\s*\d+,\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)"
)

rows = []
for f in files:
    with open(f, "r", errors="ignore") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                # loss = float(m.group(1)) 
                mse, rmse, mae = map(float, m.groups()[1:4])
                # seed is final token before extension: ..._<seed>.err
                seed = int(os.path.splitext(f)[0].split("_")[-1])
                rows.append((seed, mse, rmse, mae, os.path.basename(f)))

if not rows:
    raise SystemExit("No 'Test - Best epoch' lines found in .err files.")

# last occurrence per seed (latest/best)
by_seed = {}
for seed, mse, rmse, mae, fname in rows:
    by_seed[seed] = (mse, rmse, mae, fname)

seeds = sorted(by_seed)
fmt = lambda x: f"{x:.6f}"
mse_list  = [by_seed[s][0] for s in seeds]
rmse_list = [by_seed[s][1] for s in seeds]
mae_list  = [by_seed[s][2] for s in seeds]

def mean(xs): return sum(xs)/len(xs)
def std(xs):
    if len(xs) < 2: return 0.0
    m = mean(xs)
    return (sum((x-m)**2 for x in xs)/len(xs))**0.5  # population std to match many papers

print("Per-seed best Test metrics (from .err):")
for s in seeds:
    mse, rmse, mae, fname = by_seed[s]
    print(f"  seed {s}: MSE={fmt(mse)}  RMSE={fmt(rmse)}  MAE={fmt(mae)}  ({fname})")

print("\nMean ± std (over seeds):")
print(f"  MSE : {fmt(mean(mse_list))} ± {fmt(std(mse_list))}")
print(f"  RMSE: {fmt(mean(rmse_list))} ± {fmt(std(rmse_list))}")
print(f"  MAE : {fmt(mean(mae_list))} ± {fmt(std(mae_list))}")
