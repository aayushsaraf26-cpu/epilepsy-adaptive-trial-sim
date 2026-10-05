import sys
import os
import subprocess
import numpy as np
from scipy.stats import ranksums
import pandas as pd # Added for CSV saving
import multiprocessing # Added for parallel processing
import random # Added for seeding multiprocessing pools


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: AUTO-CLONE CHOCOLATES
# ──────────────────────────────────────────────────────────────────────────────

if not os.path.exists("./CHOCOLATE"):
    print("Cloning CHOCOLATES from GitHub...")
    subprocess.run(
        ["git", "clone", "https://github.com/GoldenholzLab/CHOCOLATE.git"],
        check=True
    )
    print("Done.")
else:
    print("CHOCOLATES repo already present.")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: IMPORT CHOCOLATES SIMULATOR
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, ".//CHOCOLATE")
from realSim import simulator_base

print("CHOCOLATES simulator_base loaded successfully.")


# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

# Patients per arm
N_PER_ARM = 100

# Trial timing
BASELINE_DAYS  = 8  * 7   # 56 days (8 weeks)
TREATMENT_DAYS = 12 * 7   # 84 days (12 weeks)

# Longer history so enrollment can happen during a "hot" seizure period,
# producing realistic regression-to-the-mean in the placebo arm
TOTAL_HISTORY_DAYS = 40 * 7  # 280 days (40 weeks)

# Eligibility: >= 4 seizures/month during baseline (standard RCT criterion)
MIN_SEIZURES_PER_MONTH = 4.0

# Dropout: 15% attrition, not allowed in the first 2 weeks
DROPOUT_RATE    = 0.15
MIN_DROPOUT_DAY = 14

# Monte Carlo
N_SIMULATIONS     = 100000
PROGRESS_INTERVAL = 500

# Significance thresholds
TRADITIONAL_ALPHA = 0.05

# O'Brien-Fleming adaptive design boundaries (3 looks: 50%, 75%, 100%)
INTERIM_LOOK_FRACTIONS = [0.50, 0.75, 1.00]
OBF_ALPHA_BOUNDARIES = [0.005, 0.02, 0.040] # Adjusted to be more conservative for better Type I error control

# Reproducibility
RANDOM_SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ──────────────────────────────────────────────────────────────────────────────

def sample_drug_efficacy():
    """
    Draw one patient's drug efficacy from Beta(3, 7).

    Mean = 0.30, matching the CHOCOLATES paper's 30% drug effect.
    The Beta distribution produces realistic heterogeneity:
        - some non-responders (efficacy near 0)
        - most patients with modest response (~20-40%)
        - occasional strong responders (efficacy near 60%+)
    """
    return np.random.beta(3, 7)


def sample_placebo_efficacy():
    """
    Draw one patient's apparent placebo improvement from Beta(1.3, 18).

    Mean ~7.7%. This represents apparent improvement from:
        - regression to the mean (main driver)
        - natural seizure rate fluctuation
        - diary/reporting effects

    IMPORTANT: This is used only in the POWER calculation (drug vs placebo).
    The TYPE I ERROR calculation uses efficacy=0 for both arms so the null
    hypothesis is truly satisfied.
    """
    return np.random.beta(1.3, 18)


# ──────────────────────────────────────────────────────────────────────────────
# BASELINE STABILITY FILTER
# ──────────────────────────────────────────────────────────────────────────────

def baseline_is_stable(baseline_diary):
    """
    Exclude patients with highly erratic baselines, matching real RCT practice.

    A patient is considered stable if:
        1) At least 4 of 8 baseline weeks have >= 1 seizure
        2) No single week accounts for > 50% of total baseline seizures

    This avoids enrolling patients whose baseline is dominated by one
    unusual cluster, which would create artificial regression-to-the-mean.
    """
    weekly = baseline_diary.reshape(-1, 7).sum(axis=1)
    total  = weekly.sum()

    if total == 0:
        return False
    if np.sum(weekly > 0) < 4:
        return False
    if weekly.max() > 0.50 * total:
        return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# PATIENT GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate_patient_diary(apparent_efficacy=0.0):
    """
    Generate one trial-like patient using the CHOCOLATES simulator.

    A long seizure history (40 weeks) is generated. A random 8-week
    baseline + 12-week treatment window is chosen from within it.
    This naturally produces regression-to-the-mean: patients enrolled
    during a high-seizure period tend to have lower rates afterward.

    Drug/placebo effect is applied only during the treatment window.
    Dropout randomly truncates the treatment diary for 15% of patients.

    Parameters
    ----------
    apparent_efficacy : float
        Probability each treatment-period seizure is removed.
        0.0 = no effect (used for true null / Type I error measurement).

    Returns
    -------
    baseline_diary  : np.array of int — daily seizure counts
    treatment_diary : np.array of int — daily seizure counts (possibly truncated)
    eligible        : bool — whether patient passed eligibility criteria
    """
    # Generate full continuous history via CHOCOLATES
    full_diary = np.array(simulator_base(1, TOTAL_HISTORY_DAYS), dtype=int)

    # Randomly select an 8-week baseline + 12-week treatment window
    max_start = TOTAL_HISTORY_DAYS - (BASELINE_DAYS + TREATMENT_DAYS)
    start = np.random.randint(0, max_start + 1)

    baseline_diary  = full_diary[start : start + BASELINE_DAYS]
    treatment_diary = full_diary[start + BASELINE_DAYS : start + BASELINE_DAYS + TREATMENT_DAYS].copy()

    # Check eligibility
    baseline_monthly_rate = np.sum(baseline_diary) / BASELINE_DAYS * 30
    eligible = (
        baseline_monthly_rate >= MIN_SEIZURES_PER_MONTH
        and baseline_is_stable(baseline_diary)
    )

    if not eligible:
        return baseline_diary, treatment_diary, False

    # Apply apparent treatment effect to treatment window only
    if apparent_efficacy > 0:
        blocked = np.random.binomial(treatment_diary, apparent_efficacy)
        treatment_diary = treatment_diary - blocked

    # Apply dropout — truncates the treatment observation period
    if np.random.rand() < DROPOUT_RATE:
        dropout_day = np.random.randint(MIN_DROPOUT_DAY, len(treatment_diary) + 1)
        treatment_diary = treatment_diary[:dropout_day]

    return baseline_diary, treatment_diary, True


# ──────────────────────────────────────────────────────────────────────────────
# OUTCOME METRICS (FDA/EMA standard endpoints)
# ──────────────────────────────────────────────────────────────────────────────

def compute_percent_change(baseline_diary, treatment_diary):
    """
    Percentage change in daily seizure rate from baseline to treatment.
    Handles dropout-truncated treatment diaries by using actual days observed.
    Negative = improvement (fewer seizures).
    """
    baseline_rate = np.sum(baseline_diary) / len(baseline_diary)
    if len(treatment_diary) == 0 or baseline_rate == 0:
        return 0.0
    treatment_rate = np.sum(treatment_diary) / len(treatment_diary)
    return (treatment_rate - baseline_rate) / baseline_rate * 100


def compute_rr50(percent_changes):
    """50% Responder Rate: proportion of patients with >= 50% reduction."""
    return np.mean([pc <= -50 for pc in percent_changes])


def compute_mpc(percent_changes):
    """Median Percentage Change."""
    return np.median(percent_changes)


# ──────────────────────────────────────────────────────────────────────────────
# ARM SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

def simulate_one_arm(n_patients, arm_type):
    """
    Simulate one trial arm, enrolling until n_patients eligible patients
    are accumulated.

    arm_type options:
        "drug"    — efficacy drawn from Beta(3,7), mean=0.30
        "placebo" — efficacy drawn from Beta(1.3,18), mean=0.077
        "null"    — efficacy fixed at 0.0 (used for Type I error measurement)

    Parameters
    ----------
    n_patients : int
    arm_type   : str

    Returns
    -------
    percent_changes : list of float
    """
    percent_changes = []

    while len(percent_changes) < n_patients:

        if arm_type == "drug":
            efficacy = sample_drug_efficacy()
        elif arm_type == "placebo":
            efficacy = sample_placebo_efficacy()
        elif arm_type == "null":
            # TYPE I ERROR FIX: both arms get zero efficacy so the null
            # hypothesis is truly satisfied — any "significant" result
            # is a genuine false positive, not a placebo imbalance artifact
            efficacy = 0.0
        else:
            raise ValueError("arm_type must be 'drug', 'placebo', or 'null'")

        baseline, treatment, eligible = generate_patient_diary(
            apparent_efficacy=efficacy
        )

        if eligible:
            percent_changes.append(compute_percent_change(baseline, treatment))

    return percent_changes


# ──────────────────────────────────────────────────────────────────────────────
# TRIAL SIMULATION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def simulate_traditional_trial(drug_arm="drug", placebo_arm="placebo"):
    """
    Simulate one traditional fixed-sample epilepsy RCT.

    arm type parameters allow switching between power (drug vs placebo)
    and Type I error (null vs null) measurement modes.

    Returns
    -------
    success      : bool
    drug_rr50    : float
    placebo_rr50 : float
    drug_mpc     : float
    placebo_mpc  : float
    n_used       : int
    """
    drug_pct    = simulate_one_arm(N_PER_ARM, drug_arm)
    placebo_pct = simulate_one_arm(N_PER_ARM, placebo_arm)
    _, p_value  = ranksums(drug_pct, placebo_pct)

    return (
        p_value < TRADITIONAL_ALPHA,
        compute_rr50(drug_pct),
        compute_rr50(placebo_pct),
        compute_mpc(drug_pct),
        compute_mpc(placebo_pct),
        N_PER_ARM
    )


def simulate_adaptive_trial(drug_arm="drug", placebo_arm="placebo"):
    """
    Simulate one adaptive group sequential epilepsy RCT using
    O'Brien-Fleming boundaries at 50%, 75%, and 100% of enrolment.

    arm type parameters allow switching between power and Type I error modes.

    Returns
    -------
    success      : bool
    drug_rr50    : float
    placebo_rr50 : float
    drug_mpc     : float
    placebo_mpc  : float
    n_used       : int
    stop_look    : int — which look triggered stopping (1, 2, or 3)
    """
    drug_pct_all    = []
    placebo_pct_all = []

    for look_idx, (look_fraction, alpha_boundary) in enumerate(
        zip(INTERIM_LOOK_FRACTIONS, OBF_ALPHA_BOUNDARIES), start=1
    ):
        target_n = int(N_PER_ARM * look_fraction)
        new_n    = target_n - len(drug_pct_all)

        drug_pct_all    += simulate_one_arm(new_n, drug_arm)
        placebo_pct_all += simulate_one_arm(new_n, placebo_arm)

        n_used     = len(drug_pct_all)
        _, p_value = ranksums(drug_pct_all, placebo_pct_all)

        if p_value < alpha_boundary:
            return (
                True,
                compute_rr50(drug_pct_all),
                compute_rr50(placebo_pct_all),
                compute_mpc(drug_pct_all),
                compute_mpc(placebo_pct_all),
                n_used,
                look_idx
            )

    return (
        False,
        compute_rr50(drug_pct_all),
        compute_rr50(placebo_pct_all),
        compute_mpc(drug_pct_all),
        compute_mpc(placebo_pct_all),
        N_PER_ARM,
        3
    )


# ──────────────────────────────────────────────────────────────────────────────
# RESULTS
# ──────────────────────────────────────────────────────────────────────────────

def print_results(records):
    """Print formatted results table with calibration check."""

    power_trad   = np.mean(records["trad_success"])
    power_adapt  = np.mean(records["adapt_success"])
    t1_trad      = np.mean(records["trad_type1"])
    t1_adapt     = np.mean(records["adapt_type1"])
    drug_rr50    = np.mean(records["trad_drug_rr50"])
    placebo_rr50 = np.mean(records["trad_placebo_rr50"])
    drug_mpc     = np.mean(records["trad_drug_mpc"])
    placebo_mpc  = np.mean(records["trad_placebo_mpc"])
    avg_n_trad   = np.mean(records["trad_n"])
    avg_n_adapt  = np.mean(records["adapt_n"])
    stop_1       = np.mean(np.array(records["adapt_stop_look"]) == 1)
    stop_2       = np.mean(np.array(records["adapt_stop_look"]) == 2)
    stop_3       = np.mean(np.array(records["adapt_stop_look"]) == 3)

    power_gain   = (power_adapt - power_trad) * 100
    n_saving_pct = (avg_n_trad - avg_n_adapt) / avg_n_trad * 100
    n_saving_abs = avg_n_trad - avg_n_adapt

    col = 42
    num = 10
    w   = col + num * 2 + 4
    div = "=" * w
    sep = "-" * w

    print("\n" + div)
    print(f"{('Metric'):<{col}} {('Traditional'):>{num}} {('Adaptive'):>{num}}")
    print(sep)
    print(f"{('Power (drug vs placebo)'):<{col}} {power_trad:>{num}.4f} {power_adapt:>{num}.4f}")
    print(f"{('Type I Error (null vs null)'):<{col}} {t1_trad:>{num}.4f} {t1_adapt:>{num}.4f}")
    print(f"{('Drug RR50'):<{col}} {drug_rr50*100:>{num-1}.1f}% {'-':>{num}} ")
    print(f"{('Placebo RR50'):<{col}} {placebo_rr50*100:>{num-1}.1f}% {'-':>{num}} ")
    print(f"{('Drug MPC'):<{col}} {drug_mpc:>{num-1}.1f}% {'-':>{num}} ")
    print(f"{('Placebo MPC'):<{col}} {placebo_mpc:>{num-1}.1f}% {'-':>{num}} ")
    print(f"{('Avg patients per arm'):<{col}} {avg_n_trad:>{num}.1f} {avg_n_adapt:>{num}.1f}")
    print(sep)
    print(f"{('Adaptive stop at look 1 (50%)'):<{col}} {'-':>{num}} {stop_1*100:>{num-1}.1f}%")
    print(f"{('Adaptive stop at look 2 (75%)'):<{col}} {'-':>{num}} {stop_2*100:>{num-1}.1f}%")
    print(f"{('Adaptive stop at look 3 (100%)'):<{col}} {'-':>{num}} {stop_3*100:>{num-1}.1f}%")
    print(sep)
    print(f"  Power gain:          {power_gain:+.2f} percentage points")
    print(f"  Sample size saving:  {n_saving_pct:.1f}% ({n_saving_abs:.1f} fewer patients per arm)")
    print(div)

    print("\nCalibration check vs CHOCOLATES paper targets:")
    print(f"  Drug RR50:     simulated={drug_rr50*100:.1f}%   target=35-45%")
    print(f"  Placebo RR50:  simulated={placebo_rr50*100:.1f}%   target=15-25%")
    print(f"  Drug MPC:      simulated={drug_mpc:.1f}%  target=-35% to -45%")
    print(f"  Placebo MPC:   simulated={placebo_mpc:.1f}%   target=-10% to -20%")
    print(f"  Type I error:  trad={t1_trad:.4f}  adapt={t1_adapt:.4f}  target=~0.05")


# ──────────────────────────────────────────────────────────────────────────────
# MONTE CARLO HELPER FOR PARALLEL EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

def _perform_one_simulation_round(sim_index):
    """
    Performs a single full simulation round (traditional and adaptive, power and Type I error).
    This function is designed to be called by a multiprocessing pool.
    """
    # Set unique seeds for each simulation iteration for reproducibility across parallel runs
    # This seeds both numpy's random and Python's built-in random, which simulator_base uses.
    np.random.seed(RANDOM_SEED + sim_index)
    random.seed(RANDOM_SEED + sim_index)

    # Power measurement: drug vs placebo (Traditional)
    success_trad_power, d_rr50, p_rr50, d_mpc, p_mpc, n_trad = simulate_traditional_trial(
        drug_arm="drug", placebo_arm="placebo"
    )
    # Type I error: null vs null (Traditional)
    t1_trad, _, _, _, _, _ = simulate_traditional_trial(
        drug_arm="null", placebo_arm="null"
    )

    # Adaptive power
    success_adapt_power, _, _, _, _, n_adapt, stop_look = simulate_adaptive_trial(
        drug_arm="drug", placebo_arm="placebo"
    )
    # Adaptive Type I error
    t1_adapt, _, _, _, _, _, _ = simulate_adaptive_trial(
        drug_arm="null", placebo_arm="null"
    )

    return {
        "trad_success": success_trad_power,
        "trad_type1": t1_trad, # Corrected from t1_adapt
        "trad_drug_rr50": d_rr50,
        "trad_placebo_rr50": p_rr50,
        "trad_drug_mpc": d_mpc,
        "trad_placebo_mpc": p_mpc,
        "trad_n": n_trad,
        "adapt_success": success_adapt_power,
        "adapt_type1": t1_adapt,
        "adapt_n": n_adapt,
        "adapt_stop_look": stop_look,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MONTE CARLO RUNNER (Parallelized)
# ──────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(output_csv_file="simulation_results.csv"):
    """
    Run N_SIMULATIONS trials for both designs in two modes:
        1. Power mode: drug arm vs placebo arm
        2. Type I error mode: null arm vs null arm (efficacy=0 both sides)
    This version uses multiprocessing for faster execution.
    """
    records = {
        "trad_success": [], "trad_type1": [],
        "trad_drug_rr50": [], "trad_placebo_rr50": [],
        "trad_drug_mpc": [], "trad_placebo_mpc": [],
        "trad_n": [],
        "adapt_success": [], "adapt_type1": [],
        "adapt_n": [], "adapt_stop_look": [],
    }
    start_simulation_index = 0

    if os.path.exists(output_csv_file):
        try:
            print(f"Loading existing data from {output_csv_file} to resume simulation...")
            df_existing = pd.read_csv(output_csv_file)
            # Ensure all columns are present before attempting to load
            missing_cols = [col for col in records.keys() if col not in df_existing.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in CSV for resume: {missing_cols}")

            for key in records.keys():
                records[key] = df_existing[key].tolist()
            start_simulation_index = len(df_existing)
            print(f"Resuming from simulation {start_simulation_index}...")
        except Exception as e:
            print(f"Error loading existing data ({e}). Starting new simulation.")
            # Reset records to empty and start from scratch if loading fails
            records = {k: [] for k in records.keys()}
            start_simulation_index = 0

    if start_simulation_index >= N_SIMULATIONS:
        print(f"All {N_SIMULATIONS:,} simulations already completed.")
        print_results(records)
        return

    # Determine which simulations still need to be run
    sim_indices_to_run = list(range(start_simulation_index, N_SIMULATIONS))

    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel simulation.")

    # Use a Pool to parallelize the simulation runs
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Use imap_unordered to collect results as they finish, maintaining responsiveness
        # The results list will contain dictionaries, one for each simulation round
        completed_sim_count = start_simulation_index
        for result_dict in pool.imap_unordered(_perform_one_simulation_round, sim_indices_to_run):
            for key, value in result_dict.items():
                records[key].append(value)

            completed_sim_count += 1
            # Save and report progress periodically or when all simulations are done
            if completed_sim_count % PROGRESS_INTERVAL == 0 or completed_sim_count == N_SIMULATIONS:
                print(f"  Progress: {completed_sim_count:,} / {N_SIMULATIONS:,}")
                try:
                    df = pd.DataFrame(records)
                    print(f"  Attempting to write simulation results to {output_csv_file} at simulation {completed_sim_count}...")
                    df.to_csv(output_csv_file, index=False)
                    print(f"  Successfully wrote {len(df)} simulations to {output_csv_file}.")
                except Exception as e:
                    print(f"  Error writing progress: {e}")

    print("\nAll simulations completed.")
    print_results(records)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
# Guarded so that multiprocessing worker processes (which re-import this module
# on macOS/Windows) don't each launch their own Monte Carlo run.

if __name__ == "__main__":
    print("\nClinical Trial Simulation - CHOCOLATES Patient Model (v4)")
    print(f"  Patients per arm         : {N_PER_ARM}")
    print(f"  Baseline period          : {BASELINE_DAYS // 7} weeks")
    print(f"  Treatment period         : {TREATMENT_DAYS // 7} weeks")
    print(f"  Total simulated history  : {TOTAL_HISTORY_DAYS // 7} weeks")
    print(f"  Dropout rate             : {DROPOUT_RATE*100:.0f}%")
    print(f"  Simulations              : {N_SIMULATIONS:,}")
    print(f"  Random seed              : {RANDOM_SEED}\n")

    np.random.seed(RANDOM_SEED)
    run_monte_carlo()
