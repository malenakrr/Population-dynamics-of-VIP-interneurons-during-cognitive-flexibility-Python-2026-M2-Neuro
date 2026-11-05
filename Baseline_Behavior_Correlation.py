import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# =============================================================================
# USER SETTINGS
# =============================================================================
NPZ_FILE = # CaImAn NPZ output
BEHAVIOR_FILE = # Your behavioral data
OUTPUT_DIR = # Choice of the output directory for figures
TRACE_VARIANT = "dFF"                          # Options: dFF, dFF_demixed, dFF_annulus, dFF_regression

# Analysis Windows (in seconds relative to the nosepoke at t=0)
PRE_WINDOW_S = -3.0                            # Time to look before the poke
POST_WINDOW_S = 10.0                           # Time to look after the poke

# =============================================================================
# 1. LOAD DATA FROM NPZ
# =============================================================================
print("Loading data from NPZ archive...")
npz_data = np.load(NPZ_FILE)

# Select requested dF/F trace variant
if TRACE_VARIANT in npz_data.files:
    dff_raw = npz_data[TRACE_VARIANT]          # Shape in NPZ: (n_rois, n_frames)
else:
    print(f"Warning: '{TRACE_VARIANT}' not found in NPZ. Loading default 'dFF'.")
    dff_raw = npz_data['dFF']

t_calcium = npz_data['time_s']                 # Shape: (n_frames,)
dff_matrix = dff_raw.T                         # Transpose to shape: (n_frames, n_rois)
n_frames, n_rois = dff_matrix.shape

# Read exact frame rate from NPZ if available, otherwise calculate from t_calcium
if 'fr' in npz_data.files:
    FRAME_RATE = float(npz_data['fr'])
elif len(t_calcium) > 1:
    FRAME_RATE = round(1.0 / np.median(np.diff(t_calcium)), 2)
else:
    FRAME_RATE = 10.0

print(f"Loaded {n_rois} ROIs across {n_frames} frames (Frame rate: {FRAME_RATE:.2f} Hz).")

# Load Behavior
behav_df = pd.read_excel(BEHAVIOR_FILE, sheet_name='nose_pokes')
behav_df.columns = behav_df.columns.str.strip()

# Filter out events that occurred outside the recording range
if 'hors_plage' in behav_df.columns:
    behav_df = behav_df[~behav_df['hors_plage'].astype(bool)]

event_frames = behav_df['frame_calcium'].values - 1  # 0-indexed for python

# Safely extract 'Correct' column
if 'Correct' in behav_df.columns:
    correct_raw = behav_df['Correct'].values
elif 'correct' in behav_df.columns:
    correct_raw = behav_df['correct'].values
else:
    print("Warning: No 'Correct' column found. Treating all nosepokes as correct.")
    correct_raw = np.ones(len(behav_df), dtype=bool)

if correct_raw.dtype == object:
    correct_trials = np.array([str(x).strip().lower() in ['true', 'vrai', '1', 'oui', 'yes'] for x in correct_raw])
else:
    correct_trials = correct_raw.astype(bool)

pre_frames = int(np.abs(PRE_WINDOW_S) * FRAME_RATE)
post_frames = int(POST_WINDOW_S * FRAME_RATE)
window_frames = pre_frames + post_frames + 1
t_window = np.linspace(PRE_WINDOW_S, POST_WINDOW_S, window_frames)

# =============================================================================
# 2. EXTRACT PERI-EVENT TRACES (PETH)
# =============================================================================
print("Extracting peri-event traces...")
event_tensor = []
valid_events = []

for idx, frame in enumerate(event_frames):
    if frame - pre_frames >= 0 and frame + post_frames < n_frames:
        chunk = dff_matrix[frame - pre_frames : frame + post_frames + 1, :]
        event_tensor.append(chunk)
        valid_events.append(idx)

event_tensor = np.array(event_tensor)  # Shape: (Events, Time, ROIs)
event_tensor = np.swapaxes(event_tensor, 1, 2)  # Reshape to: (Events, ROIs, Time)

valid_correct = correct_trials[valid_events]

# =============================================================================
# 3. CALCULATE MEAN RESPONSES
# =============================================================================
mean_response_per_roi = np.nanmean(event_tensor, axis=0)

peak_times = np.argmax(mean_response_per_roi, axis=1)
sort_idx = np.argsort(peak_times)
sorted_responses = mean_response_per_roi[sort_idx]

correct_tensor = event_tensor[valid_correct == True]
incorrect_tensor = event_tensor[valid_correct == False]

mean_correct = np.nanmean(correct_tensor, axis=(0, 1)) if len(correct_tensor) > 0 else []
mean_incorrect = np.nanmean(incorrect_tensor, axis=(0, 1)) if len(incorrect_tensor) > 0 else []

std_correct = np.nanstd(np.nanmean(correct_tensor, axis=1), axis=0) / np.sqrt(len(correct_tensor)) if len(correct_tensor) > 0 else []
std_incorrect = np.nanstd(np.nanmean(incorrect_tensor, axis=1), axis=0) / np.sqrt(len(incorrect_tensor)) if len(incorrect_tensor) > 0 else []

# =============================================================================
# 4. PLOTTING
# =============================================================================
print("Generating figures...")
sns.set_context("talk")
sns.set_style("ticks")

# --- FIGURE 1: Peri-Event Heatmap (All ROIs) ---
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(sorted_responses, aspect='auto', cmap='magma', 
               extent=[PRE_WINDOW_S, POST_WINDOW_S, n_rois, 0],
               vmin=np.percentile(sorted_responses, 5), vmax=np.percentile(sorted_responses, 98))
ax.axvline(0, color='white', linestyle='--', lw=2)
ax.set_title("Nosepoke-Triggered Activity (Sorted by Response Time)")
ax.set_xlabel("Time from Nosepoke (s)")
ax.set_ylabel("ROI ID (Sorted)")
fig.colorbar(im, ax=ax, label=r"Mean $\Delta F/F$")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Behavior_1_PETH_Heatmap.png"), dpi=300)
plt.close(fig)

# --- FIGURE 2: Event-Triggered Average (Correct vs Incorrect) ---
fig, ax = plt.subplots(figsize=(8, 6))

if len(correct_tensor) > 0:
    ax.plot(t_window, mean_correct, color='tab:green', label=f'Correct (n={len(correct_tensor)})', lw=2)
    ax.fill_between(t_window, mean_correct - std_correct, mean_correct + std_correct, color='tab:green', alpha=0.2)

if len(incorrect_tensor) > 0:
    ax.plot(t_window, mean_incorrect, color='tab:red', label=f'Incorrect (n={len(incorrect_tensor)})', lw=2)
    ax.fill_between(t_window, mean_incorrect - std_incorrect, mean_incorrect + std_incorrect, color='tab:red', alpha=0.2)

ax.axvline(0, color='k', linestyle='--', lw=1.5, label='Nosepoke')
ax.axhline(0, color='gray', linestyle=':', lw=1)
ax.set_title("Population Event-Triggered Average")
ax.set_xlabel("Time relative to Nosepoke (s)")
ax.set_ylabel(r"Population Mean $\Delta F/F$")
ax.legend()
sns.despine()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Behavior_2_Event_Triggered_Avg.png"), dpi=300)
plt.close(fig)

# --- FIGURE 3: Pre vs Post Activity Comparison ---
pre_mask = (t_window >= -2.0) & (t_window < 0)
post_mask = (t_window > 0) & (t_window <= 2.0)

pre_activity = np.nanmean(mean_response_per_roi[:, pre_mask], axis=1)
post_activity = np.nanmean(mean_response_per_roi[:, post_mask], axis=1)

t_stat, p_val = stats.ttest_rel(pre_activity, post_activity)

fig, ax = plt.subplots(figsize=(6, 6))
sns.boxplot(data=[pre_activity, post_activity], palette=['gray', 'tab:blue'], ax=ax, width=0.5)
sns.stripplot(data=[pre_activity, post_activity], color='k', alpha=0.5, jitter=True, ax=ax)
for i in range(n_rois):
    ax.plot([0, 1], [pre_activity[i], post_activity[i]], color='k', alpha=0.15, lw=0.5)

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-Poke\n(-2s to 0s)', 'Post-Poke\n(0s to +2s)'])
ax.set_ylabel(r"Mean $\Delta F/F$")
ax.set_title(f"Activity Modulation by Nosepoke\n(Paired t-test: p = {p_val:.4f})")
sns.despine()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Behavior_3_Pre_vs_Post.png"), dpi=300)
plt.close(fig)

# --- FIGURE 4: ROI Pairwise Pearson Correlation Matrix (Full 600-s Session) ---
corr_df = pd.DataFrame(dff_matrix).corr()
corr_matrix = corr_df.values

off_diag_mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
off_diag_vals = corr_matrix[off_diag_mask]
off_diag_vals = off_diag_vals[~np.isnan(off_diag_vals)]

if len(off_diag_vals) > 0:
    vmin_val = np.percentile(off_diag_vals, 2)
    vmax_val = np.percentile(off_diag_vals, 98)
else:
    vmin_val, vmax_val = -0.2, 0.5

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(
    corr_matrix,
    cmap="vlag",
    center=0,
    vmin=vmin_val,
    vmax=vmax_val,
    annot=False,
    cbar_kws={"label": "Pearson's r"},
    square=True,
    xticklabels=False,
    yticklabels=False,
    ax=ax,
)
ax.set_title(
    f"Network Pearson Correlation Matrix ({n_rois} ROIs)\n(Full 600-s Session Dynamics)"
)
ax.set_xlabel("ROI ID")
ax.set_ylabel("ROI ID")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Behavior_4_Correlation_Matrix.png"), dpi=300)
plt.close(fig)

# --- FIGURE 5: Global Population Activity Across Entire Video Timeline ---
fig, (ax_heat, ax_mean) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

im = ax_heat.imshow(dff_matrix.T, aspect='auto', cmap='viridis', origin='lower',
                    extent=[t_calcium[0], t_calcium[-1], 0, n_rois],
                    vmin=np.percentile(dff_matrix, 2), vmax=np.percentile(dff_matrix, 98))
ax_heat.set_title("Global Population Activity Across Entire Video Timeline", fontsize=13, fontweight='bold')
ax_heat.set_ylabel(f"Neuron Index (1-{n_rois})", fontsize=11)
fig.colorbar(im, ax=ax_heat, label=r"$\Delta F/F$")

mean_pop_trace = np.nanmean(dff_matrix, axis=1)
ax_mean.plot(t_calcium, mean_pop_trace, color='black', lw=1.0)
ax_mean.set_title("Average Population Fluorescence Dynamics", fontsize=12, fontweight='bold')
ax_mean.set_xlabel("Time Across Video Timeline (Seconds)", fontsize=11)
ax_mean.set_ylabel(r"Mean $\Delta F/F$", fontsize=11)
ax_mean.grid(True, alpha=0.3)

poke_duration_s = 2.0  
legend_done = False
for idx, f_idx in enumerate(event_frames):
    if f_idx < len(t_calcium):
        t_start = t_calcium[f_idx]
        t_end = t_start + poke_duration_s
        lbl = "Correct Choice" if (not legend_done and correct_trials[idx]) else None
        if lbl: legend_done = True
        color = 'green' if correct_trials[idx] else 'red'
        ax_heat.axvspan(t_start, t_end, color=color, alpha=0.35, label=lbl)
        ax_mean.axvspan(t_start, t_end, color=color, alpha=0.35)

if legend_done:
    ax_heat.legend(loc="upper right", fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Behavior_5_Global_Activity_Full_Session.png"), dpi=300)
plt.close(fig)

# --- FIGURE 6: Reward Encoding (Correct vs Incorrect Post-Poke Activity) ---
pop_activity_per_event = np.nanmean(event_tensor[:, :, post_mask], axis=(1, 2))

correct_event_activity = pop_activity_per_event[valid_correct == True]
incorrect_event_activity = pop_activity_per_event[valid_correct == False]

fig, ax = plt.subplots(figsize=(6, 6))
if len(correct_event_activity) > 0 and len(incorrect_event_activity) > 0:
    t_stat_reward, p_val_reward = stats.ttest_ind(correct_event_activity, incorrect_event_activity, equal_var=False)
    
    sns.boxplot(data=[correct_event_activity, incorrect_event_activity], palette=['tab:green', 'tab:red'], ax=ax, width=0.5)
    sns.stripplot(data=[correct_event_activity, incorrect_event_activity], color='k', alpha=0.6, jitter=True, ax=ax)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Correct / Reward\n(n={len(correct_event_activity)})', f'Incorrect / No Reward\n(n={len(incorrect_event_activity)})'])
    ax.set_ylabel(r"Population Mean $\Delta F/F$ (Post-Poke)")
    ax.set_title(f"Reward Encoding (Outcome Modulation)\n(Welch's t-test: p = {p_val_reward:.4f})")
else:
    ax.text(0.5, 0.5, "Not enough trials in both conditions to compare", ha='center', va='center')
    ax.set_title("Reward Encoding")

sns.despine()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Behavior_6_Reward_vs_NoReward.png"), dpi=300)
plt.close(fig)

# --- FIGURE 7: Hierarchically Clustered Correlation Matrix (Full Session) ---
print("Generating clustered correlation matrix...")
corr_filled = np.nan_to_num(corr_matrix, nan=0.0)
np.fill_diagonal(corr_filled, 1.0)

dist_matrix = 1.0 - corr_filled
np.fill_diagonal(dist_matrix, 0.0)
dist_matrix = np.clip(dist_matrix, 0, None)

Z = linkage(squareform(dist_matrix, checks=False), method='average')
cluster_order = leaves_list(Z)

clustered_corr_matrix = corr_matrix[np.ix_(cluster_order, cluster_order)]

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(
    clustered_corr_matrix,
    cmap="vlag",
    center=0,
    vmin=vmin_val,
    vmax=vmax_val,
    annot=False,
    cbar_kws={"label": "Pearson's r"},
    square=True,
    xticklabels=False,
    yticklabels=False,
    ax=ax,
)
ax.set_title(
    f"Clustered Network Correlation Matrix ({n_rois} ROIs)\n(Full 600-s Session Dynamics)"
)
ax.set_xlabel("ROI ID (Hierarchically Clustered)")
ax.set_ylabel("ROI ID (Hierarchically Clustered)")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Behavior_7_Correlation_Matrix_Clustered.png"), dpi=300)
plt.close(fig)

# --- FIGURE 8: Full-Session PETH with Nosepoke Markers (Sorted by Clusters) ---
print("Generating full-session clustered PETH...")
fig, ax = plt.subplots(figsize=(16, 8))

clustered_dff_traces = dff_matrix.T[cluster_order, :]

im = ax.imshow(
    clustered_dff_traces,
    aspect='auto',
    cmap='magma', 
    origin='lower',
    extent=[t_calcium[0], t_calcium[-1], 0, n_rois],
    vmin=np.percentile(clustered_dff_traces, 5),
    vmax=np.percentile(clustered_dff_traces, 98)
)

for idx, f_idx in enumerate(event_frames):
    if 0 <= f_idx < len(t_calcium):
        t_poke = t_calcium[f_idx]
        is_correct = correct_trials[idx]
        color = '#00FF00' if is_correct else '#FF0000'
        ax.axvline(t_poke, color=color, linestyle=':', lw=2.5, alpha=0.9)

ax.plot([], [], color='#00FF00', linestyle=':', lw=2.5, label='Correct Nosepoke')
if not all(correct_trials):
    ax.plot([], [], color='#FF0000', linestyle=':', lw=2.5, label='Incorrect Nosepoke')

ax.set_title("Full Session Population Activity (Sorted by Hierarchical Clustering)", fontsize=14, fontweight='bold')
ax.set_xlabel("Time Across Video Timeline (Seconds)", fontsize=12)
ax.set_ylabel("ROI ID (Clustered Structure)", fontsize=12)
ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
fig.colorbar(im, ax=ax, label=r"$\Delta F/F$")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Behavior_8_Full_Session_Clustered_PETH.png"), dpi=300)
plt.close(fig)

# --- FIGURE 9: Full-Session Population Average Trace with Nosepoke Markers ---
print("Generating full-session population average trace...")
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(t_calcium, mean_pop_trace, color='black', lw=1.5, label=r'Population Mean $\Delta F/F$')

legend_correct_done = False
legend_incorrect_done = False

for idx, f_idx in enumerate(event_frames):
    if 0 <= f_idx < len(t_calcium):
        t_poke = t_calcium[f_idx]
        is_correct = correct_trials[idx]
        
        if is_correct:
            color = '#00FF00'
            lbl = 'Correct Nosepoke' if not legend_correct_done else None
            legend_correct_done = True
        else:
            color = '#FF0000'
            lbl = 'Incorrect Nosepoke' if not legend_incorrect_done else None
            legend_incorrect_done = True
            
        ax.axvline(t_poke, color=color, linestyle=':', lw=2.5, alpha=0.9, label=lbl)

ax.set_title("Full Session Population Average Trace", fontsize=14, fontweight='bold')
ax.set_xlabel("Time Across Video Timeline (Seconds)", fontsize=12)
ax.set_ylabel(r"Population Mean $\Delta F/F$", fontsize=12)
ax.set_xlim([t_calcium[0], t_calcium[-1]])
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
sns.despine()

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Behavior_9_Full_Session_Average_Trace.png"), dpi=300)
plt.close(fig)

print(f"Analysis complete! Figures saved directly in: {OUTPUT_DIR}")