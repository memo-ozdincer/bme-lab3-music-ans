%% analyze_data.m
%  BME Lab 3 - Analysis: Music & Autonomic Nervous System
%
%  INSTRUCTIONS:
%    1. Run acquire_data.m first to produce lab3_raw_data.csv
%    2. Fill in phase_start_s below with the timestamps YOU recorded
%       during the experiment (seconds from when recording began).
%    3. Run this script.  Figures are saved as PNGs for your report.

clear; clc; close all;

%% Load raw data
data = readtable("lab3_raw_data.csv");
time_s = (data.time_ms - data.time_ms(1)) / 1000;   % seconds from start
ppg_raw = data.ppg;
gsr_raw = data.gsr;
Fs = round(1 / median(diff(time_s)));                % ~100 Hz
fprintf("Loaded %d samples  |  Fs ~ %d Hz  |  Duration ~ %.0f s\n\n", ...
        height(data), Fs, time_s(end));

%% ============ EDIT THESE TIMESTAMPS (seconds) ============
%  Write down the wall-clock time when you START each phase,
%  then convert to seconds-from-recording-start.
%
%  Example: if you hit record at 2:00:00 PM and Baseline started
%  at 2:00:30 PM, then baseline start = 30.

phase_start_s = [ ...
    0   ;  ... % 1  Baseline           (3 min)
    180 ;  ... % 2  Recovery 1          (2 min)
    300 ;  ... % 3  Calm music          (3 min)
    480 ;  ... % 4  Recovery 2          (2 min)
    600 ;  ... % 5  Intense music       (3 min)
    780 ;  ... % 6  Recovery 3          (2 min)
    900 ;  ... % 7  Happy music         (3 min)
    1080;  ... % 8  Recovery 4          (2 min)
    1200   ... % 9  END marker
];

phase_names = { ...
    'Baseline',      'Recovery 1', ...
    'Calm Music',    'Recovery 2', ...
    'Intense Music', 'Recovery 3', ...
    'Happy Music',   'Recovery 4'  ...
};

nPhases = length(phase_names);
% ===========================================================

%% Convert GSR to skin conductance (micro-siemens)
%  Grove GSR: R = ((1024 + 2*adc) * 100000) / (512 - adc)  [ohms]
%  Conductance = 1/R  -->  micro-siemens = 1e6 / R
gsr_resistance   = ((1024 + 2*gsr_raw) .* 100000) ./ max(512 - gsr_raw, 1);
eda_uS           = 1e6 ./ gsr_resistance;

%% Filter PPG and detect peaks --> Heart Rate
%  2nd-order Butterworth band-pass 0.5-5 Hz
[b, a]   = butter(2, [0.5 5] / (Fs/2), 'bandpass');
ppg_filt = filtfilt(b, a, double(ppg_raw));

%  Peak detection (min distance 0.4 s = 150 BPM ceiling)
[~, locs] = findpeaks(ppg_filt, ...
    'MinPeakDistance',    round(0.4 * Fs), ...
    'MinPeakProminence',  0.3 * std(ppg_filt));

peak_t   = time_s(locs);
rr       = diff(peak_t);           % inter-beat intervals (s)
inst_hr  = 60 ./ rr;               % instantaneous BPM
hr_t     = peak_t(2:end);

%  Discard non-physiological values
good     = inst_hr > 40 & inst_hr < 180;
inst_hr  = inst_hr(good);
hr_t     = hr_t(good);

%% Per-phase statistics
fprintf("%-20s  %12s  %12s\n", "Phase", "HR (BPM)", "EDA (uS)");
fprintf("%s\n", repmat('-', 1, 48));

mean_hr  = zeros(1, nPhases);
std_hr   = zeros(1, nPhases);
mean_eda = zeros(1, nPhases);
std_eda  = zeros(1, nPhases);

for i = 1:nPhases
    t0 = phase_start_s(i);
    t1 = phase_start_s(i+1);

    m_hr          = hr_t >= t0 & hr_t < t1;
    mean_hr(i)    = mean(inst_hr(m_hr));
    std_hr(i)     = std(inst_hr(m_hr));

    m_eda         = time_s >= t0 & time_s < t1;
    mean_eda(i)   = mean(eda_uS(m_eda));
    std_eda(i)    = std(eda_uS(m_eda));

    fprintf("%-20s  %5.1f +/- %-4.1f  %5.2f +/- %-4.2f\n", ...
            phase_names{i}, mean_hr(i), std_hr(i), mean_eda(i), std_eda(i));
end

%% ===== FIGURE 1: Full time-series with phase markers =====
fig1 = figure('Name','Time Series','Position',[50 50 1300 600]);

ax1 = subplot(2,1,1);
plot(hr_t, inst_hr, 'r.-', 'MarkerSize', 3);
hold on;
for i = 1:nPhases
    xline(phase_start_s(i), '--', phase_names{i}, ...
          'LabelOrientation','aligned', ...
          'LabelVerticalAlignment','top', 'FontSize', 7, 'Color', [.4 .4 .4]);
end
ylabel('Heart Rate (BPM)');
title('Instantaneous Heart Rate');
xlim([0 time_s(end)]); grid on;

ax2 = subplot(2,1,2);
plot(time_s, eda_uS, 'b-', 'LineWidth', 0.6);
hold on;
for i = 1:nPhases
    xline(phase_start_s(i), '--', phase_names{i}, ...
          'LabelOrientation','aligned', ...
          'LabelVerticalAlignment','top', 'FontSize', 7, 'Color', [.4 .4 .4]);
end
ylabel('Skin Conductance (\muS)');
xlabel('Time (s)');
title('Electrodermal Activity');
xlim([0 time_s(end)]); grid on;

linkaxes([ax1 ax2], 'x');
saveas(fig1, 'fig1_timeseries.png');
fprintf("\nSaved fig1_timeseries.png\n");

%% ===== FIGURE 2: Bar-chart comparison of music conditions =====
cond_idx    = [1, 3, 5, 7];   % Baseline, Calm, Intense, Happy
cond_labels = phase_names(cond_idx);
cmap = [0.70 0.70 0.70;   % grey   - baseline
        0.20 0.60 1.00;   % blue   - calm
        1.00 0.30 0.30;   % red    - intense
        1.00 0.80 0.20];  % yellow - happy

fig2 = figure('Name','Condition Comparison','Position',[50 50 950 450]);

subplot(1,2,1);
b1 = bar(mean_hr(cond_idx));
b1.FaceColor = 'flat';
for k = 1:4, b1.CData(k,:) = cmap(k,:); end
hold on;
errorbar(1:4, mean_hr(cond_idx), std_hr(cond_idx), 'k.', 'LineWidth', 1.2, 'CapSize', 8);
set(gca, 'XTickLabel', cond_labels, 'XTickLabelRotation', 25);
ylabel('Mean Heart Rate (BPM)');
title('Heart Rate by Condition');
grid on;

subplot(1,2,2);
b2 = bar(mean_eda(cond_idx));
b2.FaceColor = 'flat';
for k = 1:4, b2.CData(k,:) = cmap(k,:); end
hold on;
errorbar(1:4, mean_eda(cond_idx), std_eda(cond_idx), 'k.', 'LineWidth', 1.2, 'CapSize', 8);
set(gca, 'XTickLabel', cond_labels, 'XTickLabelRotation', 25);
ylabel('Mean Skin Conductance (\muS)');
title('EDA by Condition');
grid on;

saveas(fig2, 'fig2_comparison.png');
fprintf("Saved fig2_comparison.png\n");

%% ===== FIGURE 3: PPG waveform snippet =====
fig3 = figure('Name','PPG Waveform','Position',[50 50 900 300]);
snip0 = phase_start_s(1) + 30;     % 30 s into baseline
snip1 = snip0 + 10;                % 10-second window
mask  = time_s >= snip0 & time_s <= snip1;
plot(time_s(mask), ppg_filt(mask), 'r-', 'LineWidth', 1.2);
hold on;
pk_mask = peak_t >= snip0 & peak_t <= snip1;
plot(peak_t(pk_mask), ppg_filt(locs(pk_mask)), 'kv', 'MarkerSize', 6, 'MarkerFaceColor','k');
xlabel('Time (s)');
ylabel('Filtered PPG (a.u.)');
title('PPG Waveform with Detected Peaks (10 s Baseline Snippet)');
legend('Filtered PPG', 'Detected peaks');
grid on;

saveas(fig3, 'fig3_ppg_waveform.png');
fprintf("Saved fig3_ppg_waveform.png\n");

%% ===== FIGURE 4: Heart-rate variability (RMSSD per phase) =====
fig4 = figure('Name','HRV','Position',[50 50 600 350]);
rmssd = zeros(1, nPhases);
for i = 1:nPhases
    t0 = phase_start_s(i);
    t1 = phase_start_s(i+1);
    m  = hr_t >= t0 & hr_t < t1;
    rr_phase = 1000 * 60 ./ inst_hr(m);   % convert BPM back to RR in ms
    diffs    = diff(rr_phase);
    rmssd(i) = sqrt(mean(diffs.^2));
end
bar(rmssd(cond_idx), 'FaceColor', [0.5 0.7 0.9]);
set(gca, 'XTickLabel', cond_labels, 'XTickLabelRotation', 25);
ylabel('RMSSD (ms)');
title('Heart-Rate Variability by Condition');
grid on;

saveas(fig4, 'fig4_hrv.png');
fprintf("Saved fig4_hrv.png\n");

fprintf("\n=== Analysis complete ===\n");
