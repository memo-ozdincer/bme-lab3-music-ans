%% acquire_data.m
%  BME Lab 3 - Serial Data Acquisition from Arduino
%  Reads PPG + GSR data and saves to CSV file with a live plot.
%
%  BEFORE RUNNING:
%    1. Upload dual_sensor.ino to the Arduino via Arduino IDE.
%    2. CLOSE the Arduino IDE Serial Monitor (only one program can use the port).
%    3. Update COM_PORT below to match your Arduino port.
%       - Mac:     "/dev/cu.usbmodem14101" (check Arduino IDE > Tools > Port)
%       - Windows: "COM3", "COM4", etc.
%    4. Run this script in MATLAB.
%    5. Press Ctrl+C to stop recording -- data saves automatically.

clear; clc; close all;

%% =================== CONFIGURATION ===================
COM_PORT   = "/dev/cu.usbmodem14101";   % <-- CHANGE THIS TO YOUR PORT
BAUD_RATE  = 115200;
OUTPUT_FILE = "lab3_raw_data.csv";
MAX_DURATION_S = 30 * 60;               % 30-minute safety cutoff
% =======================================================

%% Connect to Arduino
fprintf("Connecting to Arduino on %s ...\n", COM_PORT);
s = serialport(COM_PORT, BAUD_RATE);
configureTerminator(s, "LF");
flush(s);

pause(2);          % Let Arduino finish its setup()
readline(s);       % Discard the CSV header line from Arduino

%% Prepare storage
data = [];
fprintf("Recording started at %s\n", datestr(now, 'HH:MM:SS'));
fprintf("Press Ctrl+C to stop. Data auto-saves to %s\n\n", OUTPUT_FILE);

%% Live plot
figure('Name', 'Live Biosignals', 'NumberTitle', 'off', ...
       'Position', [100 100 1000 500]);

subplot(2,1,1);
h_ppg = animatedline('Color', [0.9 0.2 0.2], 'LineWidth', 0.8);
title('PPG (PulseSensor)');
ylabel('Raw ADC (0-1023)');
grid on;

subplot(2,1,2);
h_gsr = animatedline('Color', [0.2 0.4 0.9], 'LineWidth', 0.8);
title('GSR / EDA');
ylabel('Raw ADC (0-1023)');
xlabel('Elapsed Time (s)');
grid on;

startTic = tic;

%% Auto-save on Ctrl+C or script end
cleanupObj = onCleanup(@() saveData(data, OUTPUT_FILE));

%% Acquisition loop
while toc(startTic) < MAX_DURATION_S
    if s.NumBytesAvailable > 0
        line = readline(s);
        vals = str2double(split(line, ","));

        if length(vals) == 3 && ~any(isnan(vals))
            data = [data; vals'];                     %#ok<AGROW>
            elapsed = toc(startTic);

            % Update plot every 20 samples (~5 Hz refresh)
            if mod(size(data,1), 20) == 0
                addpoints(h_ppg, elapsed, vals(2));
                addpoints(h_gsr, elapsed, vals(3));
                drawnow limitrate;
            end

            % Print a heartbeat every 10 s so you know it's working
            if mod(size(data,1), 1000) == 0
                fprintf("  %6.0f s | %d samples collected\n", elapsed, size(data,1));
            end
        end
    end
end

%% ---- helper ----
function saveData(data, filename)
    fprintf("\n--- Stopping acquisition ---\n");
    if ~isempty(data)
        T = array2table(data, 'VariableNames', {'time_ms','ppg','gsr'});
        writetable(T, filename);
        fprintf("Saved %d samples to %s\n", size(data,1), filename);
    else
        fprintf("No data collected.\n");
    end
end
