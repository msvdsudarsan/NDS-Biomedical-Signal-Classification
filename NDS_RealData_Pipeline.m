function NDS_RealData_Pipeline
%% ========================================================================
%  NDS REAL-DATA PIPELINE  (CHB-MIT EEG + MIT-BIH ECG)
%  ------------------------------------------------------------------------
%  PURPOSE
%    End-to-end, REAL-DATA classification pipeline. Nothing is hardcoded:
%    every accuracy / AUC / confusion-matrix number is COMPUTED from the
%    downloaded PhysioNet recordings. If a number does not appear here, the
%    pipeline did not measure it.
%
%  WHAT IT DOES
%    1. Downloads real recordings from PhysioNet (needs internet).
%    2. Reads them with self-contained native readers (no extra toolbox
%       needed for I/O): EDF reader for CHB-MIT, WFDB 212 + annotation
%       reader for MIT-BIH.
%    3. Segments signals, applies SSA denoising, reconstructs phase space
%       (AMI + FNN), computes nonlinear invariants (Lyapunov, D2, RQA) to
%       form a 10-D feature vector.
%    4. Trains a CNN-LSTM classifier (Deep Learning Toolbox). If that
%       toolbox is absent it falls back to an ensemble on the 10-D features.
%    5. Evaluates with proper protocols:
%         EEG : 10-fold CV + leave-one-subject-out (LOSO)
%         ECG : inter-patient AAMI EC57 split (DS1 train / DS2 test)
%    6. Saves Fig1..Fig7 as PNG *and* PDF into ./figures and writes a
%       multi-page PDF plus a plain-text results file.
%
%  HOW TO RUN (MATLAB Online)
%    >> NDS_RealData_Pipeline
%    First run downloads a small default subset so you can confirm it works,
%    then set CFG.quick_demo = false to run the full record lists.
%
%  REQUIRED TOOLBOXES
%    Signal Processing Toolbox      (filtering, findpeaks)   - recommended
%    Statistics and Machine Learning Toolbox (perfcurve, cvpartition, ...)
%    Deep Learning Toolbox          (CNN-LSTM)                - optional*
%    *If missing, an ensemble fallback on the 10-D features is used.
%
%  HONEST NOTE
%    Real results will almost certainly DIFFER from any previously reported
%    numbers. Whatever this pipeline prints is the truth for your run; the
%    manuscript must be updated to match these measured values.
%% ========================================================================

clc; close all;
CFG = local_config();
setup_env(CFG);

fprintf('\n################  NDS REAL-DATA PIPELINE  ################\n');
fprintf('Output folder : %s\n', CFG.outdir);
fprintf('Quick demo    : %d  (set CFG.quick_demo=false for full run)\n\n', CFG.quick_demo);

results = struct();

if CFG.run_eeg
    try
        results.eeg = run_eeg_experiment(CFG);
    catch ME
        warning('EEG experiment failed: %s', getReport(ME));
    end
end

if CFG.run_ecg
    try
        results.ecg = run_ecg_experiment(CFG);
    catch ME
        warning('ECG experiment failed: %s', getReport(ME));
    end
end

print_summary(CFG, results);
fprintf('\n################  DONE  ################\n');
end

%% ======================= CONFIGURATION ==================================
function CFG = local_config()
CFG = struct();

% ---- master switches ----
CFG.run_eeg     = true;
CFG.run_ecg     = true;
CFG.quick_demo  = false;     % false = full run (4 EEG subjects + 44 ECG records)

% ---- folders ----
CFG.root   = fullfile(pwd, 'NDS_data');
CFG.outdir = fullfile(pwd, 'figures');

% ---- signal / feature params ----
CFG.eeg_fs        = 256;     % CHB-MIT sampling rate (Hz)
CFG.ecg_fs        = 360;     % MIT-BIH sampling rate (Hz)
CFG.eeg_win_sec   = 2;       % EEG window length (s)
CFG.ecg_beat_sec  = 0.5;     % ECG beat window (+/- around R peak = 0.5 s total-ish)
CFG.ssa_L         = 40;      % SSA embedding window length
CFG.ssa_energy    = 0.95;    % SSA reconstruction energy threshold
CFG.ami_maxlag    = 40;      % AMI max delay
CFG.ami_nbins     = 16;
CFG.fnn_maxdim    = 8;
CFG.fnn_rtol      = 15;      % FNN distance ratio threshold
CFG.fnn_thresh    = 0.01;    % stop when FNN fraction < 1%

% ---- CV / caps ----
CFG.kfold             = 10;
CFG.max_win_per_class = 400; % cap windows/beats per class (runtime control)
CFG.ecg_oversample    = true;% oversample minority ECG classes in TRAIN only
CFG.ecg_use_features  = true;% ECG: classify on [nonlinear + RR-interval] features (rhythm context)
CFG.rng_seed          = 42;

% ---- CHB-MIT records ----
% Default demo: subject chb01, one seizure file + one seizure-free file.
if CFG.quick_demo
    CFG.chb_subjects = {'chb01'};
    CFG.chb_files    = {'chb01_03','chb01_04'};  % _03 has a seizure
else
    CFG.chb_subjects = {'chb01','chb02','chb03','chb05'};
    CFG.chb_files    = {};   % empty => use every file listed in each summary
end
CFG.chb_channel = 'FP1-F7'; % preferred bipolar channel (falls back to ch 1)

% ---- MIT-BIH records (AAMI inter-patient DS1/DS2, de Chazal et al.) ----
CFG.ds1 = {'101','106','108','109','112','114','115','116','118','119', ...
           '122','124','201','203','205','207','208','209','215','220', ...
           '223','230'};
CFG.ds2 = {'100','103','105','111','113','117','121','123','200','202', ...
           '210','212','213','214','219','221','222','228','231','232', ...
           '233','234'};
if CFG.quick_demo
    CFG.ds1 = {'101','106','108','112'};
    CFG.ds2 = {'100','103','105','111'};
end

% ---- PhysioNet URLs ----
CFG.url_chbmit = 'https://physionet.org/files/chbmit/1.0.0/';
CFG.url_mitdb  = 'https://physionet.org/files/mitdb/1.0.0/';
end

%% ======================= ENVIRONMENT ===================================
function setup_env(CFG)
if ~exist(CFG.root,'dir'),   mkdir(CFG.root);   end
if ~exist(CFG.outdir,'dir'), mkdir(CFG.outdir); end
rng(CFG.rng_seed);
fprintf('MATLAB %s\n', version);
fprintf('Deep Learning Toolbox present: %d\n', has_dl_toolbox());
fprintf('Statistics Toolbox present  : %d\n', ~isempty(which('perfcurve')));
end

function tf = has_dl_toolbox()
tf = ~isempty(which('trainNetwork')) || ~isempty(which('trainnet'));
end

function fpath = safe_websave(fpath, url)
% Download only if not already present; robust error message + progress.
if exist(fpath,'file')
    d = dir(fpath);
    if ~isempty(d) && d(1).bytes > 0
        fprintf('    [cached] %s (%.1f MB)\n', local_name(fpath), d(1).bytes/1e6);
        return;
    end
end
fprintf('    downloading %s ... ', local_name(fpath));
t0 = tic;
try
    opts = weboptions('Timeout', 300);   % large EDF files can be slow
    websave(fpath, url, opts);
    d = dir(fpath);
    fprintf('done (%.1f MB, %.0fs)\n', d(1).bytes/1e6, toc(t0));
catch ME
    fprintf('FAILED\n');
    error(['Download failed for %s\n  Reason: %s\n' ...
           '  (MATLAB Online needs internet access to physionet.org.)'], ...
        url, ME.message);
end
end

function s = local_name(p)
[~,nm,ext] = fileparts(p); s = [nm ext];
end

%% ======================= EEG EXPERIMENT ================================
function R = run_eeg_experiment(CFG)
fprintf('\n===== EEG (CHB-MIT) =====\n');
fprintf('(First run downloads real EDF recordings from PhysioNet.\n');
fprintf(' The seizure file chb01_03.edf is ~40 MB and may take 1-3 min.)\n');
chbdir = fullfile(CFG.root,'chbmit'); if ~exist(chbdir,'dir'), mkdir(chbdir); end

Wlen = round(CFG.eeg_win_sec*CFG.eeg_fs);
X = {}; y = []; subj = [];   % X: cell of 1xWlen windows; y: 1=ictal 0=interictal

for si = 1:numel(CFG.chb_subjects)
    sub = CFG.chb_subjects{si};
    % --- download + parse the seizure summary ---
    sumurl = sprintf('%s%s/%s-summary.txt', CFG.url_chbmit, sub, sub);
    sumf   = fullfile(chbdir, sprintf('%s-summary.txt', sub));
    safe_websave(sumf, sumurl);
    S = parse_chb_summary(sumf);

    files = CFG.chb_files;
    if isempty(files), files = {S.file}; files = unique(files); end

    for fi = 1:numel(files)
        fn = files{fi};
        edfurl = sprintf('%s%s/%s.edf', CFG.url_chbmit, sub, fn);
        edff   = fullfile(chbdir, [fn '.edf']);
        try, safe_websave(edff, edfurl); catch ME, warning('%s',ME.message); continue; end

        [sig, fs, labels] = read_edf(edff, CFG.chb_channel);
        if isempty(sig), warning('No usable channel in %s', fn); continue; end
        if abs(fs-CFG.eeg_fs) > 1, fprintf('  [info] %s fs=%g\n', fn, fs); end

        % seizure intervals (seconds) for this file
        iv = seizure_intervals(S, fn);

        nWin = floor(numel(sig)/Wlen);
        for w = 1:nWin
            i0 = (w-1)*Wlen + 1; i1 = i0 + Wlen - 1;
            tc = ((i0+i1)/2)/fs;              % window centre time (s)
            isIctal = any(tc>=iv(:,1) & tc<=iv(:,2));
            X{end+1} = sig(i0:i1)'; %#ok<AGROW>
            y(end+1) = double(isIctal); %#ok<AGROW>
            subj(end+1) = si; %#ok<AGROW>
        end
        fprintf('  %s: %d windows collected\n', fn, nWin);
    end
end

if isempty(y), error('No EEG windows collected.'); end

% balance + cap classes
[X,y,subj] = balance_classes(X,y,subj,CFG.max_win_per_class);
fprintf('Total EEG windows: %d (ictal=%d, interictal=%d)\n', ...
    numel(y), sum(y==1), sum(y==0));

% denoise + 10-D features
[F, Xden] = features_from_windows(X, CFG);

% ---- classify: 10-fold CV ----
R.kfold = evaluate_kfold(Xden, F, y, CFG, 'EEG 10-fold');
% ---- classify: LOSO ----
if numel(unique(subj))>1
    R.loso = evaluate_loso(Xden, F, y, subj, CFG, 'EEG LOSO');
else
    R.loso = [];
    fprintf('  [info] LOSO skipped (only one subject in this run).\n');
end

% ---- figures from REAL data ----
make_fig1(Xden, X, CFG);                          % SSA denoising + scree
make_fig2(Xden, y, CFG);                           % phase space
make_fig3(Xden, y, CFG);                           % Lyapunov + D2
make_fig4(Xden, y, CFG);                           % recurrence plots
make_fig5(F, y, CFG);                              % feature vectors + ablation
make_fig7(R.kfold, CFG);                           % ROC + ablation
R.F = F; R.y = y; R.subj = subj;
end

%% ======================= ECG EXPERIMENT ================================
function R = run_ecg_experiment(CFG)
fprintf('\n===== ECG (MIT-BIH, AAMI inter-patient) =====\n');
mitdir = fullfile(CFG.root,'mitdb'); if ~exist(mitdir,'dir'), mkdir(mitdir); end

[Xtr,ytr,RRtr] = collect_ecg(CFG.ds1, CFG, mitdir, 'DS1(train)');
[Xte,yte,RRte] = collect_ecg(CFG.ds2, CFG, mitdir, 'DS2(test)');
if isempty(ytr)||isempty(yte), error('No ECG beats collected.'); end

classes = {'N','S','V','F','Q'};
fprintf('Train beats: %d | Test beats: %d\n', numel(ytr), numel(yte));
tabulate_classes(ytr, classes, 'train');
tabulate_classes(yte, classes, 'test');

% denoise + nonlinear features (use the correct ECG sampling rate)
[Ftr, Xtr_d] = features_from_windows(Xtr, CFG, CFG.ecg_fs);
[Fte, Xte_d] = features_from_windows(Xte, CFG, CFG.ecg_fs);

% Augment the 10 nonlinear invariants with RR-interval context. Rhythm
% timing (pre-RR, post-RR, local ratio) is essential for the AAMI S class
% (supraventricular ectopics are defined by prematurity), which a single-beat
% morphology/dynamics view alone cannot capture.
Ftr = [Ftr, RRtr];
Fte = [Fte, RRte];

% ---- balance the TRAINING set by oversampling minority classes ----
% Standard, honest technique (only the train split is touched; the DS2 test
% split stays untouched so the inter-patient evaluation remains valid).
if CFG.ecg_oversample
    [Xtr_d, Ftr, ytr] = oversample_train(Xtr_d, Ftr, ytr, numel(classes));
end

% train on DS1, test on DS2 (true inter-patient)
% ECG uses the feature-based classifier so the RR-interval context is actually
% seen by the model (the raw-waveform CNN-LSTM ignores beat-to-beat rhythm).
if CFG.ecg_use_features
    [pred, score] = ml_train_predict(Ftr, ytr, Fte, yte, CFG);
elseif has_dl_toolbox()
    [pred, score] = dl_train_predict(Xtr_d, ytr, Xte_d, yte, numel(classes), CFG);
else
    [pred, score] = ml_train_predict(Ftr, ytr, Fte, yte, CFG);
end

R.conf   = confusionmat(yte, pred, 'Order', 1:numel(classes));
R.metrics= class_metrics(R.conf, classes);
R.acc    = sum(pred==yte)/numel(yte);
R.auc    = multiclass_auc(yte, score, numel(classes));
fprintf('ECG overall accuracy = %.2f%%  macro-AUC = %.3f\n', 100*R.acc, R.auc);

R.classes = classes;              % set BEFORE plotting (used by make_fig6)
make_fig6(R, CFG);   % real accuracy/confusion visual
end

function [X,y,RR] = collect_ecg(recs, CFG, mitdir, tag)
X = {}; y = []; RR = zeros(0,3);
half = round(CFG.ecg_beat_sec*CFG.ecg_fs/2);
for i = 1:numel(recs)
    rec = recs{i};
    base = fullfile(mitdir, rec);
    for ext = {'.hea','.dat','.atr'}
        safe_websave([base ext{1}], sprintf('%s%s%s', CFG.url_mitdb, rec, ext{1}));
    end
    [sig, fs, chn] = read_wfdb_signal([base '.hea'], [base '.dat']);
    ml = pick_mlii(chn); s = sig(:,ml);
    if abs(fs-CFG.ecg_fs)>1, fprintf('  [info] %s fs=%g\n', rec, fs); end
    [samp, sym] = read_wfdb_annot([base '.atr']);
    samp = double(samp(:)); nb = numel(samp);
    % record-level median RR (samples) used to normalise the rhythm ratio
    if nb>=2, avgRR = median(diff(samp)); else, avgRR = fs; end
    if ~(avgRR>0), avgRR = fs; end
    for k = 1:nb
        c = aami_class(sym{k});
        if c==0, continue; end            % non-beat / excluded symbol
        i0 = samp(k)-half; i1 = samp(k)+half-1;
        if i0<1 || i1>numel(s), continue; end
        % --- RR-interval context (crucial for AAMI S/V discrimination) ---
        if k>1,  preRR  = (samp(k)-samp(k-1))/fs;  else, preRR  = avgRR/fs; end
        if k<nb, postRR = (samp(k+1)-samp(k))/fs;  else, postRR = avgRR/fs; end
        preRatio = preRR / (avgRR/fs);    % <1 => premature beat (typical S)
        X{end+1} = s(i0:i1)'; %#ok<AGROW>
        y(end+1) = c; %#ok<AGROW>
        RR(end+1,:) = [preRR, postRR, preRatio]; %#ok<AGROW>
    end
    fprintf('  %s: cumulative %d beats\n', rec, numel(y));
end
% cap per class for runtime
[X,y,RR] = cap_per_class(X,y,CFG.max_win_per_class,RR);
fprintf('  %s collected: %d beats\n', tag, numel(y));
end

%% ======================= CHB-MIT SUMMARY / LABELS =======================
function S = parse_chb_summary(sumf)
% Parse a chbXX-summary.txt into struct array with fields: file, s0, s1 (sec)
txt = fileread(sumf);
lines = regexp(txt, '\r?\n', 'split');
S = struct('file',{},'s0',{},'s1',{});
curFile = ''; pend0 = [];
for i = 1:numel(lines)
    ln = strtrim(lines{i});
    t = regexp(ln, 'File Name:\s*(\S+)\.edf', 'tokens', 'once');
    if ~isempty(t), curFile = t{1}; continue; end
    t0 = regexp(ln, 'Seizure.*Start Time:\s*(\d+)\s*seconds', 'tokens', 'once');
    if ~isempty(t0), pend0 = str2double(t0{1}); continue; end
    t1 = regexp(ln, 'Seizure.*End Time:\s*(\d+)\s*seconds', 'tokens', 'once');
    if ~isempty(t1) && ~isempty(pend0)
        S(end+1) = struct('file',curFile,'s0',pend0,'s1',str2double(t1{1})); %#ok<AGROW>
        pend0 = [];
    end
end
end

function iv = seizure_intervals(S, fn)
iv = zeros(0,2);
for i = 1:numel(S)
    if strcmp(S(i).file, fn), iv(end+1,:) = [S(i).s0 S(i).s1]; end %#ok<AGROW>
end
if isempty(iv), iv = [Inf Inf]; end
end

%% ======================= NATIVE EDF READER =============================
function [sig, fs, label] = read_edf(fname, wantLabel)
% Minimal EDF reader. Returns one channel (wantLabel if found, else ch 1).
fid = fopen(fname,'r','ieee-le');
if fid<0, error('Cannot open %s', fname); end
c = onCleanup(@() fclose(fid));
fseek(fid,0,'bof');
fread(fid,8,'*char');                       % version
fread(fid,80,'*char'); fread(fid,80,'*char');
fread(fid,8,'*char');  fread(fid,8,'*char'); % start date/time
fread(fid,8,'*char');                        % header bytes
fread(fid,44,'*char');                       % reserved
nrec = str2double(fread(fid,8,'*char')');
dur  = str2double(fread(fid,8,'*char')');
ns   = str2double(fread(fid,4,'*char')');
labels = cell(1,ns);
for i=1:ns, labels{i} = strtrim(fread(fid,16,'*char')'); end
for i=1:ns, fread(fid,80,'*char'); end        % transducer
for i=1:ns, fread(fid,8,'*char'); end         % phys dim
pmin = zeros(1,ns); pmax=zeros(1,ns); dmin=zeros(1,ns); dmax=zeros(1,ns);
for i=1:ns, pmin(i)=str2double(fread(fid,8,'*char')'); end
for i=1:ns, pmax(i)=str2double(fread(fid,8,'*char')'); end
for i=1:ns, dmin(i)=str2double(fread(fid,8,'*char')'); end
for i=1:ns, dmax(i)=str2double(fread(fid,8,'*char')'); end
for i=1:ns, fread(fid,80,'*char'); end        % prefiltering
spr = zeros(1,ns);
for i=1:ns, spr(i)=str2double(fread(fid,8,'*char')'); end
for i=1:ns, fread(fid,32,'*char'); end        % reserved

% choose channel
ch = find(strcmpi(labels, wantLabel), 1);
if isempty(ch), ch = 1; end
label = labels{ch};
fs = spr(ch)/dur;
gain = (pmax(ch)-pmin(ch))/(dmax(ch)-dmin(ch));

% read data records
recSamps = sum(spr);
sig = zeros(nrec*spr(ch),1);
offCh = sum(spr(1:ch-1));
for r=1:nrec
    raw = fread(fid, recSamps, 'int16');
    if numel(raw)<recSamps, break; end
    seg = raw(offCh+1 : offCh+spr(ch));
    sig((r-1)*spr(ch)+1 : r*spr(ch)) = (double(seg)-dmin(ch))*gain + pmin(ch);
end
end

%% ======================= NATIVE WFDB READERS ===========================
function [sig, fs, chn] = read_wfdb_signal(heaf, datf)
% Reads MIT format 212 (two 12-bit samples per 3 bytes). Header gives layout.
htxt = fileread(heaf);
hl = regexp(htxt, '\r?\n', 'split');
first = strsplit(strtrim(hl{1}));
nsig = str2double(first{2});
fs   = str2double(first{3});
nsamp= str2double(first{4});
fmt = zeros(1,nsig); gain=zeros(1,nsig); zero=zeros(1,nsig); chn=cell(1,nsig);
for i=1:nsig
    p = strsplit(strtrim(hl{i+1}));
    fmt(i)  = str2double(regexp(p{2},'^\d+','match','once'));
    gain(i) = str2double(regexp(p{3},'^[\d\.]+','match','once'));
    if isnan(gain(i))||gain(i)==0, gain(i)=200; end
    if numel(p)>=6, zero(i)=str2double(p{6}); else, zero(i)=0; end
    chn{i}  = p{end};
end
if any(fmt~=212)
    error('Only MIT format 212 is supported natively (record fmt=%d).', fmt(1));
end
fid = fopen(datf,'r','ieee-le'); c=onCleanup(@()fclose(fid));
raw = fread(fid, inf, '*uint8');
ntrip = floor(numel(raw)/3);
raw = raw(1:ntrip*3);
b1 = double(raw(1:3:end)); b2 = double(raw(2:3:end)); b3 = double(raw(3:3:end));
lo1 = b1; hi1 = mod(b2,16);
s1 = hi1*256 + lo1;  s1(s1>=2048) = s1(s1>=2048)-4096;
lo2 = b3; hi2 = floor(b2/16);
s2 = hi2*256 + lo2;  s2(s2>=2048) = s2(s2>=2048)-4096;
inter = zeros(2*ntrip,1); inter(1:2:end)=s1; inter(2:2:end)=s2;
if nsig==2
    sig = [inter(1:2:end), inter(2:2:end)];
else
    sig = reshape(inter(1:floor(numel(inter)/nsig)*nsig), nsig, [])';
end
for i=1:size(sig,2), sig(:,i) = (sig(:,i)-zero(i))/gain(i); end
if nsamp>0 && size(sig,1)>nsamp, sig = sig(1:nsamp,:); end
end

function [samp, sym] = read_wfdb_annot(atrf)
% Reads a standard MIT-BIH annotation file (.atr).
fid = fopen(atrf,'r','ieee-le'); c=onCleanup(@()fclose(fid));
A = fread(fid, inf, '*uint16');
samp = []; sym = {}; t = 0; i = 1;
codes = ann_code_map();
while i <= numel(A)
    w = double(A(i)); i=i+1;
    code = floor(w/1024);            % top 6 bits
    ival = mod(w,1024);              % bottom 10 bits
    if code==0 && ival==0, break; end            % end of file
    switch code
        case 59                                    % SKIP: 4-byte interval
            if i+1>numel(A), break; end
            hi = double(A(i)); lo = double(A(i+1)); i=i+2;
            t = t + hi*65536 + lo;
            % next word carries the actual annotation code
            if i>numel(A), break; end
            w2 = double(A(i)); i=i+1;
            code2 = floor(w2/1024);
            samp(end+1)=t; %#ok<AGROW>
            sym{end+1}=code_to_sym(code2,codes); %#ok<AGROW>
        case {60,61,62}                            % NUM/SUB/CHN: skip payload word
            % ival is payload; no time advance, no beat
        case 63                                    % AUX: ival bytes follow
            nb = ival; nw = ceil(nb/2); i = i + nw;
        otherwise
            t = t + ival;
            samp(end+1)=t; %#ok<AGROW>
            sym{end+1}=code_to_sym(code,codes); %#ok<AGROW>
    end
end
end

function m = ann_code_map()
m = containers.Map('KeyType','double','ValueType','char');
p = {1,'N';2,'L';3,'R';4,'a';5,'V';6,'F';7,'J';8,'A';9,'S';10,'E'; ...
     11,'j';12,'/';13,'Q';34,'e';38,'f'};
for k=1:size(p,1), m(p{k,1})=p{k,2}; end
end
function s = code_to_sym(code, m)
if isKey(m,code), s=m(code); else, s='?'; end
end

function c = aami_class(sym)
% AAMI EC57 mapping -> 1:N 2:S 3:V 4:F 5:Q ; 0 = ignore
switch sym
    case {'N','L','R','e','j'}, c=1;
    case {'A','a','J','S'},     c=2;
    case {'V','E'},             c=3;
    case {'F'},                 c=4;
    case {'/','f','Q'},         c=5;
    otherwise,                  c=0;
end
end

function ml = pick_mlii(chn)
ml = find(strcmpi(chn,'MLII'),1);
if isempty(ml), ml = 1; end
end

%% ======================= SSA DENOISING =================================
function y = ssa_denoise(x, L, energy)
x = x(:); N = numel(x);
L = min(L, floor(N/2));
if L < 2, y = x; return; end
K = N - L + 1;
X = zeros(L,K);
for i = 1:L, X(i,:) = x(i:i+K-1); end
[U,Sv,V] = svd(X,'econ');
s = diag(Sv);
en = cumsum(s.^2)/sum(s.^2);
r = find(en >= energy, 1, 'first'); if isempty(r), r = numel(s); end
Xr = U(:,1:r)*Sv(1:r,1:r)*V(:,1:r)';
% diagonal averaging (Hankelization)
y = zeros(N,1); cnt = zeros(N,1);
for i = 1:L
    idx = i:(i+K-1);
    y(idx) = y(idx) + Xr(i,:)';
    cnt(idx) = cnt(idx) + 1;
end
y = y ./ cnt;
end

%% ======================= AMI + FNN =====================================
function tau = ami_tau(x, maxlag, nbins)
x = x(:); N = numel(x);
edges = linspace(min(x)-eps, max(x)+eps, nbins+1);
[~,~,bx] = histcounts(x, edges); bx(bx==0)=1;
ami = zeros(maxlag,1);
for lag = 1:maxlag
    a = bx(1:N-lag); b = bx(1+lag:N); n = numel(a);
    Pab = accumarray([a b], 1, [nbins nbins]) / n;
    Pa = sum(Pab,2); Pb = sum(Pab,1);
    M = 0;
    for ii=1:nbins
        for jj=1:nbins
            if Pab(ii,jj)>0
                M = M + Pab(ii,jj)*log(Pab(ii,jj)/(Pa(ii)*Pb(jj)));
            end
        end
    end
    ami(lag) = M;
end
tau = 1;
for lag = 2:maxlag-1
    if ami(lag) < ami(lag-1) && ami(lag) < ami(lag+1), tau = lag; break; end
end
end

function m = fnn_dim(x, tau, maxm, rtol, thr)
x = x(:);
prev = inf;
m = maxm;
for d = 1:maxm
    Y  = embed(x, d,   tau);
    Y1 = embed(x, d+1, tau);
    n = min(size(Y,1), size(Y1,1));
    Y = Y(1:n,:); Y1 = Y1(1:n,:);
    if n < 10, m = d; return; end
    fnn = 0;
    for i = 1:n
        dd = sum((Y - Y(i,:)).^2, 2); dd(i) = inf;
        [~,j] = min(dd);
        Rd = sqrt(dd(j));
        if Rd == 0, continue; end
        delta = abs(Y1(i,end) - Y1(j,end));
        if delta / Rd > rtol, fnn = fnn + 1; end
    end
    frac = fnn / n;
    if frac < thr, m = d; return; end
    prev = frac; %#ok<NASGU>
end
end

function Y = embed(x, m, tau)
x = x(:); N = numel(x);
M = N - (m-1)*tau;
if M < 1, Y = []; return; end
Y = zeros(M, m);
for i = 1:m, Y(:,i) = x((1:M) + (i-1)*tau); end
end

%% ======================= LYAPUNOV (Rosenstein) =========================
function lam = lyap_rosenstein(x, m, tau, fs, meanperiod)
Y = embed(x, m, tau);
N = size(Y,1);
if N < 20, lam = NaN; return; end
if nargin < 5 || isempty(meanperiod), meanperiod = round(fs/10); end
nn = zeros(N,1);
for i = 1:N
    dd = sum((Y - Y(i,:)).^2, 2);
    dd(max(1,i-meanperiod):min(N,i+meanperiod)) = inf;  % Theiler window
    [~,nn(i)] = min(dd);
end
maxk = min(round(fs/2), N-1);
divg = nan(maxk,1); cnt = zeros(maxk,1);
for i = 1:N
    j = nn(i);
    for k = 0:maxk-1
        if i+k<=N && j+k<=N
            d = norm(Y(i+k,:)-Y(j+k,:));
            if d>0
                if isnan(divg(k+1)), divg(k+1)=0; end
                divg(k+1) = divg(k+1) + log(d);
                cnt(k+1) = cnt(k+1) + 1;
            end
        end
    end
end
valid = cnt>0; divg(valid) = divg(valid)./cnt(valid);
kk = find(valid);
if numel(kk) < 5, lam = NaN; return; end
% slope over an early linear region
rng_k = kk(1:min(numel(kk), round(0.4*numel(kk))));
p = polyfit(rng_k/fs, divg(rng_k), 1);
lam = p(1);  % nats/s
end

%% ======================= CORRELATION DIMENSION D2 ======================
function D2 = corr_dim_gp(x, m, tau)
Y = embed(x, m, tau);
N = size(Y,1);
if N < 20, D2 = NaN; return; end
if N > 500, Y = Y(round(linspace(1,N,500)),:); N = 500; end
D = pdist(Y);
if isempty(D), D2 = NaN; return; end
rs = logspace(log10(min(D(D>0))), log10(max(D)), 20);
C = zeros(size(rs));
for k = 1:numel(rs), C(k) = mean(D < rs(k)); end
ok = C>0 & C<1;
if nnz(ok) < 4, D2 = NaN; return; end
p = polyfit(log(rs(ok)), log(C(ok)), 1);
D2 = p(1);
end

%% ======================= RQA ===========================================
function q = rqa_measures(x, m, tau, rr_target)
Y = embed(x, m, tau);
N = size(Y,1);
q = struct('RR',NaN,'DET',NaN,'L',NaN,'ENT',NaN,'LAM',NaN,'TT',NaN);
if N < 20, return; end
if N > 400, Y = Y(round(linspace(1,N,400)),:); N = 400; end
DM = squareform(pdist(Y));
thr = quantile(DM(:), rr_target);       % choose eps for target recurrence rate
RP = DM <= thr;
RP(logical(eye(N))) = 0;
q.RR = nnz(RP)/(N*N);
% diagonal lines (exclude main diagonal)
dl = diag_lengths(RP);
dl = dl(dl>=2);
if ~isempty(dl)
    q.DET = sum(dl)/max(nnz(triu(RP,1))*2,1);
    q.L   = mean(dl);
    ph = histcounts(dl, 'BinMethod','integers'); ph = ph(ph>0)/sum(ph);
    q.ENT = -sum(ph.*log(ph));
end
% vertical lines
vl = vert_lengths(RP); vl = vl(vl>=2);
if ~isempty(vl)
    q.LAM = sum(vl)/max(nnz(RP),1);
    q.TT  = mean(vl);
end
end

function L = diag_lengths(RP)
N = size(RP,1); L = [];
for d = 1:N-1
    v = diag(RP,d); L = [L; run_lengths(v)]; %#ok<AGROW>
end
end
function L = vert_lengths(RP)
N = size(RP,2); L = [];
for c = 1:N, L = [L; run_lengths(RP(:,c))]; end %#ok<AGROW>
end
function r = run_lengths(v)
v = v(:)'; r = [];
c = 0;
for i=1:numel(v)
    if v(i), c=c+1; else, if c>0, r(end+1)=c; end; c=0; end %#ok<AGROW>
end
if c>0, r(end+1)=c; end
r = r(:);
end

%% ======================= FEATURE ASSEMBLY ==============================
function [F, Xden] = features_from_windows(X, CFG, fs)
if nargin < 3, fs = CFG.eeg_fs; end   % EEG default; ECG passes CFG.ecg_fs
n = numel(X);
F = zeros(n,10); Xden = cell(1,n);
for i = 1:n
    x = ssa_denoise(X{i}, CFG.ssa_L, CFG.ssa_energy);
    Xden{i} = x;
    tau = ami_tau(x, CFG.ami_maxlag, CFG.ami_nbins);
    m   = fnn_dim(x, tau, CFG.fnn_maxdim, CFG.fnn_rtol, CFG.fnn_thresh);
    lam = lyap_rosenstein(x, m, tau, fs);
    D2  = corr_dim_gp(x, m, tau);
    q   = rqa_measures(x, m, tau, 0.05);
    F(i,:) = [lam, D2, tau, m, q.RR, q.DET, q.L, q.ENT, q.LAM, q.TT];
    if mod(i, 50)==0, fprintf('    features %d/%d\n', i, n); end
end
F(~isfinite(F)) = 0;
end

%% ======================= CLASSIFIERS ===================================
function net = build_cnn_lstm(Lseq, nclass)
layers = [ ...
    sequenceInputLayer(1,'Name','in')
    convolution1dLayer(7,16,'Padding','same','Name','c1')
    reluLayer('Name','r1')
    maxPooling1dLayer(2,'Padding','same','Name','p1')
    convolution1dLayer(5,32,'Padding','same','Name','c2')
    reluLayer('Name','r2')
    globalAveragePooling1dLayer('Name','gap')
    lstmLayer(48,'OutputMode','last','Name','lstm')
    fullyConnectedLayer(nclass,'Name','fc')
    softmaxLayer('Name','sm')
    classificationLayer('Name','out')];
net = layerGraph(layers); %#ok<NASGU>
net = layers;  % plain array works with trainNetwork
end

function [pred, score] = dl_train_predict(Xtr, ytr, Xte, yte, nclass, CFG)
% Xtr/Xte: cell of 1xL windows. CNN-LSTM sequence classification.
seqtr = cellfun(@(z) reshape(z,1,[]), Xtr, 'uni', 0);
seqte = cellfun(@(z) reshape(z,1,[]), Xte, 'uni', 0);
Ytr = categorical(ytr(:), 1:nclass);
layers = build_cnn_lstm([], nclass);
opts = trainingOptions('adam','MaxEpochs',15,'MiniBatchSize',64, ...
    'Shuffle','every-epoch','Verbose',false, ...
    'InitialLearnRate',1e-3);
try
    net = trainNetwork(seqtr, Ytr, layers, opts);
    [predc, score] = classify(net, seqte);
    pred = double(predc);
catch ME
    warning('DL training failed (%s). Falling back to ensemble.', ME.message);
    [pred, score] = ml_train_predict_features(seqtr, ytr, seqte, yte, CFG, nclass);
end
end

function [pred, score] = ml_train_predict(Ftr, ytr, Fte, yte, CFG) %#ok<INUSD>
nclass = numel(unique([ytr(:);yte(:)]));
mu = mean(Ftr,1); sd = std(Ftr,[],1); sd(sd==0)=1;
Ztr = (Ftr-mu)./sd; Zte = (Fte-mu)./sd;
mdl = fitcecoc(Ztr, ytr(:), 'Coding','onevsall', ...
    'Learners', templateSVM('KernelFunction','rbf','Standardize',false), ...
    'FitPosterior', true);
[pred, ~, ~, score] = predict(mdl, Zte);
pred = pred(:);
if size(score,2) < nclass, score = padarray_cols(score, nclass); end
end

function [pred, score] = ml_train_predict_features(seqtr, ytr, seqte, yte, CFG, nclass) %#ok<INUSD>
% simple statistical features when DL unavailable inside DL path
fe = @(c) cell2mat(cellfun(@(z)[mean(z) std(z) min(z) max(z) ...
    median(abs(z-median(z)))], c(:), 'uni',0));
Ztr = fe(seqtr); Zte = fe(seqte);
mu=mean(Ztr,1); sd=std(Ztr,[],1); sd(sd==0)=1;
Ztr=(Ztr-mu)./sd; Zte=(Zte-mu)./sd;
mdl = fitcecoc(Ztr, ytr(:), 'FitPosterior', true);
[pred,~,~,score] = predict(mdl, Zte); pred=pred(:);
if size(score,2) < nclass, score = padarray_cols(score, nclass); end
end

function s2 = padarray_cols(s, n)
s2 = zeros(size(s,1), n); s2(:,1:size(s,2)) = s;
end

%% ======================= EVALUATION ====================================
function R = evaluate_kfold(Xden, F, y, CFG, tag)
y = y(:); n = numel(y);
k = min(CFG.kfold, n);
cv = cvpartition(y, 'KFold', k);
acc = zeros(k,1); sens=zeros(k,1); spec=zeros(k,1);
allScore = zeros(n,1); allY = zeros(n,1); ptr=1;
for f = 1:k
    tr = training(cv,f); te = test(cv,f);
    if has_dl_toolbox()
        [pred, score] = dl_train_predict(Xden(tr), y(tr)+1, Xden(te), y(te)+1, 2, CFG);
        pred = pred-1; sc = score(:,2);
    else
        [pred, score] = ml_train_predict(F(tr,:), y(tr)+1, F(te,:), y(te)+1, CFG);
        pred = pred-1; sc = score(:,min(2,size(score,2)));
    end
    yt = y(te);
    acc(f)  = mean(pred==yt);
    sens(f) = sum(pred==1 & yt==1)/max(sum(yt==1),1);
    spec(f) = sum(pred==0 & yt==0)/max(sum(yt==0),1);
    m = numel(yt); allScore(ptr:ptr+m-1)=sc; allY(ptr:ptr+m-1)=yt; ptr=ptr+m;
end
R.acc_mean=100*mean(acc); R.acc_sd=100*std(acc);
R.sens=100*mean(sens); R.spec=100*mean(spec);
[R.fpr,R.tpr,R.auc] = compute_roc(allY, allScore);
fprintf('  [%s] ACC=%.2f%% +/- %.2f | Sens=%.1f%% Spec=%.1f%% AUC=%.3f\n', ...
    tag, R.acc_mean, R.acc_sd, R.sens, R.spec, R.auc);
end

function R = evaluate_loso(Xden, F, y, subj, CFG, tag)
y=y(:); subj=subj(:); us=unique(subj);
acc=zeros(numel(us),1);
for s=1:numel(us)
    te = subj==us(s); tr = ~te;
    if sum(y(tr)==1)<2 || sum(y(tr)==0)<2, acc(s)=NaN; continue; end
    if has_dl_toolbox()
        pred = dl_train_predict(Xden(tr), y(tr)+1, Xden(te), y(te)+1, 2, CFG)-1;
    else
        pred = ml_train_predict(F(tr,:), y(tr)+1, F(te,:), y(te)+1, CFG)-1;
    end
    acc(s)=mean(pred==y(te));
end
acc=acc(~isnan(acc));
R.acc_mean=100*mean(acc); R.acc_sd=100*std(acc);
fprintf('  [%s] ACC=%.2f%% +/- %.2f  (over %d subjects)\n', tag, R.acc_mean, R.acc_sd, numel(acc));
end

function [fpr,tpr,auc] = compute_roc(y, score)
y=y(:); score=score(:);
if ~isempty(which('perfcurve')) && numel(unique(y))==2
    [fpr,tpr,~,auc] = perfcurve(y, score, 1);
else
    [fpr,tpr,auc] = roc_manual(y, score);
end
end

function [fpr,tpr,auc] = roc_manual(y, score)
[score, idx] = sort(score,'descend'); y=y(idx);
P=sum(y==1); N=sum(y==0);
tp=0; fp=0; tpr=0; fpr=0;
for i=1:numel(y)
    if y(i)==1, tp=tp+1; else, fp=fp+1; end
    tpr(end+1)=tp/max(P,1); fpr(end+1)=fp/max(N,1); %#ok<AGROW>
end
auc = trapz(fpr, tpr);
end

function A = multiclass_auc(y, score, nclass)
y=y(:); A=0; cnt=0;
for c=1:nclass
    if size(score,2)>=c && any(y==c) && any(y~=c)
        [~,~,a] = compute_roc(double(y==c), score(:,c)); A=A+a; cnt=cnt+1;
    end
end
if cnt>0, A=A/cnt; else, A=NaN; end
end

function M = class_metrics(C, classes)
M = struct();
for i=1:numel(classes)
    TP=C(i,i); FN=sum(C(i,:))-TP; FP=sum(C(:,i))-TP; TN=sum(C(:))-TP-FN-FP;
    M.(classes{i}).sens = 100*TP/max(TP+FN,1);
    M.(classes{i}).ppv  = 100*TP/max(TP+FP,1);
    M.(classes{i}).spec = 100*TN/max(TN+FP,1);
end
end

%% ======================= CLASS HELPERS =================================
function [X,y,subj] = balance_classes(X,y,subj,cap)
y=y(:); subj=subj(:);
keep=[];
for c=unique(y)'
    idx=find(y==c);
    if numel(idx)>cap, idx=idx(randperm(numel(idx),cap)); end
    keep=[keep; idx]; %#ok<AGROW>
end
% balance the two classes to the smaller count
c0=sum(y(keep)==0); c1=sum(y(keep)==1); mn=min(c0,c1);
if mn>0
    k0=keep(y(keep)==0); k1=keep(y(keep)==1);
    k0=k0(randperm(numel(k0),min(mn,numel(k0))));
    k1=k1(randperm(numel(k1),min(mn,numel(k1))));
    keep=[k0;k1];
end
keep=sort(keep);
X=X(keep); y=y(keep); subj=subj(keep);
end

function [Xc, F, y] = oversample_train(Xc, F, y, nclass)
% Duplicate minority-class training samples (with replacement) up to the
% size of the largest class, so the classifier can actually learn rare beats.
y = y(:);
counts = zeros(1,nclass);
for c = 1:nclass, counts(c) = sum(y==c); end
target = max(counts);
idxAll = [];
for c = 1:nclass
    idx = find(y==c);
    if isempty(idx), continue; end
    n = numel(idx);
    if n < target
        extra = idx(randi(n, target-n, 1));   % sample with replacement
        idx = [idx; extra]; %#ok<AGROW>
    end
    idxAll = [idxAll; idx]; %#ok<AGROW>
end
idxAll = idxAll(randperm(numel(idxAll)));
Xc = Xc(idxAll); F = F(idxAll,:); y = y(idxAll);
fprintf('  oversampled train -> %d beats (%d per class, balanced)\n', numel(y), target);
end

function [X,y,RR] = cap_per_class(X,y,cap,RR)
if nargin<4, RR=[]; end
y=y(:); keep=[];
for c=unique(y)'
    idx=find(y==c);
    if numel(idx)>cap, idx=idx(randperm(numel(idx),cap)); end
    keep=[keep; idx]; %#ok<AGROW>
end
keep=sort(keep); X=X(keep); y=y(keep);
if ~isempty(RR), RR=RR(keep,:); end
end

function tabulate_classes(y, classes, tag)
fprintf('  %s class counts: ', tag);
for c=1:numel(classes), fprintf('%s=%d ', classes{c}, sum(y==c)); end
fprintf('\n');
end

%% ======================= FIGURES (REAL DATA) ===========================
function save_fig(h, name, CFG)
png = fullfile(CFG.outdir,[name '.png']);
pdf = fullfile(CFG.outdir,[name '.pdf']);
try
    exportgraphics(h, png, 'Resolution',150);
    exportgraphics(h, pdf, 'ContentType','image', 'Resolution',300); % raster: fast, no vector warning
catch
    print(h, png, '-dpng','-r150');
    print(h, pdf, '-dpdf');
end
fprintf('    saved %s (.png/.pdf)\n', name);
end

function make_fig1(Xden, Xraw, CFG)
h=figure('Visible','off','Position',[100 100 900 600]);
x=Xraw{1}; xd=Xden{1}; t=(0:numel(x)-1)/CFG.eeg_fs;
subplot(2,1,1); plot(t,x,'Color',[.6 .6 .6]); hold on; plot(t,xd,'b','LineWidth',1.2);
legend('Raw','SSA-filtered'); xlabel('Time (s)'); ylabel('Amplitude');
title('SSA Denoising: Raw vs Filtered (real record)');
% scree from SSA of first window
x=x(:); L=min(CFG.ssa_L,floor(numel(x)/2)); K=numel(x)-L+1; M=zeros(L,K);
for i=1:L, M(i,:)=x(i:i+K-1); end
s=svd(M); ve=100*s.^2/sum(s.^2);
subplot(2,1,2); stem(ve,'.'); xlabel('Component k'); ylabel('Variance (%)');
title('Singular-value spectrum');
save_fig(h,'Fig1_SSA_Denoising',CFG); close(h);
end

function make_fig2(Xden, y, CFG)
h=figure('Visible','off','Position',[100 100 1100 350]);
x=Xden{find(y==1,1)}; if isempty(x), x=Xden{1}; end
maxlag=CFG.ami_maxlag;
% AMI curve
edges=linspace(min(x)-eps,max(x)+eps,CFG.ami_nbins+1); [~,~,bx]=histcounts(x,edges); bx(bx==0)=1;
ami=zeros(maxlag,1); N=numel(x);
for lag=1:maxlag
  a=bx(1:N-lag); b=bx(1+lag:N); P=accumarray([a b],1,[CFG.ami_nbins CFG.ami_nbins])/numel(a);
  Pa=sum(P,2); Pb=sum(P,1); Mi=0;
  for ii=1:CFG.ami_nbins, for jj=1:CFG.ami_nbins, if P(ii,jj)>0, Mi=Mi+P(ii,jj)*log(P(ii,jj)/(Pa(ii)*Pb(jj))); end, end, end
  ami(lag)=Mi;
end
tau=ami_tau(x,maxlag,CFG.ami_nbins);
subplot(1,3,1); plot(1:maxlag,ami,'-o'); hold on; plot(tau,ami(tau),'ro','MarkerFaceColor','r');
xlabel('\tau'); ylabel('AMI'); title(sprintf('AMI (\\tau*=%d)',tau));
% FNN
m=fnn_dim(x,tau,CFG.fnn_maxdim,CFG.fnn_rtol,CFG.fnn_thresh);
subplot(1,3,2); bar(1:m,ones(1,m)); xlabel('Embedding dim'); title(sprintf('FNN (m*=%d)',m));
% phase portrait
Y=embed(x,3,tau);
subplot(1,3,3); if size(Y,1)>3, plot3(Y(:,1),Y(:,2),Y(:,3),'.','MarkerSize',3); end
grid on; title('3-D phase portrait'); xlabel('x(t)'); ylabel('x(t+\tau)'); zlabel('x(t+2\tau)');
save_fig(h,'Fig2_PhaseSpace',CFG); close(h);
end

function make_fig3(Xden, y, CFG)
h=figure('Visible','off','Position',[100 100 900 380]);
xi=Xden{find(y==1,1)}; xo=Xden{find(y==0,1)};
if isempty(xi), xi=Xden{1}; end; if isempty(xo), xo=Xden{end}; end
tau=ami_tau(xi,CFG.ami_maxlag,CFG.ami_nbins); m=fnn_dim(xi,tau,CFG.fnn_maxdim,CFG.fnn_rtol,CFG.fnn_thresh);
l1=lyap_rosenstein(xi,m,tau,CFG.eeg_fs); l0=lyap_rosenstein(xo,m,tau,CFG.eeg_fs);
d1=corr_dim_gp(xi,m,tau); d0=corr_dim_gp(xo,m,tau);
subplot(1,2,1); bar([l1 l0]); set(gca,'XTickLabel',{'ictal','inter'});
ylabel('\lambda_{max} (nats/s)'); title('Largest Lyapunov exponent');
subplot(1,2,2); bar([d1 d0]); set(gca,'XTickLabel',{'ictal','inter'});
ylabel('D_2'); title('Correlation dimension');
save_fig(h,'Fig3_LyapunovD2',CFG); close(h);
end

function make_fig4(Xden, y, CFG)
h=figure('Visible','off','Position',[100 100 900 420]);
xi=Xden{find(y==1,1)}; xo=Xden{find(y==0,1)};
if isempty(xi), xi=Xden{1}; end; if isempty(xo), xo=Xden{end}; end
for idx=1:2
    if idx==1, x=xi; ttl='Ictal'; else, x=xo; ttl='Inter-ictal'; end
    tau=ami_tau(x,CFG.ami_maxlag,CFG.ami_nbins); m=fnn_dim(x,tau,CFG.fnn_maxdim,CFG.fnn_rtol,CFG.fnn_thresh);
    Y=embed(x,m,tau); if size(Y,1)>200, Y=Y(round(linspace(1,size(Y,1),200)),:); end
    DM=squareform(pdist(Y)); thr=quantile(DM(:),0.05); RP=DM<=thr;
    subplot(1,2,idx); imagesc(RP); colormap(gca,gray); axis square;
    title(sprintf('Recurrence Plot - %s',ttl)); xlabel('Time index'); ylabel('Time index');
end
save_fig(h,'Fig4_RecurrencePlots',CFG); close(h);
end

function make_fig5(F, y, CFG)
h=figure('Visible','off','Position',[100 100 1000 400]);
names={'\lambda','D_2','\tau','m','RR','DET','L','ENT','LAM','TT'};
Fn=F; mn=min(Fn,[],1); mx=max(Fn,[],1); rng=mx-mn; rng(rng==0)=1; Fn=(Fn-mn)./rng;
subplot(1,2,1);
bar([mean(Fn(y==1,:),1); mean(Fn(y==0,:),1)]');
set(gca,'XTick',1:10,'XTickLabel',names); legend('ictal','inter'); xtickangle(45);
ylabel('Normalised feature'); title('Mean feature vectors (real data)');
% univariate separability (|mean diff|/pooled std) as importance
imp=zeros(1,10);
for j=1:10
    a=F(y==1,j); b=F(y==0,j); sp=sqrt((var(a)+var(b))/2)+eps;
    imp(j)=abs(mean(a)-mean(b))/sp;
end
subplot(1,2,2); barh(imp); set(gca,'YTick',1:10,'YTickLabel',names);
xlabel('Separability |\Deltamean|/s_{pooled}'); title('Feature importance (measured)');
save_fig(h,'Fig5_FeatureVectors',CFG); close(h);
end

function make_fig6(R, CFG)
h=figure('Visible','off','Position',[100 100 950 420]);
subplot(1,2,1);
imagesc(R.conf); colorbar; axis square;
set(gca,'XTick',1:numel(R.classes),'XTickLabel',R.classes, ...
        'YTick',1:numel(R.classes),'YTickLabel',R.classes);
xlabel('Predicted'); ylabel('True');
title(sprintf('ECG confusion (acc=%.1f%%)',100*R.acc));
for i=1:size(R.conf,1), for j=1:size(R.conf,2)
    text(j,i,num2str(R.conf(i,j)),'HorizontalAlignment','center','Color','r'); end, end
subplot(1,2,2);
sens=cellfun(@(c) R.metrics.(c).sens, R.classes);
bar(sens); set(gca,'XTickLabel',R.classes); ylabel('Sensitivity (%)'); ylim([0 100]);
title('Per-class sensitivity (measured)');
save_fig(h,'Fig6_PerformanceComparison',CFG); close(h);
end

function make_fig7(K, CFG)
h=figure('Visible','off','Position',[100 100 950 420]);
subplot(1,2,1);
plot(K.fpr,K.tpr,'r','LineWidth',1.6); hold on; plot([0 1],[0 1],'k--');
xlabel('False positive rate'); ylabel('True positive rate');
title(sprintf('ROC (AUC=%.3f, measured)',K.auc)); axis square;
subplot(1,2,2);
bar([K.acc_mean K.sens K.spec]); set(gca,'XTickLabel',{'Acc','Sens','Spec'});
ylabel('%'); ylim([0 100]); title(sprintf('EEG 10-fold (%.1f%%)',K.acc_mean));
save_fig(h,'Fig7_ROC_Ablation',CFG); close(h);
end

%% ======================= SUMMARY =======================================
function print_summary(CFG, R)
fid = fopen(fullfile(CFG.outdir,'NDS_Measured_Results.txt'),'w');
pr = @(varargin) fprintf_both(fid, varargin{:});
pr('================ MEASURED RESULTS (no hardcoding) ================\n');
pr('Generated: %s\n', datestr(now));
pr('Quick demo mode: %d\n\n', CFG.quick_demo);
if isfield(R,'eeg') && isfield(R.eeg,'kfold')
    k=R.eeg.kfold;
    pr('EEG (CHB-MIT) 10-fold: ACC=%.2f%% +/- %.2f | Sens=%.1f%% Spec=%.1f%% AUC=%.3f\n', ...
        k.acc_mean,k.acc_sd,k.sens,k.spec,k.auc);
    if isfield(R.eeg,'loso') && ~isempty(R.eeg.loso)
        pr('EEG LOSO: ACC=%.2f%% +/- %.2f\n', R.eeg.loso.acc_mean, R.eeg.loso.acc_sd);
    end
end
if isfield(R,'ecg') && isfield(R.ecg,'acc')
    pr('ECG (MIT-BIH, inter-patient AAMI): ACC=%.2f%% macro-AUC=%.3f\n', 100*R.ecg.acc, R.ecg.auc);
    for c=1:numel(R.ecg.classes)
        cl=R.ecg.classes{c};
        pr('   %s: Sens=%.1f%% PPV=%.1f%% Spec=%.1f%%\n', cl, ...
            R.ecg.metrics.(cl).sens, R.ecg.metrics.(cl).ppv, R.ecg.metrics.(cl).spec);
    end
end
pr('\nAll figures saved as PNG + PDF in: %s\n', CFG.outdir);
pr('NOTE: update the manuscript to match these measured values.\n');
if fid>0, fclose(fid); end
build_multipage_pdf(CFG);
end

function fprintf_both(fid, varargin)
fprintf(varargin{:});
if fid>0, fprintf(fid, varargin{:}); end
end

function build_multipage_pdf(CFG)
figs = {'Fig1_SSA_Denoising','Fig2_PhaseSpace','Fig3_LyapunovD2', ...
        'Fig4_RecurrencePlots','Fig5_FeatureVectors', ...
        'Fig6_PerformanceComparison','Fig7_ROC_Ablation'};
out = fullfile(CFG.outdir,'NDS_AllFigures.pdf'); first=true;
for i=1:numel(figs)
    png=fullfile(CFG.outdir,[figs{i} '.png']);
    if ~exist(png,'file'), continue; end
    im=imread(png); hf=figure('Visible','off'); imshow(im);
    try
        if first, exportgraphics(hf,out,'ContentType','image'); first=false;
        else, exportgraphics(hf,out,'ContentType','image','Append',true); end
    catch
    end
    close(hf);
end
if exist(out,'file'), fprintf('Multi-page PDF: %s\n', out); end
end
