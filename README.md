# Population dynamics of cortical interneurons during cognitive flexibility 
# Python 2026 - M2 Neuro
This repository contains the computational workflow used to analyze miniscope recordings in mice undergoing a rule-reversal behavioral paradigm.
The goal of this project is to investigate whether NDNF interneurons display specific, task-modulated activity patterns (such as differential responses to rewarded versus unrewarded nose pokes) during behavioral flexibility events and whether they act as a coordinated ensemble.
This repository was developed by Florian Plumel, Léonie Raveniaud and Malena Kirchacker during the Python Applications for
Neuroscience Research course, Université Paris Cité, 2025. © 2025 The contributors.
Released under the MIT License.

## In this GitHub, you can find the codes used for the data processing and analysis of :

### Data processing and fluorescence traces extraction
- NDNF_Fluo_Extraction : Implements a full processing pipeline for 1-photon calcium imaging data, uuses the CaImAn library to perform motion correction, source extraction and quality evaluation. It is based on a pre-existing pipeline, but has been modified to include additional features (produce specific outputs), and optimizes the parameters for 1-photon miniscope data. Reference : A. Giovannucci et al., (2019), CaImAn an open source tool for scalable calcium imaging data analysis, eLife

### GUI application to synchronize animal behavioral video recordings with calcium imaging data
- nosepoke_to_calcium_frame.py is a Tkinter-based Python GUI application designed to synchronize animal behavioral video recordings with calcium imaging data. The tool allows users to manually annotate specific behavioral events, such as "reward" or "no reward" nose pokes, while automatically computing the exact corresponding calcium imaging frame using customizable temporal offsets and row shifts. It features interactive video playback, timestamp alignment verification, and automated data export to Excel or CSV to seamlessly integrate into downstream data processing pipelines.

### Data correlation with behavior
All of these codes give the same outputs : PETH heatmap during the session, Population event-triggered average before and after nosepoke, Comparison of activity before and after nosepoke, Correlation matrix (non-clustered and clustered), Global activity during the session, Comparaison of activity between reward and non-reward.
- Baseline_Behavior_Correlation : correlates the timestamps of calcium peaks and nosepokes during the training phase
- RR_Behavior_Correlation : correlates the timestamps of calcium peaks and nosepokes during the rule reversal phase
- RS_Behavior_Correlation : correlates the timestamps of calcium peaks and nosepokes during the rule shift phase

### Population analysis
NDNF_Population_Analysis : Run after the CaImAn pipeline. This script measures how strongly calcium activity is correlated between every pair of neurons in a miniscope recording. It reads traces_and_rois.npz, computes Pearson correlations between ROI ΔF/F traces, and tests each pair against a circular-shift null with FDR correction. Cells are ordered by hierarchical clustering, and correlations are related to the distance between cells.

