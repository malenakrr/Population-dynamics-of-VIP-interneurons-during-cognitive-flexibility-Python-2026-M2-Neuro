# Population dynamics of cortical interneurons during cognitive flexibility 
# Python 2026 - M2 Neuro
This repository contains the computational workflow used to analyze cranial window miniscope recordings in mice undergoing a rule-reversal behavioral paradigm.
The goal of this project is to investigate whether cortical interneurons display specific, task-modulated activity patterns (such as differential responses to rewarded versus unrewarded nose pokes) during behavioral flexibility events and whether they act as a coordinated ensemble.
This repository was developed by Florian Plumel, Léonie Raveniaud and Malena Kirchacker during the Python Applications for
Neuroscience Research course, Université Paris Cité, 2025. © 2025 The contributors.
Released under the MIT License.

## In this GitHub, you can find the codes used for the data processing and analysis of :

### Data processing and fluorescence traces extraction
- NDNF_Fluo_Extraction : Implements a full processing pipeline for 1-photon calcium imaging data, uuses the CaImAn library to perform motion correction, source extraction and quality evaluation. It is based on a pre-existing pipeline, but has been modified to include additional features (produce specific outputs), and optimizes the parameters for 1-photon miniscope data. Reference : A. Giovannucci et al., (2019), CaImAn an open source tool for scalable calcium imaging data analysis, eLife

### Data correlation with behavior
All of these codes give the same outputs : PETH heatmap during the session, Population event-triggered average before and after nosepoke, Comparison of activity before and after nosepoke, Correlation matrix (non-clustered and clustered), Global activity during the session, Comparaison of activity between reward and non-reward.
- Baseline_Behavior_Correlation : correlates the timestamps of calcium peaks and nosepokes during the training phase
- RR_Behavior_Correlation : correlates the timestamps of calcium peaks and nosepokes during the rule reversal phase
- RS_Behavior_Correlation : correlates the timestamps of calcium peaks and nosepokes during the rule shift phase

### Population analysis
NDNF_Population_Analysis : 
