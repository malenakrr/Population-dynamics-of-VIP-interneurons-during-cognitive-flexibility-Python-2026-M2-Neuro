# Population dynamics of cortical interneurons during cognitive flexibility 
# Python 2026 - M2 Neuro
This repository contains the computational workflow used to analyze cranial window miniscope recordings in mice undergoing a rule-reversal behavioral paradigm.
The goal of this project is to investigate whether cortical interneurons display specific, task-modulated activity patterns (such as differential responses to rewarded versus unrewarded nose pokes) during behavioral flexibility events and whether they act as a coordinated ensemble.

## In this GitHub, you can find the codes used for the data processing and analysis of :

### Data processing and fluorescence traces extraction
- Fluorescence_extraction :

### Data correlation with behavior
All of these codes give the same outputs : PETH heatmap during the session, Population event-triggered average before and after nosepoke, Comparison of activity before and after nosepoke, Correlation matrix (non-clustered and clustered), Global activity during the session, Comparaison of activity between reward and non-reward.
- Baseline_Behavior_Correlation : correlates the timestamps of calcium peaks and nosepokes during the training phase
- RR_Behavior_Correlation : correlates the timestamps of calcium peaks and nosepokes during the rule reversal phase
- RS_Behavior_Correlation : correlates the timestamps of calcium peaks and nosepokes during the rule shift phase
