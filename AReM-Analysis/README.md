# Human Activity Classification with Time Series Data

This repository contains work on classifying human activities based on time series data obtained from a Wireless Sensor Network (WSN). The project leverages the AReM dataset, focusing on feature extraction and machine learning techniques for time series classification. 

An interesting task in the machine learning course DSCI552, I wrap up my homework here. 

## 📘 Project Overview

The objective is to classify seven types of human activity based on the AReM dataset. The dataset includes time series data collected from multiple sensors, where each instance represents a human performing an activity.

### Dataset: AReM
The AReM dataset, available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/label+Recognition+system+based+on+Multisensor+data+fusion+\%28AReM\%29), consists of:
- **7 Activity Labels**: Each folder represents a distinct activity.
- **Instances**: Each folder contains multiple files representing individual instances of human activity.
- **Time Series Variables**: Each file contains six time series measurements (`avg_rss12`, `var_rss12`, `avg_rss13`, `var_rss13`, `vg_rss23`, and `ar_rss23`), each capturing 480 consecutive values.

### Some Refelction...
While larger datasets can enhance prediction accuracy, this alone may not improve understanding of the complex, underlying social phenomena.
