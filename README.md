# music-sync-internship-upf

Python port of the *onsetsync* R package for analysis and visualization of synchrony in musical performances performances.

Eerola, T. & Clayton, M. (2022). onsetsync - Analysis and Visualisation of Synchronisation of Music Onset Data. https://github.com/tuomaseerola/onsetsync

## Table of Functions

Original function reference can be accessed by clicking on the function

| No. | Status                                   | Function                                                 |
|-----|------------------------------------------|----------------------------------------------------------|
|     | **Input**                                |                                                          |
| 1   | get_df_csv()                             | [get_OSF_csv()](https://tuomaseerola.github.io/onsetsync/reference/get_OSF_csv.html) |
|     | **Annotation**                           |                                                          |
| 2   | done                                     | [add_annotation()](https://tuomaseerola.github.io/onsetsync/reference/add_annotation.html) |
| 3   | done                                     | [add_isobeats()](https://tuomaseerola.github.io/onsetsync/reference/add_isobeats.html) |
|     | **Visualising**                          |                                                          |
| 4   | done* (split into two functions)         | [plot_by_beat() / boxplot_by_beat()](https://tuomaseerola.github.io/onsetsync/reference/plot_by_beat.html) |
| 5   | -                                        | [plot_by_dataset()](https://tuomaseerola.github.io/onsetsync/reference/plot_by_dataset.html) |
| 6   | done                                     | [plot_by_pair()](https://tuomaseerola.github.io/onsetsync/reference/plot_by_pair.html) |
| 7   |                                          | [plot_by_var_time()](https://tuomaseerola.github.io/onsetsync/reference/plot_by_var_time.html) |
| 8   |                                          | [plot_by_variable()](https://tuomaseerola.github.io/onsetsync/reference/plot_by_variable.html) |
| 9   |                                          | [plot_timeline()](https://tuomaseerola.github.io/onsetsync/reference/plot_timeline.html) |
|     | **Synchrony**                            |                                                          |
| 10  | done                                     | [sync_execute_pairs()](https://tuomaseerola.github.io/onsetsync/reference/sync_execute_pairs.html) |
| 11  | done                                     | [sync_joint_onsets()](https://tuomaseerola.github.io/onsetsync/reference/sync_joint_onsets.html) |
| 12  | done                                     | [sync_sample_paired()](https://tuomaseerola.github.io/onsetsync/reference/sync_sample_paired.html) |
| 13  | done                                     | [sync_sample_paired_relative()](https://tuomaseerola.github.io/onsetsync/reference/sync_sample_paired_relative.html) |
|     | **Periodicity**                          |                                                          |
| 14  |                                          | [period_to_BPM()](https://tuomaseerola.github.io/onsetsync/reference/period_to_BPM.html) |
| 15  |                                          | [periodicity()](https://tuomaseerola.github.io/onsetsync/reference/periodicity.html) |
| 16  |                                          | [periodicity_nPVI()](https://tuomaseerola.github.io/onsetsync/reference/periodicity_nPVI.html) |
| 17  |                                          | [gaussify_onsets()](https://tuomaseerola.github.io/onsetsync/reference/gaussify_onsets.html) |
|     | **Summary**                              |                                                          |
| 18  | done                                     | [summarise_onsets()](https://tuomaseerola.github.io/onsetsync/reference/summarise_onsets.html) |
| 19  |                                          | [summarise_periodicity()](https://tuomaseerola.github.io/onsetsync/reference/summarise_periodicity.html) |
| 20  | done                                     | [summarise_sync()](https://tuomaseerola.github.io/onsetsync/reference/summarise_sync.html) |
| 21  | done                                     | [summarise_sync_by_pair()](https://tuomaseerola.github.io/onsetsync/reference/summarise_sync_by_pair.html) |
|     | **Other**                                |                                                          |
| 22  |                                          | [synthesise_onsets()](https://tuomaseerola.github.io/onsetsync/reference/synthesise_onsets.html) |
