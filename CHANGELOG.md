## v0.3.0 (2023-05-15)
- Revamp _Overview_ section. Collate error counts into colored bars.
- Use HDBSCAN instead of DBSCAN to cluster crop visualizations.
- Group visualizations within clusters by sub-clusters.
- Combine _Classification Errors_ and _Classification And Localization Errors_ into a single section (_Confusions_).
- Update bounding box size visualization in _Missed Errors_ section. <br/>
    - Use a violin plot instead of a box plot.
    - Use a logarithmic scale for the y-axis.
    - Add jitter to the data points.
- Group missed errors by likely causes.
- Standardize colors of bounding boxes for each error type.
- Define new comparison report to compare the performance of two models.


## v0.2.0 (2023-04-06)
- Defined new module (`gt-flow`) to visualize changes in ground truth predictions between modules.
- Order image samples by visual similarity, copmuted using DBSCAN. Identify commonalities between failing cases more easily.
