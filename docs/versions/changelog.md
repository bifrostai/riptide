## 0.4.0 <small>2023-05-30</small>
- Display class names when available<br>
- Refinements to error highlighting:<br>
    - In addition to grouping detections by perceptual similarity, detections for the same target are now grouped together as well.<br>
    - Badges are now displayed as `U|T` to denote the number of <u>u</u>nique targets and <u>t</u>otal number of detections within a group<br>
    - High confidence detections (`confidence > 0.85`) are extracted for easier identification of most serious failures<br>
- Generate placeholder images when images cannot be loaded<br>
- Bounding boxes are now drawn after cropping<br>
    - Stroke widths are now consistent across crops<br>
- Behind the scenes:<br>
    - Implemented various tests to ensure correctness of analysis and visualizations.

## 0.3.0 <small>2023-05-15</small>
- Revamp _Overview_ section. Collate error counts into colored bars.<br>
- Use HDBSCAN instead of DBSCAN to cluster crop visualizations.<br>
- Group visualizations within clusters by sub-clusters.<br>
- Combine _Classification Errors_ and _Classification And Localization Errors_ into a single section (_Confusions_).<br>
- Update bounding box size visualization in _Missed Errors_ section.<br>
    - Use a violin plot instead of a box plot.<br>
    - Use a logarithmic scale for the y-axis.<br>
    - Add jitter to the data points.<br>
- Group missed errors by likely causes.<br>
- Standardize colors of bounding boxes for each error type.<br>
- Define new comparison report to compare the performance of two models.

## 0.2.0 <small>2023-04-06</small>
- Added visualizations for LOC, CLL, DUP errors.<br>
- Defined new module to visualize changes in ground truth predictions between modules.<br>
- Image samples are now ordered by visual similarity. Identify commonalities between failing cases more easily.

## 0.1.0 <small>2023-03-10</small>
- Initial release
