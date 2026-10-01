# Label mapping (original Vietnamese -> English)

| Original | English |
|---|---|
| EAGB (HCE đồng thời) | EAGB (simultaneous HCE) |
| GBDT + HCE tuần tự (GT2) | GBDT + sequential HCE (GT2) |
| GBDT + mã hóa phẳng co ngót (H1c) | GBDT + flat shrinkage encoding (H1c) |
| GBDT + mã hóa cấp quận (lùi về nút cha) | GBDT + county-level encoding (fallback to parent) |
| GBDT phân loại nội tại (bang/quận/tract) | GBDT native categorical (state/county/tract) |
| GBDT không biến địa lý | GBDT without geographic variables |
| EAGB-PG tần suất | EAGB-PG frequency |
| Tín nhiệm phân cấp thuần (HCE) | Pure hierarchical credibility (HCE) |
| GLM Tweedie (đối chứng chính) | Tweedie GLM (primary baseline) |
| GLM Tweedie + spline (xấp xỉ GAM) | Tweedie GLM + spline (GAM approximation) |
| CatBoost OTS (có thứ tự thời gian) | CatBoost OTS (time-ordered) |
| Mô hình rỗng | Null model |
