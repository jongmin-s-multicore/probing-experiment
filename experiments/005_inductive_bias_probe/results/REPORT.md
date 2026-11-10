# REPORT (자동 생성: 규칙 적용 결과와 수치만, 해석 없음)

판정 규칙 적용 결과 요약: pred_out6: H-e=not_evaluable H-f=supported H-g=refuted

- pilot: True
- 4.6 분기 행: ['no_vanishing'] ; hold: False ; unresolved: []

| layer | 하위 실험 | H-e | H-f | H-g |
|---|---|---|---|---|
| pred_out6 | 5-A,5-B,5-C,5-D | not_evaluable (area_baseline_saturated) | supported () | refuted () |

## 근거 수치 (results.json 의 verdicts 에서)

```json
[
 {
  "layer": "pred_out6",
  "sub_experiments": [
   "5-A",
   "5-B",
   "5-C",
   "5-D"
  ],
  "H-e": {
   "status": "not_evaluable",
   "reason": "area_baseline_saturated",
   "metric": "D-IB",
   "area_score": 1.0690844647054114
  },
  "H-f": {
   "status": "supported",
   "D-IB_area_eq": 0.9278421751577877,
   "D-IB_area_neq": 1.015019751816809,
   "sd_eq": 0.04524816035891906,
   "sd_neq": 0.030581949817887744,
   "abs_diff": 0.08717757665902126,
   "pair_group": "all"
  },
  "H-g": {
   "status": "refuted",
   "per_slice": {
    "early": {
     "slope_mean": 0.6065553619020202,
     "slope_sd": 0.057353325801782935,
     "n_seeds": 5,
     "within": false
    },
    "extrap": {
     "slope_mean": 0.5432237753217674,
     "slope_sd": 0.06039141033620929,
     "n_seeds": 5,
     "within": false
    },
    "interp": {
     "slope_mean": 0.9355640544843145,
     "slope_sd": 0.08727196493664931,
     "n_seeds": 5,
     "within": true
    },
    "labels_a060+a160": {
     "slope_mean": 0.5408659225116843,
     "slope_sd": 0.005602139265507944,
     "n_seeds": 5,
     "within": false
    },
    "labels_a060+ind": {
     "slope_mean": 0.304108588779245,
     "slope_sd": 0.026612911874351844,
     "n_seeds": 5,
     "within": false
    },
    "labels_ind+a160": {
     "slope_mean": 0.2523103948428077,
     "slope_sd": 0.024276614286922196,
     "n_seeds": 5,
     "within": false
    },
    "late": {
     "slope_mean": 0.5361526622775878,
     "slope_sd": 0.050875874651532496,
     "n_seeds": 5,
     "within": false
    }
   },
   "range": [
    0.8,
    1.2
   ]
  }
 }
]
```

## 임계값 (config.yaml verdict)

```json
{
 "baseline_saturated_score": 0.95,
 "h_e": {
  "calibration_group": "extrap_any",
  "dev_max": 0.1,
  "pair_group": "ood_any"
 },
 "h_f": {
  "pair_group": "all"
 },
 "h_g": {
  "slope_hi": 1.2,
  "slope_lo": 0.8
 },
 "sd_mult": 2.0
}
```

## 실행 비용

```json
{
 "entries": [
  {
   "what": "base/oracle_linear/seed0",
   "seconds": 70.03379392623901
  },
  {
   "what": "base/oracle_mlp/seed0",
   "seconds": 87.20909452438354
  },
  {
   "what": "base/area/seed0",
   "seconds": 241.36983966827393
  },
  {
   "what": "base/oracle_linear/seed1",
   "seconds": 67.19171214103699
  },
  {
   "what": "base/oracle_mlp/seed1",
   "seconds": 85.86089015007019
  },
  {
   "what": "base/area/seed1",
   "seconds": 240.71271634101868
  },
  {
   "what": "base/oracle_linear/seed2",
   "seconds": 66.69194388389587
  },
  {
   "what": "base/oracle_mlp/seed2",
   "seconds": 87.75158381462097
  },
  {
   "what": "base/area/seed2",
   "seconds": 240.02438974380493
  },
  {
   "what": "base/oracle_linear/seed3",
   "seconds": 67.01795697212219
  },
  {
   "what": "base/oracle_mlp/seed3",
   "seconds": 88.121337890625
  },
  {
   "what": "base/area/seed3",
   "seconds": 238.62472438812256
  },
  {
   "what": "base/oracle_linear/seed4",
   "seconds": 66.32097744941711
  },
  {
   "what": "base/oracle_mlp/seed4",
   "seconds": 86.88164353370667
  },
  {
   "what": "base/area/seed4",
   "seconds": 240.81130027770996
  },
  {
   "what": "pred_out6/ours/seed0",
   "seconds": 296.9824664592743
  },
  {
   "what": "pred_out6/ours/seed1",
   "seconds": 296.2484378814697
  },
  {
   "what": "pred_out6/ours/seed2",
   "seconds": 294.05796003341675
  },
  {
   "what": "pred_out6/ours/seed3",
   "seconds": 295.7530424594879
  },
  {
   "what": "pred_out6/ours/seed4",
   "seconds": 296.6630537509918
  },
  {
   "what": "pred_out6/random/seed0",
   "seconds": 296.4791135787964
  },
  {
   "what": "pred_out6/random/seed1",
   "seconds": 298.3630883693695
  },
  {
   "what": "pred_out6/random/seed2",
   "seconds": 300.5498342514038
  },
  {
   "what": "pred_out6/random/seed3",
   "seconds": 297.83817529678345
  },
  {
   "what": "pred_out6/random/seed4",
   "seconds": 296.9634027481079
  }
 ],
 "gpu": "NVIDIA L4",
 "max_vram_mb": 1607.18701171875,
 "last_heads_stage_seconds": 4946.51225733757
}
```

그림: fig_calibration.png, fig_dib_decomp.png (있는 경우). 표: ib_scores.csv, calibration.csv, calibration_dev.csv, law_fits.csv, predictions/.
