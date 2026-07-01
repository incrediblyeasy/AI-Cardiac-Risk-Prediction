# MIT-BIH Inter-Patient Split (de Chazal DS1 / DS2)

EchoFuseNet is evaluated under the **inter-patient** protocol: the model trains
on one group of patients (DS1) and is tested on a disjoint group (DS2). No
heartbeat from a DS2 patient is ever seen in training. This is the honest
protocol for MIT-BIH; the older intra-patient protocol (randomly shuffling beats
into train/test) leaks patient identity and inflates accuracy by 10-20 points.

The four **paced** records — 102, 104, 107, 217 — are excluded per the AAMI
EC57 recommendation, leaving 44 records: 22 train + 22 test.

Source of truth in code: [`paper1_echofusenet/data/splits.py`](../paper1_echofusenet/data/splits.py).

## DS1 — training (22 records)

```
101 106 108 109 112 114 115 116 118 119 122
124 201 203 205 207 208 209 215 220 223 230
```

## DS2 — testing (22 records)

```
100 103 105 111 113 117 121 123 200 202 210
212 213 214 219 221 222 228 231 232 233 234
```

## Excluded — paced (4 records)

```
102 104 107 217
```

## Reference

de Chazal, O'Dwyer, Reilly, "Automatic classification of heartbeats using ECG
morphology and heartbeat interval features," *IEEE Trans. Biomedical
Engineering* 51(7):1196-1206, 2004 (Table II).
