# Disease Reference: Clinical & Technical Background

Technical background for the five conditions covered by Health-OK, compiled from current
clinical and ML literature. This supplements the in-app **About** page and the dataset
notes in `README.md` / `diseases.txt`. Educational reference only — not medical advice.

---

## 1. Cardiovascular (Heart) Disease

### Pathophysiology
Coronary heart disease develops through atherosclerosis: lipid deposition and chronic
inflammation in arterial walls narrow the coronary vessels, reducing myocardial blood
supply. Advanced lesions can rupture and thrombose, causing acute coronary syndromes.

### Epidemiology
- Cardiovascular disease is the leading cause of morbidity and mortality worldwide.
- In the US, hypertension affects roughly 1 in 2 adults (ACC/AHA 2017 criteria), a
  prevalence that has changed little between 2009 and 2023.
- ~38 million US adults have diabetes and a further ~96 million meet criteria for
  prediabetes — both major comorbid drivers of cardiovascular risk.

### Risk Factors
| Category | Factors |
|---|---|
| Behavioral | Smoking, physical inactivity, poor diet |
| Metabolic/clinical | Hypertension, dyslipidemia, diabetes, obesity |
| Demographic | Age, sex, family history, race/ethnicity |

### Clinical / Dataset Features (Cleveland dataset)
`age`, `sex`, `cp` (chest pain type), `trestbps` (resting BP), `chol` (serum cholesterol),
`fbs` (fasting blood sugar > 120 mg/dl), `restecg`, `thalach` (max heart rate achieved),
`exang` (exercise-induced angina), `oldpeak` (ST depression), `slope`, `ca` (number of
major vessels colored by fluoroscopy), `thal` (thalassemia result).

### ML Notes
Random Forest and gradient-boosted trees consistently outperform linear baselines on this
dataset because chest pain type (`cp`), max heart rate (`thalach`), and `thal` interact
non-linearly with risk — confirmed by this project's own feature-importance analysis
(`cp` is the top-ranked feature).

### Prevention & Prognosis
- Cardiovascular disease caused **32% of all global deaths in 2019**, and annual deaths
  are projected to reach ~24 million by 2030.
- The 2026 multisociety dyslipidemia guideline replaced the older Pooled Cohort Equations
  with **PREVENT equations** for 10-year ASCVD risk, and added 30-year risk assessment;
  risk tiers are low (<3%), borderline (3–5%), intermediate (5–10%), high (≥10%).
- Statin therapy started by age 65–75 in high-risk patients (QRISK2 ≥ 20%) measurably
  lowers mortality; high-adherence to a DASH-style diet independently lowers CVD incidence.

**Sources:** [2026 Heart Disease and Stroke Statistics (AHA/Circulation)](https://www.ahajournals.org/doi/10.1161/CIR.0000000000001412) · [Epidemiology of CVD and diabetes in 2026 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13041046/) · [Enhancing CVD Risk Prediction with ML (arXiv)](https://arxiv.org/pdf/2401.17328) · [2026 statin guideline coverage (STAT News)](https://www.statnews.com/2026/03/13/heart-disease-prevention-new-statin-guideline-age-30/) · [Lifestyle interventions and pharmacotherapy for premature CV mortality (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12722816/)

---

## 2. Diabetes Mellitus (Type 2)

### Pathophysiology
Type 2 diabetes results from two intertwined defects that do not necessarily arise
together: peripheral **insulin resistance** and progressive **pancreatic β-cell failure**,
which prevents adequate compensatory insulin secretion. Both genetic predisposition and
environmental factors (diet, obesity, inactivity) drive onset.

### Diagnostic Criteria (ADA)
A diagnosis requires any one of:
- HbA1c ≥ 6.5% (48 mmol/mol)
- Fasting plasma glucose ≥ 126 mg/dL (7.0 mmol/L)
- 2-hour plasma glucose ≥ 200 mg/dL during a 75g OGTT
- Random plasma glucose ≥ 200 mg/dL with classic hyperglycemia symptoms

### Complications
- **Microvascular:** nephropathy, neuropathy, retinopathy
- **Macrovascular:** accelerated atherosclerosis → coronary artery disease,
  cerebrovascular disease, peripheral artery disease

### Clinical / Dataset Features (Pima Indians dataset)
`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`,
`DiabetesPedigreeFunction`, `Age`.

### ML Notes
`Glucose` and `BMI` dominate feature importance in nearly every published model on this
dataset; the dataset's modest size (768 rows) and class imbalance (~35% positive) make
recall-focused metrics (F1, sensitivity) more informative than raw accuracy, consistent
with this project's evaluation approach.

### Prevention & Prognosis
- IDF estimates **537 million adults** had diabetes in 2021, projected to reach **643
  million by 2030** and **783 million by 2045**; global prevalence in adults 20–79 is
  currently ~10.5%.
- The Diabetes Prevention Program (lifestyle intervention: diet + exercise) and
  metformin both reduce progression from prediabetes to Type 2 diabetes; community-based
  lifestyle coaching is effective even in low-resource settings.
- Well-controlled diabetes (HbA1c near target) substantially reduces microvascular
  complication risk; macrovascular risk reduction additionally requires blood pressure
  and lipid management.

**Sources:** [Pathophysiology, Diagnostic Criteria, and Approaches to T2D Remission (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9936340/) · [T2DM Workup — Medscape](https://emedicine.medscape.com/article/117853-workup) · [IDF Diabetes Atlas 11th edition (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S2213858725002992) · [IDF Diabetes Atlas — NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK581934/)

---

## 3. Breast Cancer

### Dataset Origin
The Breast Cancer Wisconsin (Diagnostic) dataset comprises 569 samples (357 benign, 212
malignant), each derived from a digitized image of a **fine needle aspirate (FNA)** of a
breast mass.

### Cell Nuclei Features
Ten base measurements are computed per nucleus, each summarized as mean, standard error,
and "worst" (mean of the three largest values), producing 30 total features:

| Feature | Meaning |
|---|---|
| Radius | Mean distance from center to perimeter points |
| Texture | Standard deviation of gray-scale values |
| Perimeter / Area | Size of the nucleus |
| Smoothness | Local variation in radius length |
| Compactness | perimeter² / area − 1.0 |
| Concavity | Severity of concave contour portions |
| Concave points | Number of concave contour portions |
| Symmetry | Nucleus symmetry |
| Fractal dimension | "Coastline approximation" − 1 |

### Clinical Significance
Malignant nuclei tend to be larger, more irregular (higher concavity/concave points),
and less symmetric than benign nuclei — the geometric basis for why tree-based and
margin-based classifiers (SVM, Random Forest) separate the classes with very high
accuracy (this project's SVM model: ROC-AUC 0.995).

### Prevention & Prognosis
- In 2026, an estimated **382,640 US women** will be diagnosed (321,910 invasive +
  60,730 in-situ), with **~42,140 deaths**.
- Survival is highly stage-dependent: **Stage I ≈ 99% five-year survival** vs.
  **Stage IV ≈ 31%** — early detection is the single biggest lever on outcomes.
- Regular mammography screening lowers breast-cancer mortality by ~26%, though it
  carries a real overdiagnosis rate (~11–22%) and a >50% chance of at least one
  false-positive over 10 years of annual screening — a genuine screening trade-off.
- Global disparities are stark: mortality-to-incidence ratio is ~0.16 in the US vs.
  ~0.55 in Central Africa, reflecting unequal access to early detection and treatment.

**Sources:** [Breast Cancer Wisconsin (Diagnostic) — Kaggle/UCI mirror](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) · [GeeksforGeeks feature reference](https://www.geeksforgeeks.org/machine-learning/breast-cancer-wisconsin-diagnostic-dataset/) · [Breast Cancer Facts & Stats 2026](https://www.nationalbreastcancer.org/breast-cancer-facts/) · [Breast Cancer Survival Rates by Stage](https://mymedicineadvisor.com/health/breast-cancer-survival-rates/)

---

## 4. Chronic Kidney Disease (CKD)

### Definition & Staging (KDIGO)
CKD is defined as reduced kidney function — **eGFR < 60 mL/min/1.73m²** — and/or
albuminuria persisting for ≥ 3 months. KDIGO staging combines:
- **GFR categories G1–G5** (decreasing function)
- **Albuminuria categories A1–A3** (increasing protein leakage)

Combined creatinine + cystatin C-based eGFR (eGFRcr-cys) is now considered superior to
creatinine alone for risk stratification, per the 2024 KDIGO guideline update.

### Risk Factors
Diabetes and hypertension are the two dominant drivers of CKD onset and progression;
additional risk factors include age > 60, obesity, family history, and pre-existing
cardiovascular disease.

### Clinical / Dataset Features (UCI CKD dataset, 400 patients)
Numeric: `age`, `bp`, `sg` (urine specific gravity), `al` (albumin), `su` (sugar), `bgr`
(blood glucose random), `bu` (blood urea), `sc` (serum creatinine), `sod`, `pot`, `hemo`
(hemoglobin), `pcv` (packed cell volume), `wc`/`rc` (white/red cell counts).
Binary/categorical: `rbc`, `pc`, `pcc`, `ba` (urinalysis findings), `htn`, `dm`, `cad`,
`appet`, `pe`, `ane` (comorbidities/symptoms).

### ML Notes
This dataset is known in the literature to be highly linearly/non-linearly separable —
published models routinely report 98–100% accuracy — because urinalysis markers like
specific gravity and albumin are themselves near-diagnostic for CKD rather than indirect
risk proxies. This project's LightGBM model reflects that (ROC-AUC 1.0 on held-out data),
which should be read as a property of the dataset, not evidence the problem is trivial in
real clinical practice.

### Prevention & Prognosis
- **788 million adults** were living with CKD in 2023, more than double the 378 million
  recorded in 1990 — global CKD burden is rising while stroke/ischaemic heart disease
  burden is falling, and CKD is projected to become western Europe's **3rd leading
  cause of death by 2050**.
- **4.6 million people** globally are on kidney replacement therapy (dialysis or
  transplant), nearly 3× the 1.6 million on KRT in 1990 — and dialysis patients have a
  **five-year survival rate below 50%**, making prevention disproportionately valuable
  compared to late-stage treatment.
- Onset and progression are often preventable through blood pressure and glycemic
  control, yet CKD prevention has historically received far less public-health attention
  than stroke/heart-disease prevention.

**Sources:** [KDIGO 2024 CKD Guideline (Kidney International)](https://www.kidney-international.org/article/%20S0085-2538(23)00766-4/fulltext) · [KDIGO CKD 2024 Guidelines Part 1 — NephJC summary](http://www.nephjc.com/news/kdigo-ckd-part1) · [Chronic Kidney Disease dataset mirror (GitHub)](https://github.com/ArjunAnilPillai/Chronic-Kidney-Disease-dataset) · [Updated global burden of CKD: one death every 20 seconds (NDT/Oxford)](https://academic.oup.com/ndt/advance-article/doi/10.1093/ndt/gfag040/8497446) · [Global CKD burden nearly doubles since 1990 (AJMC)](https://www.ajmc.com/view/global-ckd-burden-nearly-doubles-since-1990-reaching-788-million-adults-worldwide)

---

## 5. Parkinson's Disease

### Pathophysiology
Parkinson's disease results from progressive degeneration of dopaminergic neurons in the
**substantia nigra**, disrupting basal ganglia circuitry. Motor symptoms typically emerge
only after **60–80% of dopamine-producing neurons have already been lost**. Pathological
aggregation of alpha-synuclein and neuroinflammation are implicated in disease progression.

### Cardinal Motor Symptoms
- **Bradykinesia** (core symptom) — slowness, reduced amplitude, and impaired rhythm of
  voluntary movement
- **Rigidity** — increased muscle tone
- **Resting tremor**
- **Postural instability** (later-stage)

Clinical diagnosis (UK Brain Bank / MDS criteria) generally requires bradykinesia plus at
least one of rigidity or resting tremor.

### Voice Biomarkers
Roughly 90% of Parkinson's patients develop vocal impairment, making speech a practical,
non-invasive diagnostic signal well before imaging is warranted. Key acoustic features:

| Feature | What it measures |
|---|---|
| Jitter | Cycle-to-cycle variation in fundamental frequency (pitch instability) |
| Shimmer | Cycle-to-cycle variation in amplitude (loudness instability) |
| HNR (Harmonics-to-Noise Ratio) | Ratio of periodic (harmonic) to aperiodic (noise) energy in the voice signal |
| Fundamental frequency (F0) | Baseline pitch and its variability |
| MFCCs | Spectral envelope features capturing timbre changes |

### Clinical / Dataset Features (UCI Parkinson's voice dataset)
`MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)` (frequency range), multiple jitter/shimmer
variants, `NHR`/`HNR`, `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE` — nonlinear dynamical
and fundamental-frequency measures extracted from sustained vowel phonations. Target:
`status` (1 = Parkinson's).

### ML Notes
Because jitter/shimmer/HNR encode the same underlying vocal-tremor phenomenon from
several angles, these features are highly collinear — dimensionality-appropriate models
(regularized linear models, tree ensembles, or a compact ANN as used here) generalize
better than high-capacity models on this dataset's relatively small sample size (195 voice
recordings from 31 subjects).

### Prevention & Prognosis
- Global prevalence reached roughly **10 million cases in 2023** (~1.51 per 1,000
  people, all ages) and is expected to **rise 50% by 2030** as populations age; there is
  no established primary prevention.
- **Levodopa** remains the most effective initial treatment (70–90% of patients see
  significant motor improvement), though motor complications (dyskinesia, "on-off"
  fluctuations) commonly emerge after a few years of therapy.
- **Deep brain stimulation (DBS)** improves motor symptoms in ~60% of advanced patients
  and can reduce medication burden; reported survival after DBS is 98.4% (1yr), 89.8%
  (5yr), and 69.2% (10yr) in treated cohorts.
- Average disease duration from onset to death is **10–20 years**, and early detection
  (including via voice biomarkers, as this project explores) matters because motor
  symptoms only appear after 60–80% of dopaminergic neurons are already lost.

**Sources:** [Voice biomarkers as prognostic indicators for PD (PMC/Nature Sci Reports)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11982320/) · [Motor symptoms of PD: markers for AI-assisted diagnosis (Frontiers)](https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2025.1602426/full) · [Vocal Feature Changes for Monitoring PD Progression (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11939921/) · [Parkinson's Disease Statistics 2026 Market Report](https://worldmetrics.org/parkinson-s-disease-statistics/) · [Update on Medical and Surgical Treatments of PD (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8219497/) · [Long-Term Mortality Analysis in PD Treated with DBS (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3960527/)

---

## Cross-Disease Takeaways for This Project

1. **Dataset size varies by 2 orders of magnitude** (195 rows for Parkinson's vs. 768 for
   diabetes) — smaller datasets favor simpler/regularized models, which is reflected in
   which model wins per disease in `models/<disease>/schema.json`.
2. **Near-diagnostic features exist in 2 of 5 datasets** (CKD's urinalysis markers,
   breast cancer's cell morphology) — these datasets will always show very high ROC-AUC
   and shouldn't be compared apples-to-apples against heart disease/diabetes, where all
   features are indirect risk correlates rather than direct diagnostic markers.
3. **Recall/sensitivity should be prioritized over raw accuracy** across all five,
   consistent with the medical-diagnostics principle in `README.md` — missing a true
   positive (false negative) is costlier than a false alarm.
