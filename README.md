Patient-specific identification of hyperelastic parameters requires mechanical
testing of each patient's tissue, which is not possible in clinical practice.
We formulate a hierarchical Bayesian model in which the parameters of an
Matched invariant-based angular-integration (AI) constitutive law are expressed as
log-linear functions of patient covariates (age, sex, smoking history) plus a
subject-level random effect. The posterior provides (i) parameters for every
tested specimen with full uncertainty, (ii) covariate effect sizes, and (iii) a
posterior-predictive parameter distribution for a new, untested patient
from covariates alone. This removes the patient-specific bottleneck while
honestly propagating inter-patient variability into downstream simulations.
<img width="1950" height="930" alt="fig13_holdout_prediction" src="https://github.com/user-attachments/assets/fe02946d-755a-4bca-98fe-813cfa225261" />
<img width="1950" height="495" alt="fig14_truth_recovery" src="https://github.com/user-attachments/assets/cbc40799-20f2-4a13-9c16-300e196cf155" />
<img width="1950" height="1350" alt="fig08_sample_fits" src="https://github.com/user-attachments/assets/626a9e7c-c9c6-4119-b777-6174f57885f6" />
