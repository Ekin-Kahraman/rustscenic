# Direction-preserving activity diagnostic, 8 November 2026

Hypothesis fixed before new evaluations: subtracting separately normalised
positive and negative AUCell channels loses information when a TF retains
both directions. This occurs often for RustScenic and not for the common
SCENIC+ TFs in these archived networks.

Test exactly one alternative: keep both directional AUC channels. Use the
original common TF roster, two named columns per TF for both methods and zero
for a missing direction. Report effective nonconstant features as well as
padded width. Keep the original contrast endpoint alongside the alternative.

No GRN, topic, motif, enhancer or target-set fitting or filtering changes.
Keep the original 80% external gene-coverage rule, AUCell top fraction 0.05,
tie seed 20261106, standardisation, balanced logistic regression C=1 and
donor splits. No model selection or threshold search on external labels.

Development: leave-one-donor-out classification within the original seven
BMMC training donors, at both network training sizes and all three seeds.
These donors also contributed to network fitting; this is development, not
independent validation. Assess consistency before integrating a public option.

Replication: rerun the frozen three-donor BMMC test endpoint and eight-donor
Kang endpoint, retaining all settings and all results. These cohorts have
already been inspected, so replication here is retrospective and cannot be
claimed as untouched validation or a prespecified equivalence test.

Report donor-level changes, paired donor-bootstrap intervals, class-level
confusions and convergence. Do not drop unfavourable seeds/classes/endpoints.
Measure scoring and representation cost on the existing real matrices in
fresh processes. An improvement establishes better use of existing networks,
not improved edge inference or causal biology.

If the alternative is inconsistent, retain the diagnosis without adopting it
as a default or starting a broad search. Preserve existing public defaults and
all prior benchmark evidence. No new Huangfu run or external message is authorised.
