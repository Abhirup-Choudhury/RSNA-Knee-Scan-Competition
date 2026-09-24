# RSNA Knee Abnormality Challenge

## 1. Competition Objective

Predict the probability of **12 knee abnormalities** for each MRI study.

The evaluation metric is **macro-averaged ROC AUC** across the 12 labels:

```text
Final Score = mean(AUC_1, ..., AUC_12)
```

Optimize for study-level ranking quality. Predictions must be continuous confidence scores, not hard 0/1 predictions.

### Target labels

Keep this order consistent throughout the project:

1. `ACL` - anterior cruciate ligament injury
2. `MCL` - medial collateral ligament injury
3. `Medial Meniscus` - medial meniscus tear
4. `Lateral Meniscus` - lateral meniscus tear
5. `Medial OA` - medial tibiofemoral osteoarthritis
6. `Lateral OA` - lateral tibiofemoral osteoarthritis
7. `PF OA` - patellofemoral osteoarthritis
8. `Effusion` - joint effusion
9. `Synovitis` - synovitis
10. `Baker's` - Baker's cyst
11. `Contusion` - bone contusion
12. `Fracture` - acute fracture

Do not change the label order between training, inference, and submission.

---

## 2. Dataset Structure

The official dataset uses:

```text
train_series/
    <StudyInstanceUID>/
        <SeriesInstanceUID>/
            <SOPInstanceUID>.dcm
```

The local dataset uses:

```text
train_data/
    train/
        <study_id>/
            <series_id>/
                <sequence_id>.dcm

test_data/
    test/
        <study_id>/
            <series_id>/
                <sequence_id>.dcm
```

Each study contains multiple MRI series. Each series contains multiple DICOM slices.

Typical series contain approximately 20-45 slices, with some series containing many more.

### Metadata files

`train.csv`

* One row per training study.
* Contains `StudyInstanceUID`.
* Contains the original radiology `Report`.
* Contains the 12 binary training labels.

`train_series.csv`

* One row per training series.
* Contains:

  * `StudyInstanceUID`
  * `SeriesInstanceUID`
  * `Fluid_Sensitive`
  * `Fat_Suppression`
  * `Anatomical_Plane`

`test.csv`

* Contains test study IDs.
* **Does not contain reports during actual testing.**

`test_series.csv`

* Contains the same series metadata as `train_series.csv`.

---

## 3. DICOM Handling

### Never use filenames to determine slice order

DICOM filenames / sequence IDs are not guaranteed to represent spatial order.

Do not assume:

```python
os.listdir(...)
```

returns slices in the correct order.

Do not rely on lexicographic filename sorting.

### Correct slice ordering

When loading a series:

1. Read the DICOM metadata.
2. Determine the spatial slice direction using DICOM orientation/position metadata.
3. Sort slices using the spatial position, preferably using `ImagePositionPatient` and `ImageOrientationPatient`.
4. Use `InstanceNumber` only as a fallback or after verifying that it is valid for the series.

The resulting slice order must represent the actual anatomical progression through the volume.

Keep the original DICOM metadata available until ordering and preprocessing are complete.

---

## 4. Series and Sequence Information

Each series has:

* Anatomical plane:

  * `Sagittal`
  * `Coronal`
  * `Axial`
* `Fluid_Sensitive`
* `Fat_Suppression`

`Fluid_Sensitive` and `Fat_Suppression` are related but **not equivalent**. Do not derive one from the other.

Fluid-sensitive sequences are particularly important because edema, fluid, many tears, contusions, and other abnormalities are more visible on these sequences.

Different abnormalities are better represented by different planes and sequences. The model should therefore use information from multiple series rather than treating the study as a single image.

---

## 5. Study-to-Slot Representation

Represent each study using up to **six sequence/series slots**.

Each slot corresponds to a selected series representation.

Maintain a **presence mask** indicating which slots actually exist.

Do not replace missing slots with arbitrary image data and do not treat missing slots as valid observations.

Example:

```text
slot embeddings:
[x1, x2, x3, x4, x5, x6]

presence mask:
[1, 1, 0, 1, 0, 1]
```

The model must explicitly mask unavailable slots.

The six-slot representation is used because a complete knee study contains complementary information across multiple MRI acquisitions.

---

## 6. Image Preprocessing

For every selected series:

### 6.1 Load and order slices

* Read all DICOM files.
* Convert pixel data to an image array.
* Sort slices spatially using DICOM geometry.
* Keep the complete ordered stack before sampling.

### 6.2 Physical-space processing

Use DICOM `PixelSpacing` rather than assuming every scan has the same physical resolution.

The preprocessing is defined in physical units where appropriate, so differences in scanner resolution do not directly change the anatomical field represented by the crop.

A **130 mm physical crop** is used around the relevant knee region before resizing.

Do not treat pixel dimensions as equivalent to physical dimensions.

### 6.3 Resize

After physical-space cropping, resize the resulting images to the model's fixed spatial input resolution.

Keep the resize operation identical between training and inference.

### 6.4 Intensity normalization

Normalize each series using its own intensity distribution.

Current approach:

* Compute the approximately 1st and 99th intensity percentiles over the series stack.
* Clip intensities to this range.
* Normalize to the model input range.

Do not use a global dataset-wide intensity range.

### 6.5 Laterality / orientation normalization

Normalize the orientation so equivalent anatomical structures have a consistent left/right representation.

Use DICOM patient/image geometry rather than filename assumptions.

The current preprocessing uses image-center patient-coordinate information to infer laterality and applies the required flips/reversal for the different anatomical planes.

For sagittal series, the anatomical slice direction may need to be reversed.

For coronal and axial series, apply the corresponding orientation normalization.

The goal is that the same anatomical side and direction have a consistent representation across studies.

### 6.6 Storage

Preprocessed slices can be cached as `uint8` after normalization to reduce storage and loading overhead.

Do not repeatedly decode and preprocess the same DICOM files during every training epoch if cached preprocessing is available.

---

## 7. Slice Sampling

A full MRI series may contain many slices, but processing every slice for every series is unnecessarily expensive.

Use sampled **consecutive slice groups** rather than isolated random slices.

The current representation uses a small group of adjacent slices as a 2.5D input where appropriate.

The important properties are:

* slices remain spatially ordered;
* sampled slices are consecutive;
* the model receives local through-plane context;
* sampling is performed from the ordered volume, not from arbitrary filenames.

During training, sample different valid slice groups to increase coverage of the volume.

During inference, aggregate information from multiple sampled groups so the final series representation is not dependent on one arbitrary slice location.

---

## 8. Study-Level Architecture

The architecture has three main stages:

```text
DICOM series
    ↓
preprocessed slice groups
    ↓
image encoder
    ↓
series/slot embeddings
    ↓
12 label-specific query heads
    ↓
study-level predictions
```

### 8.1 Image encoder

Use a pretrained image encoder to convert each slice group into an embedding.

Avoid unnecessarily large models because the competition contains many MRI images and compute is limited.

Fine-tuning should initially be conservative:

* keep most of the pretrained encoder fixed;
* fine-tune only later encoder blocks;
* use a lower learning rate for the encoder than for newly initialized prediction layers.

The prediction head should learn faster than the pretrained image representation.

### 8.2 Slot embeddings

Each selected series produces one slot embedding.

Add a learned embedding representing the slot/sequence identity so that the model can distinguish different MRI sequence types.

The model therefore receives both:

```text
visual information
+
sequence/slot identity
```

rather than treating all series as interchangeable.

---

## 9. Twelve Label-Specific Decisions

The study has six possible slot embeddings but twelve different abnormalities.

Do not simply average all six slot embeddings and use one shared representation for every label.

Instead, each target has its own learned query.

Conceptually:

```text
label query
     ↓
attention over six slot embeddings
     ↓
masked aggregation
     ↓
one study-level logit
```

There are **12 independent label queries**, one for each target.

This allows different abnormalities to attend to different MRI sequences.

For example, a label may rely more heavily on sagittal/coronal information while another may rely more heavily on axial information.

---

## 10. Masked Slot Attention

Slot attention must respect the study's slot-presence mask.

For a study with:

```text
mask = [1, 1, 0, 1, 0, 1]
```

the absent slots must receive zero attention.

Use a masked softmax:

```text
attention_logits[missing_slot] = -inf
attention = softmax(masked_logits)
```

The attention weights must therefore sum only over available slots.

Never allow a missing slot to influence a prediction.

Do not zero-fill a missing embedding and then apply an ordinary softmax, because the zero vector can still receive attention.

---

## 11. Aggregating Slice Groups

A series can produce multiple sampled slice-group embeddings.

Aggregate the groups into a single series/slot representation before study-level prediction.

The final flow is:

```text
multiple slice groups
        ↓
image encoder
        ↓
group embeddings
        ↓
aggregation
        ↓
one embedding per slot
        ↓
six-slot study representation
```

The inference pipeline should use multiple groups to cover the series rather than relying on one random group.

---

## 12. Labels and Radiology Reports

Only a small subset of training studies has direct per-condition labels.

The original radiology report is available for training studies and can be used to derive additional supervision.

Reports may be written in different languages.

The report must **not** be used at test inference because the test `Report` field is not provided.

Do not allow test reports to enter the model because they do not exist in the actual test set.

---

## 13. Ground-Truth Label Definitions

Use the competition definitions when interpreting labels.

Borderline or ambiguous findings were labelled negative.

### ACL

Positive:

* high-grade partial tear or full-thickness tear;
* more than 50% of fibers disrupted or complete discontinuity.

Negative:

* mild signal change;
* degeneration;
* thickening without significant discontinuity.

### MCL

Positive:

* high-grade partial or complete acute tear;
* disrupted fibers with surrounding edema.

Negative:

* low-grade sprain;
* chronic/remote stress changes.

### Meniscus tears

Positive:

* abnormal signal definitely reaching the meniscal surface on at least two images;
* or clear morphological abnormality such as truncation, displacement, or a displaced fragment.

Negative:

* intrasubstance degeneration that does not reach the surface.

This applies separately to medial and lateral menisci.

### Osteoarthritis

Evaluate three compartments separately:

* medial tibiofemoral;
* lateral tibiofemoral;
* patellofemoral.

Positive:

* moderate/large region of high-grade cartilage loss;
* approximately 1 cm or larger;
* more than 50% cartilage thickness loss.

### Effusion

Positive when there is a moderate or large amount of fluid distending the joint.

### Synovitis

Positive when there is inflammation/thickening of the synovial lining.

### Baker's cyst

Positive when there is a moderate/large fluid collection in the characteristic popliteal location behind the knee.

### Contusion

Positive for bone marrow edema-like signal caused by impact without a discrete fracture line.

### Fracture

Positive for an acute cortical break or fracture line.

---

## 14. Train/Validation Split

The split must be performed at the **study level**.

Never split individual slices or series from the same study across training and validation sets.

All series belonging to one `StudyInstanceUID` must remain in the same split.

The validation pipeline must reproduce the real test-time study-level inference process.

---

## 15. Test-Time Constraints

At actual competition inference:

* test reports are unavailable;
* test DICOMs follow the same study/series/slice structure;
* test series metadata follows the same schema;
* approximately 1300 test studies are expected.

The model must therefore be able to produce predictions using:

```text
test DICOMs
+
test_series.csv
```

without requiring `Report`.

---

## 16. Submission

The submission must contain one row per test study and one confidence score for each of the 12 labels.

Use the exact label names and column order required by `sample_submission.csv`.

Before creating a submission:

1. Verify every test study has one prediction.
2. Verify all 12 target columns are present.
3. Verify no predictions are NaN or infinite.
4. Verify predictions are continuous confidence scores.
5. Verify the row identifiers match the test studies.
6. Verify column names exactly match the sample submission.

---

## 17. Important Rules for Code Changes

### Never

* reorder DICOM slices based on filenames;
* assume `os.listdir()` is ordered;
* mix studies between train and validation;
* use test reports;
* treat missing slots as real zero-valued images;
* remove the slot-presence mask;
* silently change preprocessing between training and inference;
* change label ordering without updating the complete pipeline;
* modify raw DICOM files.

### Always

* preserve `StudyInstanceUID` and `SeriesInstanceUID`;
* use DICOM metadata for spatial ordering;
* keep series metadata attached to the corresponding series;
* preserve the six-slot presence mask;
* use the same preprocessing for validation and test;
* make preprocessing deterministic when reproducibility is required;
* cache expensive preprocessing where practical;
* validate tensor shapes and masks before training.

---

## 18. Current Pipeline

The current intended pipeline is:

```text
Study
  ↓
Group DICOMs by series
  ↓
Read DICOM metadata
  ↓
Spatially order slices
  ↓
Select/assign up to six useful sequence slots
  ↓
Normalize orientation/laterality
  ↓
Physical-space crop (~130 mm)
  ↓
Resize
  ↓
Per-series percentile intensity normalization
  ↓
Sample consecutive slice groups
  ↓
Image encoder
  ↓
Aggregate groups into one embedding per slot
  ↓
Six slot embeddings + presence mask
  ↓
12 label-specific queries
  ↓
Masked softmax attention over available slots
  ↓
12 study-level logits
  ↓
Sigmoid
  ↓
12 probability/confidence predictions
  ↓
Submission
```

---

## 19. Architecture Decisions to Preserve

The following decisions are intentional and should not be changed casually:

1. **Study-level prediction**, not independent slice-level classification.
2. **Multiple MRI series**, because different sequences and planes contain complementary information.
3. **Up to six sequence slots** with an explicit presence mask.
4. **Consecutive slice groups** to retain local 3D context.
5. **Pretrained image encoder** with conservative fine-tuning.
6. **Learned slot/sequence identity embeddings**.
7. **One learned query per target**, allowing each abnormality to select relevant sequences.
8. **Masked attention** so missing sequences cannot affect predictions.
9. **Study-level aggregation** before producing the final twelve predictions.
10. **Report-derived supervision**, when used, must have lower confidence than directly annotated labels.
11. **Macro-AUC** is the optimization/evaluation target, so preserve continuous probability outputs.

Any architectural change should be tested against the current baseline rather than replacing these components without evidence.

---

## 20. Experiment Tracking

Keep competition experiments reproducible.

For every meaningful experiment record:

* model/encoder;
* input resolution;
* slice-group configuration;
* selected sequence slots;
* preprocessing version;
* train/validation split;
* loss;
* learning rates;
* frozen/unfrozen layers;
* report-label usage;
* augmentation;
* validation macro-AUC;
* per-label AUCs;
* inference configuration.

Do not judge a model only by overall AUC. Track all twelve label AUCs because the final score is their unweighted mean.

When changing preprocessing or architecture, identify exactly which component changed.
