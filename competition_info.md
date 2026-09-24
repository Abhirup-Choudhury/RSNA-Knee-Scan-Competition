## Description
The knee is the most commonly injured and imaged joint in the body. Osteoarthritis alone affects an estimated 654 million people worldwide, while acute knee injuries account for 15 to 40 percent of all sports-related trauma. MRIs show clinicians ligaments, cartilage, menisci, and bone in detail, without exposing patients to radiation.

Reading those scans isn’t always straightforward. ACL and MCL tears, meniscal damage, cartilage loss, fractures, and other abnormalities can be subtle, and radiologists don’t always interpret them the same way. Access to musculoskeletal radiologists is also limited, especially outside major medical centers, leading to delays and inconsistent diagnoses.

In this competition, you will develop multimodal machine learning models to detect twelve clinically important knee abnormalities. You'll work with the first RSNA AI Challenge dataset that pairs every imaging study with its original radiology report, enabling your models to learn from both visual scans and written diagnostic text.

High-performing models can act as robust decision support tools, delivering the accuracy, consistency, and speed needed to elevate expert-level knee MRI interpretation and improve care across disparate clinic settings.

## Evaluation
Submissions are evaluated by the average area under the ROC curve between the predicted confidence scores and the observed targets across the twelve targets:

Final_Score = 1/12 * (\sigma i = 1...11 (AUC_i))

The final score is, in other words, the macro-averaged AUC ROC.

## Submission File
For each row in the test set, you must predict a confidence score for each of the twelve target labels. The file should contain a header and have the following format:


## Dataset Description

This dataset contains knee MRI studies annotated for twelve common findings: ligament and meniscus injuries, three compartments of osteoarthritis, joint effusion, synovitis, Baker's cyst, bone contusion, and fracture. Each study comprises a collection of individual MRI sequences from a single scanning session formatted as DICOM series. Your task is to predict the per-study probability of each of the twelve findings.

Studies come from a diverse international mix of imaging sites and span a wide range of scanners, protocols, and populations. Only a small subset of training studies carry per-condition labels. We also provide the original text of the radiology report from which you may wish to derive the labels for the remaining studies.

### Files
train.csv One row per training study.

StudyInstanceUID - unique identifier for the study; matches the folder name under train_series/.
Report - the free-text radiology report. May be in any of several languages, depending on the reporting institution.
Twelve binary labels:

ACL - anterior cruciate ligament injury (0/1).
MCL - medial collateral ligament injury (0/1).
Medial Meniscus - medial meniscus tear (0/1).
Lateral Meniscus - lateral meniscus tear (0/1).
Medial OA - osteoarthritis of the medial tibiofemoral compartment (0/1).
Lateral OA - osteoarthritis of the lateral tibiofemoral compartment (0/1).
PF OA - patellofemoral osteoarthritis (0/1).
Effusion - joint effusion / excess fluid (0/1).
Synovitis - inflammation of the joint lining (0/1).
Baker's - Baker's cyst (0/1).
Contusion - bone contusion / bone bruise (0/1).
Fracture - fracture (0/1).
train_series.csv One row per training series. Each series is a single MRI acquisition and each study comprises several series.

StudyInstanceUID - study this series belongs to.
SeriesInstanceUID - unique identifier for the series; matches the folder name under train_series/<StudyInstanceUID>/.
Fluid_Sensitive - 1 if the sequence emphasizes fluid signal (T2, PD, STIR, and similar), 0 otherwise.
Fat_Suppression - 1 if the sequence applies fat suppression, 0 otherwise. Note that although Fluid_Sensitive and Fat_Suppression are often correlated, as observed in the training set, they are not necessarily equivalent for every case.
Anatomical_Plane - imaging plane: Sagittal, Coronal, or Axial.
train_series/ Training DICOMs, organized as train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm. Each .dcm is a single image slice. Series typically contain 20–45 slices (median 30), with a long tail out to a few hundred.


**NOTE**: In local machine data lies inside train_data/train/ instead of train_series/ and same goes for test data.

test.csv Example test file with three study IDs from the public test set. During scoring, this example data will be replaced with the actual test data. There are about 1300 studies in the test set. The Report field will not be provided at the testing stage.

StudyInstanceUID - unique identifier for a test study.
test_series.csv Same schema as train_series.csv, for the example test studies. Replaced with the real test-series descriptors during scoring.

test_series/ Example test DICOMs, same layout as train_series/. Replaced with the real test DICOMs during scoring.

sample_submission.csv A valid submission with all label columns set to 0.5.

### Dataset Distribution Notice
Although efforts have been made to ensure each abnormality is represented in each dataset, the prevalence of abnormalities is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets.


## Extra Guidelines

### Anatomical Overview

The knee is a synovial hinge joint formed by three bones: the femur (thigh bone) above, the tibia (shin bone) below, and the patella (kneecap) in front. The ends of these bones are covered by articular cartilage, a smooth layer that allows near-frictionless motion and distributes load. The joint is enclosed by a capsule lined with synovium, which produces lubricating fluid; an abnormal excess of this fluid is termed a joint effusion, and inflammation of the synovium is termed synovitis.

Knee anatomy: anterior view (top) and lateral view (bottom), showing the femur, tibia, patella, cruciate ligaments, and meniscus. Illustrations by BruceBlaus, CC BY 3.0, via Wikimedia Commons.

For the purposes of this challenge, the knee is considered in three compartments. The medial compartment lies between the medial femoral condyle and the medial tibial plateau, on the inner side of the knee. The lateral compartment lies between the lateral femoral condyle and the lateral tibial plateau, on the outer side. The patellofemoral compartment lies between the patella and the femoral trochlea, the groove on the front of the femur. Osteoarthritis, or cartilage loss, is assessed separately in each of these three compartments.

Four ligaments stabilize the knee. The anterior cruciate ligament (ACL) and posterior cruciate ligament (PCL) lie within the joint, in the intercondylar notch, and control front-to-back stability and rotation. The medial collateral ligament (MCL) and lateral collateral ligament (LCL) run along the inner and outer sides of the knee and resist side-to-side stress. This challenge focuses on tears of the ACL and the MCL.

Between the femur and tibia sit two menisci, the medial meniscus and the lateral meniscus, C-shaped wedges of fibrocartilage that cushion the joint, absorb shock, and improve the fit between the rounded femur and the relatively flat tibia. Tears of the medial and lateral meniscus are evaluated separately.

Other relevant structures include the bone marrow within the femur, tibia, and patella, where a bruise from impact is called a bone contusion (or bone marrow edema) and a break in the bone is a fracture; and the soft tissues behind the knee, where a fluid-filled outpouching of the joint lining is called a Baker, or popliteal, cyst.

### Imaging Overview

Several modalities can image the knee. Radiographs (X-rays) show bones and joint-space narrowing but cannot directly visualize ligaments, menisci, or cartilage. Ultrasound and CT have specific roles but offer limited soft-tissue contrast for the internal structures of the joint. Magnetic resonance imaging (MRI) is the reference standard for evaluating internal derangement of the knee because it provides excellent soft-tissue contrast, depicts ligaments, menisci, cartilage, and bone marrow simultaneously, and does not use ionizing radiation.

A knee MRI examination is not a single image but a set of series, each acquired with a particular pulse sequence and in a particular imaging plane. The three standard planes are axial (cross-sectional, viewed as if looking up through the leg), coronal (a front-facing, side-to-side view), and sagittal (a side-profile, front-to-back view). Different structures are best seen in different planes; for example, the cruciate ligaments and menisci are well evaluated on sagittal and coronal images, and the patellofemoral cartilage on axial images.

Sequences differ in how they weight tissue signal. Fluid-sensitive sequences, such as proton-density or T2-weighted images, often with fat suppression, make edema, effusion, and tears appear bright and are central to detecting most abnormalities; a meniscal tear, for instance, appears as abnormally increased signal that reaches the surface of the meniscus on more than one image. Other sequences emphasize anatomy or cartilage detail. Interpreting a study therefore requires integrating information across multiple series, planes, and slices for the same knee, an inherently three-dimensional task.

Sagittal proton-density, fat-suppressed (fluid-sensitive) MRI of the knee, the type of sequence on which most abnormalities are detected. Image by Ptrump16, CC BY-SA 4.0, via Wikimedia Commons.

For the purpose of the 2026 Knee Abnormality Challenge, a “fluid sensitive” sequence refers to one in which edema, hemorrhage, and other types of fluid appear bright and fat is suppressed in some way.

Because findings can be subtle and the criteria for what counts as clinically significant are nuanced, interpretation varies between readers. This variability, combined with limited access to subspecialty MSK radiologists, motivates the development of consistent automated tools.

### Label Description

Models are evaluated on twelve binary labels, each indicating the presence or absence of a specific finding in the imaged knee. The labels, and the criteria used by the annotating radiologists, are summarized below. In each case, ambiguous or borderline findings (“on the fence”) were graded as negative to favor specificity.

ACL tear: A high-grade partial or full-thickness tear of the anterior cruciate ligament, meaning complete discontinuity of the ligament, or more than 50 percent of fibers disrupted, with or without secondary signs such as characteristic pivot-shift bone contusions. Mild signal change, degeneration, or thickening without discontinuity is graded negative.

MCL tear: A high-grade partial or complete acute tear of the medial collateral ligament, with disrupted fibers and edema within and adjacent to the ligament. Low-grade sprains and chronic or remote stress changes are graded negative.

Medial meniscus tear: Abnormal signal that definitely contacts the meniscal surface on at least two images, or a morphologic abnormality such as a truncated, diminutive, or displaced fragment, involving the medial meniscus. Intrasubstance degeneration that does not reach the surface is negative.

Lateral meniscus tear: The same criteria applied to the lateral meniscus.

Medial compartment osteoarthritis: A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as greater than 50 percent of cartilage thickness, in the medial compartment, with or without underlying subchondral marrow changes.

Lateral compartment osteoarthritis: The same criteria applied to the lateral compartment.

Patellofemoral compartment osteoarthritis: The same criteria applied to the patellofemoral compartment.

Joint effusion: A moderate or large amount of fluid distending the joint.

Synovitis: Inflammation and thickening of the synovial lining of the joint.

Baker (popliteal) cyst: A moderate or large fluid collection in the characteristic location behind the knee.

Contusion: A bone contusion, seen as bone marrow edema-like signal from impact, without a discrete fracture line.

Acute fracture: An acute cortical break or fracture line.

Each study in the annotated reference set was independently labeled by two subspecialty-trained MSK radiologists, with disagreements adjudicated by a third radiologist to produce a single consensus ground truth. Labels are assigned at the level of the whole examination, for a single knee.
