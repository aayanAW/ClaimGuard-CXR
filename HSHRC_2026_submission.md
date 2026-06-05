Aayan Alwani & Ethan Wang

High School Health Research Challenge

1 May 2026

*Audit the Bots Before You Trust Them: A Teen-Led Protocol for Verifying AI in Our Hospitals*

When wildfires choke a region or a heatwave overwhelms a city, hospitals get pushed into crisis mode. To keep up, more of them are deploying artificial intelligence (AI) tools that read chest X-rays, draft radiology reports, and flag urgent conditions before a human doctor reviews the case (Bannur et al. 2024). Smaller, lower-resourced hospitals are leaning on these tools the most. That puts AI inside healthcare infrastructure, which is part of every community's built environment. Climate change is now driving up demand at these same hospitals: rural clinics, low-income urban hospitals, and the ones receiving climate-displaced patients.

The problem is structural. Almost every AI radiology tool in clinical use was trained at a well-resourced research hospital, on imaging data from that hospital's own patients. When the same tool is deployed somewhere else, the new hospital usually has different scanners, different demographics, and reports written in a different language. Performance drops, often without anyone noticing. A 2018 study showed that an AI pneumonia detector trained at one U.S. hospital lost most of its accuracy at a different one (Zech et al. 2018). Across income lines or national borders, the gap can be much wider (Pooch et al. 2020). And the hospitals most exposed to climate-driven surges of patients (heat illness, smoke inhalation, displaced populations) are usually the ones least able to check whether their AI tool works on the patients in front of them.

Current solutions do not close this gap. Hospitals usually decide to deploy an AI tool based on the accuracy numbers in the original research paper, often over 90% on standard benchmarks (Mohri and Hashimoto 2024). But aggregate accuracy can hide a serious failure. An AI system can score above 90% on a chest X-ray benchmark without ever looking at the image, by exploiting patterns in the report text instead. In our own work, the AI verifier with the highest accuracy turned out to be completely image-blind. We replaced every image with a blank, and accuracy dropped by only 2 percentage points. The headline number was excellent. The medicine was not happening. Newer safety wrappers like conformal trust-selection do not fix this either, because they certify whatever the underlying AI outputs, even when the AI never used the X-ray.

We propose ClaimGuard, a free, open audit protocol any hospital can run to check whether an AI radiology tool is using the medical image at all. It works on the same laptop a small clinic already has, and it runs both before deployment and during it. ClaimGuard has two parts.

The first part is a three-step diagnostic. The image test (IMG) replaces the X-ray with a blank image and measures how much accuracy drops. If accuracy barely changes, the AI was not using the image. The evidence test (ESG) shuffles the supporting text across patients and measures the same drop. If the drop is small, the AI is ignoring the patient's record. The side-flip test (IPG) flips the X-ray left-to-right on cases where left-right anatomy matters, like a one-sided pneumothorax, and measures the drop again. A small drop here means the AI is ignoring the spatial layout of the image. The three tests cost almost nothing, take a handful of extra computer runs per case, and work on closed commercial AI tools, since the hospital only has to change the inputs and read off the outputs.

The second part is a deployment alarm. The hospital re-runs the three tests every so often on the AI's actual production traffic. If one of the gaps suddenly collapses, for example after a wildfire pushes a new displaced population into the hospital's queue, the AI has lost grounding on the new patient mix. The hospital learns about it before any patient is harmed.

ClaimGuard is realistic for teenagers to build and spread. The audit can be written in roughly 200 lines of open-source Python and run on a laptop. A team of high schoolers, especially those in CS or health equity clubs, can package the protocol with plain-language documentation, partner with student-led health nonprofits, and offer it for free to community clinics and teaching hospitals in their region. The Pioneer in Health Research seed money would cover a pilot at five sites, including outreach materials translated into Spanish and other languages spoken in underserved patient populations. A public registry, with notes on which AI tools have been verified on which populations, would work like a consumer-report layer for hospital AI, maintained by students who care whether the tools work for the communities they serve.

Climate change is pushing more hospitals to rely on AI in the same places where verification capacity is thinnest. Healthcare's built environment is no longer just the building, the imaging machines, and the bed count. It now includes the AI tools running inside, and those tools decide who gets safe care and who does not. ClaimGuard is a cheap, teen-built check on that environment, so the AI in our hospitals is not just accurate on a paper benchmark but accurate for the patients in front of it.

Works Cited *(not counted towards word count)*

Bannur, Shruthi, et al. "MAIRA-2: Grounded Radiology Report Generation." *arXiv*, 6 June 2024, arxiv.org/abs/2406.04449. Accessed 1 May 2026.

Mohri, Christopher, and Tatsunori Hashimoto. "Language Models with Conformal Factuality Guarantees." *Proceedings of the 41st International Conference on Machine Learning (ICML)*, 2024, arxiv.org/abs/2402.10978. Accessed 1 May 2026.

Pooch, Eduardo H. P., et al. "Can We Trust Deep Learning Based Diagnosis? The Impact of Domain Shift in Chest Radiograph Classification." *Thoracic Image Analysis Workshop, Medical Image Computing and Computer Assisted Intervention (MICCAI)*, 2020, arxiv.org/abs/1909.01940. Accessed 1 May 2026.

Zech, John R., et al. "Variable Generalization Performance of a Deep Learning Model to Detect Pneumonia in Chest Radiographs: A Cross-Sectional Study." *PLOS Medicine*, vol. 15, no. 11, 6 Nov. 2018, e1002683, journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683. Accessed 1 May 2026.

---

**Submission instructions:**
- Open this file's body text (everything from the header through the closing paragraph) in Google Docs or Word.
- Set 12pt Times New Roman throughout.
- Center the title and italicize it (matches sample: *Audit the Bots Before You Trust Them: A Teen-Led Protocol for Verifying AI in Our Hospitals*).
- Left-align the byline block (Name / "High School Health Research Challenge" / Date) at the top.
- Body paragraphs in standard left-aligned blocks; first-line indent optional (sample uses indent).
- Works Cited section: hanging indent, alphabetical by author surname (already alphabetical here).
- Word count of body: ~860 (under the 1000 cap; references not counted).
- Submit through https://hshrf.org/hshrc2026 by **1 May 2026, 11:59 PM EST**.
- The form requires references entered separately — paste the four Works Cited entries into the references field.
