# Leveraging Large Language Models for Reusable Requirements Management in Aerospace Software (AeroRM)

## Summary of Artifact
The reuse of requirements artifacts is essential in software development, particularly in aerospace systems where reliability and efficiency are paramount. However, the process of modeling and retrieving reusable requirements remains labor-intensive and error-prone due to the dispersed nature of artifacts across various documents and repositories. To address this challenge, we introduce AeroR, a novel format for encapsulating reusable requirements artifacts, and propose AeroRM, an LLM-based approach for automated AeroR encapsulation and organizing. AeroRM operates in two phases: first aligning and encapsulating requirements artifacts into AeroRs across disparate sources, then organizing these AeroRs into a hierarchical library for efficient retrieval. Our industrial case study demonstrates AeroRM's effectiveness in encapsulating 1,624 AeroRs from six aerospace systems, with practitioners confirming the correctness of the encapsulated requirements. We outline future plans and provide recommendations for requirements engineers working with AeroRM. This innovation offers a promising pathway toward automated, context-aware requirements management that can significantly improve reuse practices in safety-critical domains.

## Sun Search Control System
The mission of the Sun Search Control System (SSCS) is to perform sun localization and orientation by measuring the spacecraft's current attitude using gyroscopes, sun sensors, and star trackers, and it is the control software that rotates the spacecraft's direction along the pitch and roll axes, enabling the sun sensors to detect the Sun and maintain a sun-pointing orientation.

In this folder, we present the orginial software assets for SSCS, including requirement specification and code repository. The encasuplated AeroRs are stored in the `AeroR.xlsx` file, while the hierarchical structure of the AeroR library is visualized in the `AeroR Lib.pdf` file.

## Codes
This repository contains the codes for the *Knowledge Model Abstraction* and *Hierarchical Organization*.

### Alignment
We extract the requirement-device triple from existing artifacts use docx python libraries.

### Encapsulation
We leverage LLMs to abstract high-level knowledge by abstracting the five features, i.e., name, id, keyword, domain, and description, with few-shot learning.

### Grouping and Indexing
We compare EmbStruct with different unsupervised clustering methods and cluster number selection criteria metrics. 
- Clustering Method: Gaussian Mixture Models (GMM), k-means++
- Cluster Number Selection Criteria: Bayesian Information Criterion (BIC), Elbow Method