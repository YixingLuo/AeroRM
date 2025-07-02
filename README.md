# Leveraging Large Language Models for Reusable Requirements Management in Aerospace Software (AeroRM)

## Summary of Artifact
The reuse of requirements artifacts is essential for software development, particularly in aerospace systems where high reliability and efficiency are paramount. However, current methods for managing these artifacts are predominantly manual and costly, as the artifacts are dispersed across multiple documents and exist in heterogeneous formats. Leveraging recent advances in large language models (LLMs) offers a promising opportunity for automating and scaling requirements reuse. Nonetheless, this approach faces two critical challenges: (1) encapsulating scattered, diverse requirement artifacts into coherent and reusable components, and (2) organizing these components into a structured, easily retrievable library. To address these challenges, we introduce AeroR, a novel format for encapsulating aerospace requirements artifacts, and propose AeroRM, an LLM-based method for automated requirements artifact management. AeroRM operates in two phases: first, it consolidates requirements from disparate sources into reusable components (i.e., AeroRs); then, it organizes these AeroRs into a hierarchical library to enable efficient retrieval. We validate AeroRM on artifacts from six aerospace projects, successfully encapsulating 1,624 AeroRs. A user study with senior engineers shows that 67% of sampled AeroRs are high-quality, and a comparative retrieval study across 12 configurations achieves a best-case Recall@10 exceeding 80%. These results demonstrate the potential of AeroRM to automate requirements reuse at scale, offering a practical solution for safety-critical domains. 

## Codes
This repository contains the code for the *Encapsulation of Requirements Artifacts* and *Hierarchical Organization of AeroRs*.

### Encapsulation
We leverage LLMs to abstract high-level knowledge by abstracting the five features, i.e., name, id, keyword, domain, and description, with few-shot learning.

### Hierarchical Organization
We compare AeroRM with different unsupervised clustering methods and cluster number selection criteria metrics. 
- Clustering Method: Gaussian Mixture Models (GMM), k-means++
- Cluster Number Selection Criteria: Bayesian Information Criterion (BIC), Elbow Method
- Evaluation Metrics: SC, CHI, Rouge-L, Recall@10
