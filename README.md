# Hallucinated Visual Belief Persistence in Sequential Autonomous Vehicle Perception

Course project for CMPE 249: IA Systems (Track 6, Safety and Evaluation).

**Team:** Nickzad Bayati, Toney Zhen

## Abstract

Vision-language models (VLMs) are increasingly used for perception and reasoning in autonomous systems, including vehicles. A model that forms an incorrect visual belief at one timestep may continue to rely on that belief even after later observations contradict it. This project studies that failure mode: given a sequence of observations from an autonomous-driving environment, we evaluate whether a VLM revises previously incorrect beliefs as new visual evidence arrives, or whether those hallucinations persist and shape later scene understanding.

We consider several categories of visual belief, including object presence, attributes, spatial relationships, scene state, and other driving-relevant information. We propose metrics for hallucination persistence, correction latency, and resistance to contradictory evidence, and compare these behaviors across multiple VLMs on temporally continuous driving data. The goal is to characterize how reliably VLMs update incorrect internal beliefs over time and to identify conditions under which earlier hallucinations contaminate later perception. These findings may inform the design and evaluation of more reliable vision-language components for safety-critical autonomous systems.
