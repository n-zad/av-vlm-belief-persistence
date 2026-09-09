# Hallucinated Visual Belief Persistence in Sequential Autonomous Vehicle Perception

Course project for CMPE 249: IA Systems (Track 6, Safety and Evaluation).

**Team:** Nickzad Bayati, Toney Zhen

## Abstract

Vision-language models (VLMs) are increasingly used for perception and reasoning in autonomous systems, including vehicles. A model that forms an incorrect visual interpretation at one timestep may continue to rely on it even after later observations provide contradictory evidence. This project evaluates whether VLMs revise these incorrect interpretations over time or allow them to influence later scene understanding. Using sequential driving-image data, we examine object presence, attributes, spatial relationships, scene state, and other driving-relevant information. We compare stateful and stateless inference across Qwen2.5-VL and LLaVA-OneVision using metrics for hallucination persistence, correction latency, performance difference, and safety-action accuracy. The results will characterize when earlier visual errors persist and inform the evaluation of more reliable VLM components for safety-critical autonomous systems.
