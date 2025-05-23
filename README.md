# mlx-swift-gemma-port
## `mlx-swift-gemma-port`

**Porting Google's MedGemma to Run Natively on iPhone for Offline Healthcare AI**

-----

This repository contains the proof-of-concept code and insights from our 48-hour sprint to port Google's MedGemma, a powerful medical language model, to run natively on an iPhone. Our goal was to demonstrate that server-grade medical AI can operate within the limited resources of mobile devices, specifically targeting offline healthcare solutions for underserved communities.

**Visit our organization:** [Apna Vaidya](https://apnavaidya.ai)

-----

### 🚀 The Mission: Medical AI in Every Pocket

Three days after Google I/O 2025 unveiled MedGemma, we challenged ourselves to make it run where it was never intended: on a smartphone with just 4GB of RAM. This project showcases the feasibility of bringing advanced medical intelligence directly to the hands of healthcare workers in regions with limited connectivity and resources.

### ✨ Why This Matters

  * **Offline Capability:** Crucial for rural areas with unreliable internet access.
  * **Accessibility:** Enables AI-powered medical triage and guidance on common, affordable devices.
  * **Data Privacy:** Patient data remains on the device, eliminating cloud transfer concerns.
  * **Sustainable Scaling:** No server costs per query, making it economically viable for large-scale deployment.

### 💡 The Challenge & The Breakthrough

MedGemma is a complex model with specialized attention mechanisms and medical vocabulary. Porting it to mobile presented significant hurdles:

  * **Android's LiteRT Limitations:** Our initial attempt on Android with LiteRT proved restrictive due to its "black box" nature, hindering our ability to implement MedGemma's custom operations.
  * **Apple's MLX Framework:** We pivoted to Apple's open-source MLX framework. Its transparency allowed us to dive into the source code, understand tensor operations, and make the necessary modifications for MedGemma's unique architecture. This open-source nature was existential to our success.

### ⏱️ The 48-Hour Sprint (Abridged Timeline)

  * **Hour 1-3:** Confidence met reality. Android's LiteRT proved incompatible with MedGemma's custom attention.
  * **Hour 4:** The iPhone pivot to MLX – open-source access proved invaluable.
  * **Hour 6:** Compiler Hell – wrestling with Swift's strict typing and 73 unique errors.
  * **Hour 12:** Architectural Breakthrough – understanding MedGemma's specialized attention heads and dual-stage training.
  * **Hour 18:** The Great Tokenization Battle – hand-rolling a medical-specific tokenizer for iOS.
  * **Hour 24:** First clean compile, but nonsensical output.
  * **Hour 36:** Tensor-Shape Revelation – resolving `NCHW` vs. `NHWC` discrepancies.
  * **Hour 42:** Real-World Optimization – 4-bit quantization, micro-batching, and fused pre/post-processing.
  * **Hour 48:** Success\! Identical answers on two models, sub-second latency, manageable battery drain, and running natively on device.

### ⚙️ What We Built & What It Costs

We successfully fit MedGemma into 4GB of iPhone RAM through:

  * **Reduced Parameter Count:** Knowledge distillation.
  * **Aggressive 4-bit Quantization:** Less than 1% accuracy loss.
  * **Cached Embeddings:** For common medical vocabulary.

The result is a clinically robust model that passes validation against over 1,000 medical QA pairs, suitable for PRANAM's triage scenarios.

### 🌟 Why This Changes Healthcare

This proof-of-concept is a critical step towards:

  * **Empowering Community Health Workers:** Providing instant, offline medical guidance in remote areas.
  * **Bridging the Healthcare Gap:** Addressing the severe doctor-to-patient ratio in many parts of the world.
  * **Innovation in Resource-Constrained Environments:** Proving that high-impact AI doesn't require high-end infrastructure.

### 👨‍💻 Our Process & Learnings

Our sprint involved:

  * **Cross-Platform Debugging:** Layer-by-layer tensor comparisons.
  * **Source Code Deep Dives:** Navigating documentation gaps by reading source code.
  * **Community Support:** The MLX Discord was a lifeline for debugging and shared experiences.

**Key Lessons:**

  * Start with the paper, not just the code.
  * Embrace platform quirks (Swift's strictness caught hidden Python bugs).
  * Test on target hardware from day one.
  * Find your tribe of dedicated developers.

### 🗺️ What's Next

We are committed to open-sourcing this work to accelerate the development of accessible medical AI. Our next steps include:

  * **Fine-tuning Compact Models:** For specialized tasks like maternal health, chronic care, and emergency protocols.
  * **Targeting Budget Android Devices:** Extending support to lower-cost smartphones that are more prevalent in rural communities.
  * **Field Testing:** Collaborating with public health programs to validate offline AI workflows.

### 🤝 Contribute

We believe in AI-powered healthcare for every Indian, and ultimately, for everyone. This project is a testament to what's possible when passion meets engineering.

Want to help build the future of accessible healthcare AI? Follow our journey at Apna Vaidya. We're solving this one village, one phone, one breakthrough at a time.

-----

**This is an open-source project. Contributions and feedback are welcome\!**

-----
