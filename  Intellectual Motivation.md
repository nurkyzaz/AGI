## **Appendix: Intellectual Motivation — The Physics of Information Becoming Law**

### **The Core Intuition**

This research is motivated by a single, deep question: How does information become law?

Across domains, we observe the same pattern. In physics, the Renormalization Group (RG) explains how macroscopic laws—thermodynamics, field theories—emerge from microscopic chaos by systematically discarding irrelevant degrees of freedom while preserving those that determine collective behavior. In learning, neural networks extract concepts and rules from raw sensory data by compressing input while preserving what predicts the target. In intelligence, an agent distills a torrent of sensory information into a compact causal model that generalizes to novel situations.

The intuition is that these are not three separate phenomena united by loose analogy. They are the same phenomenon, viewed from different angles. The mathematical object that governs coarse-graining in physics—the Renormalization Group—may be the same object that governs compression in learning—the Information Bottleneck. And the architecture that implements this compression in biological and artificial systems—the neural network—may be the physical substrate through which information becomes law.

### **The Theoretical Bridge: IB \= RG**

A landmark result establishes this bridge rigorously. Gordon, Banerjee, Koch-Janusz, and Ringel (2021) proved the equivalence between the information-theoretic notion of relevance defined in the Information Bottleneck (IB) formalism and the field-theoretic relevance of the Renormalization Group. In plain terms: the variables that survive coarse-graining in physics are exactly the variables that a learning system would preserve to maximize predictive power. The theorem was published in *Physical Review Letters* and has become a cornerstone of the physics-of-learning literature.

This is not merely an analogy. It is a proven mathematical equivalence—within the domain of field theory. The question that motivates this research is whether the same principle operates in neural networks: do the layers of a deep network perform an RG-like coarse-graining, discarding nuisance degrees of freedom while preserving the causally relevant ones?

### **The Empirical Precedent: Statistical Mechanics of Neural Networks**

The application of statistical mechanics to neural networks has a long and rich history. Mathematical equivalence between statistical mechanics and machine learning theory has been known since the 20th century. Recent reviews have systematized this connection:

* Cui (2025) provides a comprehensive review of statistical physics techniques—including the replica method and approximate message-passing algorithms—applied to the analysis of narrow neural networks. This work demonstrates that statistical mechanics can predict learning performance from macroscopic summary statistics.  
* Malatesta (2025) reviews the statistical mechanics approach to neural networks, focusing on the perceptron architecture as a paradigmatic example.  
* Meshulam and Bialek (2025) , in a *Reviews of Modern Physics* article, review the progress in bringing statistical mechanics and experiment together for networks of real neurons, with a focus on maximum entropy methods and a phenomenological Renormalization Group.

These works establish that statistical mechanics provides the right language for describing learning systems at the macroscopic level—just as it provides the right language for describing physical systems at the macroscopic level.

### **Criticality and Optimal Learning**

A particularly striking line of research suggests that neural networks learn best when they are tuned to a critical state—the boundary between order and chaos, where the system is maximally sensitive to input while remaining stable.

* Ghavasieh et al. (2025) show that the equations used to describe neuronal avalanches in living brains can also be applied to cascades of activity in deep neural networks. They find that maximal susceptibility is a more reliable predictor of learning than proximity to the critical point itself. This suggests that biological and artificial networks may exploit similar macroscopic principles to optimize computation.  
* Roberts, Yaida, and Hanin (2021) develop an effective theory of deep learning, using a representation group flow (RG flow) to characterize signal propagation through networks. By tuning networks to criticality, they show how RG flow leads to near-universal behavior and categorize networks built from different activation functions.

These findings point to a deep connection: the optimal point for learning—where generalization is maximized—may be the same point where a physical system exhibits critical behavior. The information-theoretic counterpart is the Information Bottleneck phase transition, where increasing the compression parameter β causes the model to abruptly discover new relevant features.

### **The Unresolved Question**

Despite these advances, a central question remains unanswered: Is the selective compression of causally relevant variables the mechanism that explains generalization in neural networks? And if so, is this mechanism the same as the coarse-graining of physics and the compression of information theory?

Existing work has established:

1. IB \= RG in field-theoretic systems.  
2. Statistical mechanics provides the right tools for analyzing neural networks.  
3. Criticality correlates with optimal learning.

But no study has yet combined these threads into a single, controlled experiment that directly tests whether a network's ability to recover causally relevant variables predicts its generalization performance—and whether this recovery process mirrors RG coarse-graining.

### **What This Research Adds**

This research is designed to fill that gap. Using procedurally generated ARC tasks with known ground-truth causal structure, we can:

1. Measure whether a network preserves causal vs. nuisance variables.  
2. Test whether this selective preservation predicts OOD generalization.  
3. Compare neural and symbolic representations to see if they converge on the same causal abstraction.  
4. Observe whether compression dynamics (IB phase transitions, layerwise coarse-graining) mirror RG flow.

If the hypothesis holds, it provides evidence that intelligence—whether biological, artificial, or physical—may be governed by a common principle: the selective compression of information to preserve what predicts the future. The laws of physics, the concepts in a neural network, and the rules in a solver's program would all be instances of the same phenomenon: information becoming law.

### **Key References for This Motivation**

| Reference | Contribution |
| :---- | :---- |
| Gordon et al. (2021), *Phys. Rev. Lett.* | IB \= RG equivalence |
| Cui (2025), *J. Stat. Mech.* | Statistical mechanics of narrow neural networks |
| Meshulam & Bialek (2025), *Rev. Mod. Phys.* | Statistical mechanics for networks of real neurons |
| Ghavasieh et al. (2025), "Toward a Physics of Deep Learning and Brains" | Criticality and maximal susceptibility in DNNs |
| Roberts, Yaida & Hanin (2021), *The Principles of Deep Learning Theory* | RG flow in deep networks |
| Tishby, Pereira & Bialek (1999) | Information Bottleneck principle |

### **One Sentence Summary**

This research is motivated by the hypothesis that selective compression of causally relevant information—a principle proven equivalent to RG coarse-graining in physics and observed in critical neural systems—is the mechanism by which information becomes law, and that ARC provides the controlled laboratory to test this hypothesis directly.  
