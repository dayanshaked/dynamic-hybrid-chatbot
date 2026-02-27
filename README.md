# Dynamic Hybrid Chatbot Routing Framework  
### Learning When to Rely on Intent Recognition or Direct Generation

---

## Overview

This project presents a dynamic hybrid conversational AI system that dynamically decides whether to process a query using explicit intent recognition or direct end-to-end generation.

The architecture integrates:

- BERT for intent classification  
- Two T5 generation pipelines (Simple & Intent-Conditioned)  
- A Logistic Regression router  
- Self-supervised routing supervision via BLEU comparison  

---

## System Architecture

### Intent Recognition (BERT)

- Model: `bert-base-uncased`
- Multi-class classification (27 intents)
- Outputs:
  - Predicted intent
  - Full softmax distribution

---

### Dual Generation Pipelines (T5)

**Simple Generation**
```
Input:  user instruction
Output: generated response
```

**Intent-Conditioned Generation**
```
Input:  intent: {predicted_intent} | {instruction}
Output: generated response
```

Both generators are based on `t5-small`.

---

### Learned Routing Mechanism

A Logistic Regression model predicts whether to route the query to:

- Direct generation
- Intent-conditioned generation

Routing features:

```
- Maximum softmax probability
- Margin between top two intents
- Entropy of intent distribution
- Input length
```

Routing labels are constructed automatically using:

```
Per-example BLEU score comparison
```

---

## Dataset

Bitext Customer Support Dataset

- 26,872 dialogue pairs
- 27 intent categories

Data Split:

```
63% Training
7% Validation
15% Meta-development
15% Final Evaluation
```

---

## Results

Intent Classification:
```
Macro-F1 > 0.99
```

Routing:
```
Optimal threshold ≈ 0.637
Balanced routing rate (~50%)
Stable under perturbations (~95%)
```

The router learns a non-trivial decision boundary that balances:

- Response quality
- Computational cost
- Structural reliability

---

## How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Open the notebook:

```
dynamic_hybrid_chatbot_routing_py.ipynb
```

Run all cells sequentially.

---

## Repository Structure

```
dynamic-hybrid-chatbot/
│
├── README.md
├── requirements.txt
├── dynamic_hybrid_chatbot_routing_py.ipynb
├── paper.pdf
└── figures/
```

---

## Authors

Shaked Dayan  
Anaelle Zarviv  
Reuven Eliyahu  

School of Data Science  
Afeka Academic College of Engineering  
Tel Aviv, Israel  

---

## Academic Context

This project was developed as part of an advanced research-oriented study in conversational AI systems.

It investigates dynamic architectural decision-making in hybrid LLM-based chatbot systems.

---

## License

For academic and research purposes.
