# CS303 Project3 Report 

A report should at least include the following sections. The order given here is suggested but not necessary. You can re-organize it as long as the expected information is presented.

## 1. Introduction

1. Give a general introduction to the problem studied in this project. For example, where does it originate from, how can it be characterized, and to what kind of real-world problems can it be applied? 
2. State the purpose of this project/report.

## 2. Preliminary 

Formally formulate the problem, and explain the terminology and notation you will use throughout this report.

- A *formulation* is an abstract but accurate description of the problem. It should disambiguate the potential confusion in natural languages.
- Example: The problem can be formulated as a Markov Decision Process, which is specified by a tuple $(\mathcal{A},\mathcal{S},\mathcal{T},r,\gamma)$, where $\mathcal{A}$ is the action space, ..., and the objective of an agent is to maximize $\sum_{t=1}^T r(s_t,a_t)$.

## 3. Methodology 

1. General workflow. 
   - Example: The proposed method is divided into steps 1, 2, and 3, each involving algorithms A, B, and C, respectively.
2. Algorithm/model design. 
   - Describe which algorithm/model used in the baseline(or your own) algorithm/model with pseudo-code/flow charts/diagrams. 
   - **DO NOT paste (edited) Python code**.
   - If there is some complex data structure that is not intuitive to understand how it is implemented, give additional explanations.
3. Analysis. Discuss, for example, the optimality and complexity of your algorithm, and what is the deciding factor of its performance.

## 4. Experiments 

### 4.1 Task 1
1. Metrics： how to measure your performance, describe the test flow.
2. Your Experimental results:
3. Try to find through the experiments:
     - the effect of different models (if any) or algorithms
     - the effect of hyperparameters (if any).
     - Analyze the effect of different algorithms/models and hyperparameters if you have corresponding experiments.

### 4.2 Task 2
1. Metrics： how to measure your performance, describe the test flow.
2. Your Experimental results:
3. Try to find through the experiments:
	 - the effect of different models (if any) or algorithms
	 - the effect of hyperparameters (if any).
	 - Analyze the effect of different algorithms/models and hyperparameters if you have corresponding experiments.
	   
## 5. Conclusion

Draw **informative** conclusions from what you have done and written. 

Possible things you can write:

- Comments on the advantages/disadvantages of the model you used.
- Does the experimental result match our expectations/analysis?
- Further thoughts on how it can be improved.
- ...


## Tips

You can write the report in Word, Markdown, or LaTeX, but the submission must be in PDF format.
