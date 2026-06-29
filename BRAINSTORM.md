# Soft MOE Idea!

The premise is simple: how can we use the EM process to motivate soft MoE-like partitioning within a single model? That is, how can we allow for training subspaces of the weight matrix of the LLM based on the input (scaffolding training better)? 

This is extremely similar to the idea of MoE and also to the LLM-Persona project I'm working on with Levin + friends!

## General ideas / lit review

*   Two versions for performance!
    *   Supervised version - train expert-tokens on user-set or algorithm-set boundaries through the EM lifestyle and show that the same weight space can learn multiple experts
    *   Unsupervised version - train expert-tokens and assign expert-tokens across the broader corpus during the training process, have experts acquire selectivity!

### Supervised Version

*   Intuitive understanding is to take a dataset with MoE ideas and train one expert per sub-domain!
    *   We can do an EM-style technique and we can also just choose extremely orthogonal directions and see how they scaffold the latent space!!

*   c-BTM-like approach: Use Cobweb as the Clusterer, base it off of CBTM, condition a PATH of tokens on a given corpus and use a mixture of concepts to analyze novel queries!
    *   We don't have to use Cobweb, we can just use any clusterer, but c-BTM actively uses multiple LM experts, whereas our idea would leverage a single LM and learn expert TOKENS as a method of subdividing the space!
    *   Alternatively - learn a latent space based on Cobweb as the clusterer so that we can interpolate the best expert for the job!!
*   We can do a method of hierarchical passing down where we give multiple tokens
    *   Generally, the most important thing is that we leverage the same "mixture-of-concepts" idea that produced good representations!

### Unsupervised Version

*   This is a little harder, but the vibes are that we can measure what token is the best for a given task and use the expectation-maximization technique to take differentiate tokens with meaning over different sectors
*   Idea is that we select the best token from a set of tokens at each interval while also maximizing token difference while also maximizing token balance!
    *   So, the regularization will have a term for pushing tokens apart and a term for load-balancing on tokens given a batch
    *   It should prioritize keeping tokens apart and then prioritize load-balancing as a secondary factor
    *   This is really fragile so weights are really really important - we should combine the two into one term by weighted sum first (alpha and 1 - alpha) and then multiply the whole thing within the loss

## Implications

This is by far the easiest way to validate whether our idea is as good as straight MoE!! For Persona-LLM or other things, and especially the dichotomy of Supervised vs. Unsupervised
*   If we find a big enough LLM, the goal is that we can condition 

## Baselines

Testing against the following ideas:
*   Original MoE ideas
*   c-BTM
*   MoP - prompt embeddings
*   Our solution but with a fixed and well-defined set of tokens (orthogonal, constant)
    *   Can see here how the idea of learning the embeddings improves or hinders performance