# Domain-separability report — `demix8` corpus

- **Domains:** 8 (wiki, news, reviews, arxiv, pubmed, math, legal, stories)
- **Documents:** 77,400  |  **Blocks:** 119,677
- **Silhouette (domain labels, n=6000):** **0.015**  _(>0.1 meaningful, >0.25 strong separation in embedding space)_

## Per-domain volume

| domain | docs | blocks | mean chars |
|---|---|---|---|
| wiki | 10,000 | 14,279 | 729 |
| news | 32,000 | 14,968 | 238 |
| reviews | 6,000 | 15,337 | 1308 |
| arxiv | 5,000 | 16,004 | 1638 |
| pubmed | 6,500 | 15,510 | 1221 |
| math | 7,400 | 3,396 | 234 |
| legal | 1,500 | 25,077 | 8558 |
| stories | 9,000 | 15,106 | 857 |

## Unsupervised recovery (k-means clusters vs true domains)

NMI **0.629** · ARI **0.350** · purity **0.601** (8 clusters). A high score means an unsupervised clusterer recovers the domains — the premise the unsupervised expert router exploits.

Contingency `[cluster × domain]` (rows=clusters, cols=wiki, news, reviews, arxiv, pubmed, math, legal, stories):

```
c0: [0, 795, 0, 0, 0, 0, 74, 0]
c1: [1, 2756, 0, 0, 0, 0, 0, 0]
c2: [23, 1, 14249, 0, 2, 1, 0, 0]
c3: [11, 9, 1, 2, 1, 3084, 0, 0]
c4: [11618, 333, 845, 15731, 14828, 135, 24966, 40]
c5: [1, 888, 0, 0, 5, 0, 0, 0]
c6: [415, 9, 104, 0, 0, 11, 0, 15006]
c7: [2210, 10177, 138, 271, 674, 165, 37, 60]
```

## Inter-domain distance (centroid cosine distance)

Larger = more distinct. `0`=identical direction, `1`=orthogonal.

| | wiki | news | reviews | arxiv | pubmed | math | legal | stories |
|---|---|---|---|---|---|---|---|---|
| **wiki** | 0.00 | 0.22 | 0.32 | 0.22 | 0.19 | 0.44 | 0.34 | 0.42 |
| **news** | 0.22 | -0.00 | 0.45 | 0.39 | 0.37 | 0.51 | 0.46 | 0.56 |
| **reviews** | 0.32 | 0.45 | 0.00 | 0.44 | 0.43 | 0.54 | 0.55 | 0.47 |
| **arxiv** | 0.22 | 0.39 | 0.44 | -0.00 | 0.22 | 0.56 | 0.40 | 0.64 |
| **pubmed** | 0.19 | 0.37 | 0.43 | 0.22 | 0.00 | 0.54 | 0.39 | 0.58 |
| **math** | 0.44 | 0.51 | 0.54 | 0.56 | 0.54 | 0.00 | 0.63 | 0.55 |
| **legal** | 0.34 | 0.46 | 0.55 | 0.40 | 0.39 | 0.63 | 0.00 | 0.71 |
| **stories** | 0.42 | 0.56 | 0.47 | 0.64 | 0.58 | 0.55 | 0.71 | 0.00 |

## Sample documents per domain

**wiki**
> Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and M…
> The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more fo…

**news**
> Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again.…
> Carlyle Looks Toward Commercial Aerospace (Reuters) Reuters - Private investment firm Carlyle Group,\which has a reputation for making well-timed and occasionally\controversial plays in the defense industry, has quietly placed\its bets on a…

**reviews**
> I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefor…
> "I Am Curious: Yellow" is a risible and pretentious steaming pile. It doesn't matter what one's political views are because this film can hardly be taken seriously on any level. As for the claim that frontal male nudity is an automatic NC-1…

**arxiv**
> additive models play an important role in semiparametric statistics . this paper gives learning rates for regularized kernel based methods for additive models . these learning rates compare favourably in particular in high dimensions to rec…
> we have studied the leptonic decay @xmath0 , via the decay channel @xmath1 , using a sample of tagged @xmath2 decays collected near the @xmath3 peak production energy in @xmath4 collisions with the cleo - c detector . we obtain @xmath5 and …

**pubmed**
> background : the present study was carried out to assess the effects of community nutrition intervention based on advocacy approach on malnutrition status among school - aged children in shiraz , iran.materials and methods : this case - con…
> backgroundanemia in patients with cancer who are undergoing active therapy is commonly encountered and may worsen quality of life in these patients . the effect of blood transfusion is often temporary and may be associated with serious adve…

**math**
> Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?…
> Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?…

**legal**
> SECTION 1. LIABILITY OF BUSINESS ENTITIES PROVIDING USE OF FACILITIES TO NONPROFIT ORGANIZATIONS. (a) Definitions.--In this section: (1) Business entity.--The term ``business entity'' means a firm, corporation, association, partnership, con…
> SECTION 1. SHORT TITLE. This Act may be cited as the ``Human Rights Information Act''. SEC. 2. FINDINGS. Congress finds the following: (1) The people of the United States consider the national and international protection and promotion of h…

**stories**
> One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to share the needle with her mom, so she could sew a button on her shirt. Lily went to her mom and sai…
> Once upon a time, there was a little car named Beep. Beep loved to go fast and play in the sun. Beep was a healthy car because he always had good fuel. Good fuel made Beep happy and strong. One day, Beep was driving in the park when he saw …
