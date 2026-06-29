# Domain-separability report — `dev` corpus

- **Domains:** 5 (wiki, news, reviews, arxiv, pubmed)
- **Documents:** 100,000  |  **Blocks:** 42,929
- **Silhouette (domain labels, n=6000):** **0.051**  _(>0.1 meaningful, >0.25 strong separation in embedding space)_

## Per-domain volume

| domain | docs | blocks | mean chars |
|---|---|---|---|
| wiki | 20,000 | 6,235 | 726 |
| news | 20,000 | 2,146 | 240 |
| reviews | 20,000 | 11,896 | 1325 |
| arxiv | 20,000 | 13,186 | 1572 |
| pubmed | 20,000 | 9,466 | 1216 |

## Unsupervised recovery (k-means clusters vs true domains)

NMI **0.759** · ARI **0.737** · purity **0.897** (5 clusters). A high score means an unsupervised clusterer recovers the domains — the premise the unsupervised expert router exploits.

Contingency `[cluster × domain]` (rows=clusters, cols=wiki, news, reviews, arxiv, pubmed):

```
c0: [5356, 2143, 292, 753, 171]
c1: [438, 0, 11589, 1, 0]
c2: [33, 0, 1, 8312, 100]
c3: [326, 2, 9, 43, 9194]
c4: [82, 1, 5, 4077, 1]
```

## Inter-domain distance (centroid cosine distance)

Larger = more distinct. `0`=identical direction, `1`=orthogonal.

| | wiki | news | reviews | arxiv | pubmed |
|---|---|---|---|---|---|
| **wiki** | -0.00 | 0.60 | 0.65 | 0.82 | 0.99 |
| **news** | 0.60 | -0.00 | 0.87 | 0.94 | 1.09 |
| **reviews** | 0.65 | 0.87 | 0.00 | 0.96 | 1.00 |
| **arxiv** | 0.82 | 0.94 | 0.96 | -0.00 | 1.11 |
| **pubmed** | 0.99 | 1.09 | 1.00 | 1.11 | 0.00 |

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
