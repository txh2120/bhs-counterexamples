# Structural Counterexamples to Five Conjectured BHS Laplacian Bounds

This repository contains the paper, code, and data for:

**"Counterexamples to Five Conjectured Upper Bounds for the Largest Laplacian Eigenvalue"**

by Taewoo Ha (2026)

## Summary

Brankov, Hansen, and Stevanović (2006) used the AutoGraphiX system to generate 68 conjectured upper bounds for the Laplacian spectral radius μ(G) of simple connected graphs. Ghebleh et al. (2024) disproved 30 of these via reinforcement learning. We disprove 5 of the remaining 38 open bounds using structural graph construction and equitable partition spectral analysis.

### Counterexample Families

| Bound | Counterexample | n | μ(G) | Bound value | Gap |
|-------|---------------|---|------|-------------|-----|
| 11 | StarOfCliques(K₈, 19) | 153 | 20.5574 | 19.1914 | −1.3660 |
| 13 | StarOfCliques(K₉, 19) | 172 | 20.6847 | 19.8447 | −0.8399 |
| 40 | StarOfStars(39, 8) | 352 | 40.2560 | 38.9818 | −1.2741 |
| 45 | DoubleStar(6, 6) | 14 | 8.7720 | 8.6762 | −0.0958 |
| 48 | P₃ | 3 | 3.0000 | 2.9289 | −0.0711 |

## Repository Structure

```
paper/          LaTeX source and compiled PDF
  main.tex      Paper source (elsarticle class)
  main.pdf      Compiled paper
  references.bib BibTeX references
src/            Python code
  exhaustive_bound_search.py    All 38 BHS bound implementations
  structural_counterexample_search.py  Graph family construction + search
resources/      Data
  structural_search_results.txt  Full counterexample search results (111 counterexamples)
```

## References

- Brankov, V., Hansen, P., Stevanović, D. (2006). "Automated conjectures on upper bounds for the largest Laplacian eigenvalue of graphs." *Linear Algebra Appl.* 414, 407–424.
- Ghebleh, M., Al-Yakoob, S., Kanso, A., Stevanović, D. (2025). "Reinforcement learning for graph theory, I." *Discrete Applied Mathematics*.

## License

MIT
