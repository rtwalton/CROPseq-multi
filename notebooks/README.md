## Notebook Descriptions

These notebooks provide templates for the design and NGS analysis of CROPseq-multi libraries

### Library Design
- **notebooks/single_target_library_design.ipynb** - This notebook provides a highly customizable template for designing single-target dual-guide libraries (per construct, both guides target the same gene)
- **notebooks/single_target_library_design_simplified.ipynb** - This notebook provides a simplified template for designing single-target dual-guide libraries (per construct, both guides target the same gene)
- **notebooks/single_target_construct_design.ipynb** - This notebook provides a simplified template for designing single-target dual-guide constructs (per construct, both guides target the same gene)
- **notebooks/combinatorial_library_design_programmed_pairs.ipynb** - This notebook provides a template for designing combinatorial libraries (e.g. gene_A-control, gene_B-control, and gene_A-gene_B) to perturb a *user-specified set of pairs of genes*, as well as individual genes.
- **notebooks/combinatorial_library_design_all_by_all.ipynb** - This notebook provides a template for designing combinatorial libraries (e.g. gene_A-control, gene_B-control, and gene_A-gene_B) to perturb *all possible pairs of genes*, as well as individual genes, from a user-provided gene list.
- **notebooks/single_guide_library_design.ipynb** - This notebook provides a template for designing libraries with only ONE guide per construct (analagous to standard CROPseq libraries, but in the CROPseq-multi architecture). This is a non-standard application of CROPseq-multi and is not yet recommended for general use.

### NGS analysis
- **notebooks/library_NGS_analysis.ipynb** - This notebook will map reads from fastq files against a design file to generate per-construct counts, summarizing key metrics like mapping rate, uniformity, and recombination.
