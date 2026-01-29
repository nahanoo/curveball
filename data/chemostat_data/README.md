## Test data set

This a dataset of OD data of different chemostat experiments. It consists of three different species growing on three different resources. 
For two species and one resource there are biological replicates.
Dataframe columns:
- `time`: experimental time in hours
- `OD`: OD measurement
- `replicate`: name of biological replicate if present (only for Ct, Oa on acetate)
- `substrate`: name of growth limiting carbon source
- `species`: abbreviation of species:
    - Ct: Comamonas testosteroni
    - Oa: Ochrobactrum anthropi
    - At: Agrobacterium tumefaciens
- `dilution_rate`: experimental dilution rate in 1/h

You can check `data_overview.pdf` for a plot that summarizes all conditions. 