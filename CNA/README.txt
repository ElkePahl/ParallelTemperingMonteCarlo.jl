=====================================================================================
Structure Analysis Tool for Atomic Clusters

Author:  AJ Tyler (atyl386@aucklanduni.ac.nz)

================================== Purpose ==========================================

This analysis code provides an implementation of the Common Neighbour Analysis (CNA)
algorithm.The CNA algorithm serves two main purposes in this code: to be able to
distinguish very similar clusters from eachother,and to identify geometric
symmetries and structural features within clusters.

============================= Running the code ======================================

This analysis tool has two callable functions, compare() and classify(), contained in
the "main" module. Both functions have only a single required parameter, which is the
path to the directory containing the input .xyz files. Three example directories have
been inlcuded: "Test", "Gray" and "Diana" "Configs" and runCNA.jl demonstrates useage
and expected results of both the callable functions. Both functions have additional
keyword arguments, see their docstrings for further information.

=========================== Input File structure ====================================

The input .xyz files have to follow a strict structure as follows:

1

N

Energy_1

AtomicNumber x_1 y_1 z_1
.            .   .   .
.            .   .   .
.            .   .   .
AtomicNumber x_N y_N z_N


.
.
.

L

N

Energy_L

.
.
.

Where N is the the number of atoms in the cluster and L is the number of clusters in
the file. See the example directories for full example .xyz files. The xyz file
names must follow the following format: NAME_N_L.xyz
Where NAME is any string (without the characters "_" or ".") that is unique within
its directory. For example, having both configs_13_10.xyz and configs_38_10.xyz in
the same directory would lead to the code overwritting some of the output files due
to the repeated "configs" prefix. There is redundancy in the N and L values,
which serves two purposes. Firstly it is convenient for the functions to know the
size of arrays when initialising, and secondly it is a confirmation that the user
is certain of the contents of the input files. Furthermore, since N is specified in
the filename, it is required that all clusters within a file have the same number of
atoms. 

================================= Outputs ===========================================

The compare() function generates and populates the directory "comparison" in the
local directory. Within "classification", sub directories for each rCut value are
also generated, which in turn contain profile_NAME.txt files. These files contain
the total CNA profiles of each configurations in the input NAME_N_L.xyz file.
For example:

Dict("(3,2,2)" => 30, "(5,5,5)" => 12)

means the cluster had 30 (3,2,2) signatures and 12 (5,5,5) signatures. Also within 
"comparison", two other files are generated for each input file, comparison_NAME.txt
and unique_NAME.txt.The first contains the clusters which each cluster in the .xyz
file is most similar to, based on a similarity score based on their CNA profiles, and
for which rCut values.
For example:

Similarity Score    Most similar to (Config #; rCut)
       0.709                Dict(2 => [1.333])

means that this cluster was most similar to cluster #2 at rCut = 1.333, with a
maximum similarity score of 0.709.
The second file groups the clusters within each .xyz file together, where each group
is so similar, that they should be considered as indistinguishable. The mean and
standard deviation of the energies of each group is also displayed.

The clasify() function generates two directories, "classification" and
"visualisation". For each input file, a classification_NAME.txt file is created
within "classification",containing any identified symmetries for each configuration
in the file. These symmetries can either be icosahedral (ICO), base-centred-cubic
(BCC), face-centered-cubic (FCC), hexagonal-close-packed (HCP) or OTHER. The file
lists which atoms of the each cluster have which type of symmetry and furthermore
if they were found to be part of the cluster's core, or part of a cap.
For example:

Symmetries: Dict([1] => "OTHER(Core)"); Caps: Dict(Dict(10 => [11, 12, 13]) =>
"OTHER", Dict(4 => [6, 7, 9]) => "OTHER")

means that atom #1 was found to be the core of this cluster, with an unidentifiable
symmetry. Additioanlly, atoms # 10 & # 4 are cap atoms, capping atoms (11,12,13)
and (6,7,9) respectively, with both caps also having an unidentifiable symmetry.

For each cluster in each .xyz file, a NAME_i.vesta file is generated inside the
visualisation directory, where i is the cluster index in the NAME .xyz file. These
files can be opened with the VESTA programme (https://jp-minerals.org/vesta/en/) and
can be used to visualise the clusters with atoms being colour coded to aid
visualisation.
The colours have the following meanings:
Red -> icosahedral, Orange -> BCC, Green -> FCC, Blue -> HCP, Purple -> OTHER,Black ->
No identifiable structure.
The shades of the colour also have meanings. Darker shades mean the atom is part of an
identified atom core. Light shades mean the atom is a'cap' atom. An inbetween shade
means the atom is either a capped atom or if core classification failed but the atom
still has an identified symmetry.

==================== Assumptions/Potential Improvements ==============================

The keyword arguments, similarityThreshold in compare() and rCutThreshold in
groupConfigs(), may need more tuning, and also depend on simiarityMeasure. The default
values of 0.95 and 0.9 respectively tuned for the "total" similarityMeasure.

Currently, equilibirum bond length adjustments due to magnetic fields are only for 0.3
au strength fields. This is due to the EBL() function having hardcoded pre-fitted
values to 0.3 au data.

A single rCut value has been assumed to be appropriate for all clusters when calling
the classify() function. The default value is 4/3, however larger values are likely
more approrpiate at higher temperatures.

The coreSize parameter in innerCore() has been assummed to be 0.85, has been narrowed
down to between 0.75 and 1 not inclusive.This parameter is used to determine the
radius of the sphere around centre of mass of radius = corseSize*EBL, which encompasses
all the atoms which form the cluster's inner most shell (shell number 0). This simple
model will likely fail in non-spherical clusters, like those found in strong magnetic
fields.

It has been assumed that iff all atoms that have shell number <=
floor(0.2*maximumShellNumber) have the same CNA profile, then they form the cluster's
core. This assumption was made on the basis that atoms far away from the outermost
shell will likely be highly ordered and thus have the same symmetries.

It has been assumed that outer-most shell atoms are cap atoms if there are less
outer-most shell atoms than second-most outer shell atoms. This is a first order
approximation, so a more complex condition will be required, especially for
larger clusters.

Exact CNA profiles have been required for symmetries to be counted. This may be
relaxed by altering conditions in the classifySymmetry() function, e.g. lowering the
required coordination and number of (5,5,5) signatures from 12 to 9 in order to
identify icosahedral symmetries.