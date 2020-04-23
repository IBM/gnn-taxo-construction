================================================================================
                              SEMEVAL-2015 TASK 17
                   TExEval: Taxonomy Extraction Evaluation
	Paul Buitelaar,Georgeta Bordea, Roberto Navigli and Stefano Faralli
================================================================================
			http://alt.qcri.org/semeval2015/task17/
================================================================================
TExEval_trialdata_1.1
================================================================================

==================
PACKAGE CONTENTS
==================

The trial data package contains the following:

README.txt                                        This file
ontolearn_AI.taxo				  Artificial Intelligence taxonomy[¹] 
ontolearn_AI.taxo.eval				  Human evaluation for the Artificial Intelligence taxonomy relations[¹]
WN_plants.taxo					  WordNet plants taxonomy 
WN_plants.terms					  WordNet plants terminology
WN_vehicles.taxo				  WordNet vehicles taxonomy
WN_vehicles.terms				  WordNet vehicles terminology

=============
FILE FORMAT
=============

The input files format for the taxonomies (.taxo) is a
tab-separated fields:

relation_id <TAB> term <TAB> hypernym 

where:
- relation_id: is a relation identifier; 
- term: is a term of the taxonomy;
- hypernym: is a hypernym for the term. 

e.g

0<TAB>cat<TAB>animal
1<TAB>dog<TAB>animal
2<TAB>car<TAB>animal
....

The input files format for the system relation evaluation (.taxo.eval) is a
tab-separated fields:

relation_id <TAB> eval

where:
- relation_id: is a relation identifier; 
- eval: is an empty string if the relation is good, an "x" otherwise

e.g.
0<TAB>
1<TAB>
2<TAB>x
....


The input files format for the terminologies (.terms) is a
tab-separated fields:

term_id <TAB> term

where:
- term_id: is a term identifier; 
- term: is a domain term.

