# Author: AJ Tyler
# Date: 25/11/21
module Similarity # This module computes the Jccard Index
export JacardIndex # This function returns the Jacard Index
"""
This function calculates the Jacard Index for two dictionaries,
with their keys being the unique elements of a multiset and the values are their frequencies

Inputs:
A: (Dictionary) Multiset A
B: (Dictionary) Multiset B

Outputs:
similarity: (Float64) The Jacard Index of multsets A & B
"""
function JacardIndex(A,B)
	intersection = 0.0 # Initialise the number of elements in the intersection of A & B
	union = 0.0 # Initialise the number of elements in the union of A & B
	for key in keys(A) # For all the unique elements of A
		intersection += min(get(A,key,0),get(B,key,0)) # Add the minimum number of occurances of the key across both A & B
		union += get(A,key,0) # Add all the elements of A
	end

	for key in keys(B) # For all the unique elements of B
		union += get(B,key,0) # Add all the elements of B
	end

	union -= intersection # union = A + B - intersection
	similarity = intersection/union # Comput the Jacard index
	return similarity # Return the Jacard Index
end

end # End of Module
