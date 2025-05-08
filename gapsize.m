function sz=gapsize(x)
% Calculates the gap size of a vector. 
%
% A gap is defined as the number of consequtive nans.
%
% USAGE: sz=gapsize(x)
% sz is same length as input. 
%
% example:
% x=rand(20,1);
% x(x>.5)=nan; 
% [x gapsize(x)]
%
% aslak grinsted 2010
x=~isnan(x);
hasdata=[0;find(x(:)); length(x)+1];
sz=zeros(size(x));
for ii=1:length(hasdata)-1
    ix=hasdata(ii)+1:hasdata(ii+1)-1;
    sz(ix)=length(ix);
end

end

