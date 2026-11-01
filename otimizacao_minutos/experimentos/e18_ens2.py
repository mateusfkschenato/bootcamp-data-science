from modelos import *
fam=["HGB todas-TE","HGB todas-TE lr.05","HGB todas-TE leaves16","HGB todas-TE leaf20","HGB todas-TE it150"]
famS=["HGB S2 l2=5","HGB S2 lr.05","HGB S2 leaves16","HGB S2 leaf20","HGB S2 it150"]
ensemble("ENS família todas-TE (5)", fam)
ensemble("ENS família S2 (5)", famS)
ensemble("ENS 5 todas-TE + 5 S2", fam+famS)
ensemble("ENS todas-TE 2:1 S2", ["HGB todas-TE","HGB S2 l2=5"], pesos=[2,1])
